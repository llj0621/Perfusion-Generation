import os
import sys
import random
import math

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.amp import autocast

# Add project root to path so we can import shared utils
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ldm_v3.config import LDMConfig
from ldm_v3.dataset import LatentCTPDataset
from ldm_v3.model.unet3d import UNet3D
from ldm_v3.model.flow_matching import FlowMatching
from ldm_v3.losses import velocity_loss
from utils import EMA, save_checkpoint


def set_seed(seed, rank=0):
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)


def get_scheduler(optimizer, cfg, steps_per_epoch, scaled_lr):
    """Cosine annealing with linear warmup."""
    warmup_steps = cfg.warmup_epochs * steps_per_epoch
    total_steps = cfg.epochs * steps_per_epoch

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return max(cfg.min_lr / scaled_lr, 0.5 * (1 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def validate(model, ema, val_loader, flow, device, cfg):
    """Run validation with EMA weights."""
    raw_model = model.module if hasattr(model, "module") else model
    ema.apply(raw_model)
    raw_model.eval()

    total_loss = 0.0
    count = 0
    for batch in val_loader:
        z_full = batch["z_full"].to(device)
        z_cond = batch["z_cond"].to(device)
        timestamps = batch["timestamps"].to(device)
        heart_rate = batch["heart_rate"].to(device)
        mask = batch["mask"].to(device)
        B = z_full.shape[0]

        with autocast("cuda", dtype=torch.bfloat16, enabled=cfg.use_amp):
            noise = torch.randn_like(z_full)
            t = torch.rand(B, device=device)

            z_t = flow.add_noise(z_full, noise, t)
            v_target = flow.velocity_target(z_full, noise)

            cond_exp = z_cond.unsqueeze(2).expand_as(z_full)
            model_input = torch.cat([z_t, cond_exp], dim=1)

            v_pred = raw_model(model_input, t, timestamps=timestamps,
                               heart_rate=heart_rate, mask=mask)
            loss = velocity_loss(v_pred, v_target)

        total_loss += loss.item()
        count += 1

    ema.restore(raw_model)
    return total_loss / max(count, 1)


def train(cfg: LDMConfig):
    # DDP setup
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    set_seed(cfg.seed, rank)

    if rank == 0:
        os.makedirs(cfg.output_dir, exist_ok=True)
        log_f = open(cfg.log_file, "w", encoding="utf-8")

    scaled_lr = cfg.lr * math.sqrt(world_size)

    # Data
    train_ds = LatentCTPDataset(cfg, split="train")
    val_ds = LatentCTPDataset(cfg, split="val")

    # Save split info
    if rank == 0:
        train_ds.save_split(cfg.output_dir)
        val_ds.save_split(cfg.output_dir)
        print(f"Saved train/val split to {cfg.output_dir}")

    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, sampler=train_sampler,
                              num_workers=cfg.num_workers, pin_memory=True,
                              drop_last=True, persistent_workers=cfg.num_workers > 0)

    # Val loader: only used on rank 0, no DistributedSampler
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers, pin_memory=True,
                            persistent_workers=cfg.num_workers > 0) if rank == 0 else None

    if rank == 0:
        print(f"World size: {world_size}, Per-GPU batch: {cfg.batch_size}, "
              f"Effective batch: {cfg.batch_size * world_size}")
        print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")
        print(f"Steps per epoch: {len(train_loader)}")
        print(f"Base lr: {cfg.lr}, Scaled lr: {scaled_lr}")

    # Model
    model = UNet3D(
        latent_ch=cfg.latent_ch, base_ch=cfg.base_ch, ch_mult=cfg.ch_mult,
        t_emb_dim=cfg.t_emb_dim, attn_levels=cfg.attn_levels, num_res_blocks=cfg.num_res_blocks,
        num_mask_ch=cfg.num_mask_channels,
    ).to(device)

    flow = FlowMatching(num_inference_steps=cfg.num_inference_steps)

    if rank == 0:
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"UNet3D parameters: {n_params / 1e6:.2f}M")

    # EMA on rank 0 only
    ema = EMA(model, decay=cfg.ema_decay) if rank == 0 else None

    # Wrap with DDP
    model = DDP(model, device_ids=[local_rank])

    # Optimizer & scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=scaled_lr, weight_decay=cfg.weight_decay)
    scheduler = get_scheduler(optimizer, cfg, len(train_loader), scaled_lr)

    global_step = 0

    for epoch in range(cfg.epochs):
        train_sampler.set_epoch(epoch)
        model.train()

        for step, batch in enumerate(train_loader):
            z_full = batch["z_full"].to(device)       # (B, 16, 25, 32, 32)
            z_cond = batch["z_cond"].to(device)       # (B, 16, 32, 32)
            timestamps = batch["timestamps"].to(device)  # (B, 25)
            heart_rate = batch["heart_rate"].to(device)  # (B,)
            mask = batch["mask"].to(device)            # (B, 5, 32, 32)
            B = z_full.shape[0]

            optimizer.zero_grad(set_to_none=True)
            with autocast("cuda", dtype=torch.bfloat16, enabled=cfg.use_amp):
                noise = torch.randn_like(z_full)
                t = torch.rand(B, device=device)

                z_t = flow.add_noise(z_full, noise, t)
                v_target = flow.velocity_target(z_full, noise)

                cond_exp = z_cond.unsqueeze(2).expand_as(z_full)
                model_input = torch.cat([z_t, cond_exp], dim=1)  # (B, 32, 25, 32, 32)

                v_pred = model(model_input, t, timestamps=timestamps,
                               heart_rate=heart_rate, mask=mask)
                loss = velocity_loss(v_pred, v_target)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            scheduler.step()
            if ema is not None:
                ema.update(model.module)

            global_step += 1

            if rank == 0 and global_step % cfg.log_interval == 0:
                lr = optimizer.param_groups[0]["lr"]
                msg = (f"[Epoch {epoch+1}/{cfg.epochs}] Step {global_step} | "
                       f"loss={loss.item():.6f} lr={lr:.2e}")
                print(msg)
                log_f.write(msg + "\n")
                log_f.flush()

        # Validation
        if (epoch + 1) % cfg.val_interval == 0 and rank == 0:
            val_loss = validate(model, ema, val_loader, flow, device, cfg)
            val_msg = f"  [Val] epoch={epoch+1} | loss={val_loss:.6f}"
            print(val_msg)
            log_f.write(val_msg + "\n")
            log_f.flush()

        # Save checkpoint
        if (epoch + 1) % cfg.save_interval == 0 and rank == 0:
            ckpt_path = os.path.join(cfg.output_dir, f"ckpt_epoch{epoch+1:04d}.pt")
            save_checkpoint(ckpt_path, model.module, ema, optimizer, scheduler, epoch + 1)
            print(f"  Saved checkpoint: {ckpt_path}")

        dist.barrier()

    # Save final
    if rank == 0:
        save_checkpoint(os.path.join(cfg.output_dir, "ckpt_final.pt"),
                        model.module, ema, optimizer, scheduler, cfg.epochs)
        log_f.close()
        print("Training complete.")

    dist.destroy_process_group()


if __name__ == "__main__":
    train(LDMConfig())
