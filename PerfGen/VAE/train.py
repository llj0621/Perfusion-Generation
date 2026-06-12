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

# Add PerfGen root to path so package-qualified imports resolve
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from VAE.config import Config
from VAE.dataset import CTPSliceDataset
from VAE.model.vae import SpatioTemporalVAE
from VAE.losses import compute_loss
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
def validate(model, ema, val_loader, device, cfg):
    """Run validation with EMA weights. Uses raw model directly (not DDP)."""
    raw_model = model.module if hasattr(model, "module") else model
    ema.apply(raw_model)
    raw_model.eval()

    totals = {}
    count = 0
    for batch in val_loader:
        x = batch.to(device)
        with autocast("cuda", dtype=torch.bfloat16, enabled=cfg.use_amp):
            recon, mean, logvar = raw_model(x, sample=False)
            _, loss_dict = compute_loss(recon, x, mean, logvar,
                                        cfg.w_recon, cfg.w_ssim, cfg.w_temporal, cfg.w_kl)
        for k, v in loss_dict.items():
            totals[k] = totals.get(k, 0.0) + v.item()
        count += 1

    ema.restore(raw_model)
    return {k: v / max(count, 1) for k, v in totals.items()}


def train(cfg: Config):
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

    # Scale lr by world size (linear scaling rule)
    scaled_lr = cfg.lr * world_size

    # Data
    train_ds = CTPSliceDataset(cfg, split="train")
    val_ds = CTPSliceDataset(cfg, split="val")

    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, sampler=train_sampler,
                              num_workers=cfg.num_workers, pin_memory=True,
                              drop_last=True, persistent_workers=cfg.num_workers > 0)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, sampler=val_sampler,
                            num_workers=cfg.num_workers, pin_memory=True,
                            persistent_workers=cfg.num_workers > 0)

    if rank == 0:
        print(f"World size: {world_size}, Per-GPU batch: {cfg.batch_size}, "
              f"Effective batch: {cfg.batch_size * world_size}")
        print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")
        print(f"Steps per epoch: {len(train_loader)}")
        print(f"Base lr: {cfg.lr}, Scaled lr: {scaled_lr}")

    # Model
    model = SpatioTemporalVAE(
        in_ch=cfg.in_ch, base_ch=cfg.base_ch, dec_base_ch=cfg.dec_base_ch,
        latent_ch=cfg.latent_ch, ch_mult=cfg.ch_mult, temporal_hidden=cfg.temporal_hidden,
    ).to(device)

    if rank == 0:
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model parameters: {n_params / 1e6:.2f}M")

    # EMA on raw model (rank 0 only to save memory)
    ema = EMA(model, decay=cfg.ema_decay) if rank == 0 else None

    # Wrap with DDP
    model = DDP(model, device_ids=[local_rank])

    # Optimizer & scheduler (BF16 doesn't need GradScaler)
    optimizer = torch.optim.AdamW(model.parameters(), lr=scaled_lr, weight_decay=cfg.weight_decay)
    scheduler = get_scheduler(optimizer, cfg, len(train_loader), scaled_lr)

    global_step = 0

    for epoch in range(cfg.epochs):
        train_sampler.set_epoch(epoch)
        model.train()

        for step, batch in enumerate(train_loader):
            x = batch.to(device)

            optimizer.zero_grad(set_to_none=True)
            with autocast("cuda", dtype=torch.bfloat16, enabled=cfg.use_amp):
                recon, mean, logvar = model(x)
                total_loss, loss_dict = compute_loss(recon, x, mean, logvar,
                                                     cfg.w_recon, cfg.w_ssim, cfg.w_temporal, cfg.w_kl)

            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            scheduler.step()
            if ema is not None:
                ema.update(model.module)

            global_step += 1

            if rank == 0 and global_step % cfg.log_interval == 0:
                lr = optimizer.param_groups[0]["lr"]
                msg = (f"[Epoch {epoch+1}/{cfg.epochs}] Step {global_step} | "
                       f"loss={loss_dict['total']:.4f} recon={loss_dict['recon']:.4f} "
                       f"ssim={loss_dict['ssim']:.4f} temp={loss_dict['temporal']:.4f} "
                       f"kl={loss_dict['kl']:.2f} lr={lr:.2e}")
                print(msg)
                log_f.write(msg + "\n")
                log_f.flush()

        # Validation (rank 0 only with EMA)
        if (epoch + 1) % cfg.val_interval == 0 and rank == 0:
            val_losses = validate(model, ema, val_loader, device, cfg)
            val_msg = (f"  [Val] epoch={epoch+1} | " +
                       " ".join(f"{k}={v:.4f}" for k, v in val_losses.items()))
            print(val_msg)
            log_f.write(val_msg + "\n")
            log_f.flush()

        # Save checkpoint (rank 0 only)
        if (epoch + 1) % cfg.save_interval == 0 and rank == 0:
            ckpt_path = os.path.join(cfg.output_dir, f"ckpt_epoch{epoch+1:04d}.pt")
            save_checkpoint(ckpt_path, model.module, ema, optimizer, scheduler, epoch + 1)
            print(f"  Saved checkpoint: {ckpt_path}")

        # Barrier to sync all ranks before next epoch
        dist.barrier()

    # Save final (rank 0)
    if rank == 0:
        save_checkpoint(os.path.join(cfg.output_dir, "ckpt_final.pt"),
                        model.module, ema, optimizer, scheduler, cfg.epochs)
        log_f.close()
        print("Training complete.")

    dist.destroy_process_group()


if __name__ == "__main__":
    train(Config())
