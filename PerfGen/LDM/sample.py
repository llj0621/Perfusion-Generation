"""Generate CTP sequences from first frame using trained LDM (mask-conditioned).

Two modes:
  --mode original : use timestamps + HR from CSV for the subject (default)
  --mode custom   : user specifies custom values

Usage:
    python LDM/sample.py --ldm_ckpt LDM/checkpoints/ckpt_final.pt --subject 0
    python LDM/sample.py --ldm_ckpt LDM/checkpoints/ckpt_final.pt --subject 0 --mode custom --custom_hr 80
"""
import os
import sys
import json
import argparse

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

# Add PerfGen root to path so package-qualified imports resolve
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from LDM.config import LDMConfig
from LDM.model.unet3d import UNet3D
from LDM.model.flow_matching import FlowMatching
from LDM.mask_utils import find_crop_center, crop_offset, downsample_masks
from VAE.model.vae import SpatioTemporalVAE


def load_ldm(ckpt_path, cfg, device):
    """Load LDM with EMA weights."""
    model = UNet3D(
        latent_ch=cfg.latent_ch, base_ch=cfg.base_ch, ch_mult=cfg.ch_mult,
        t_emb_dim=cfg.t_emb_dim, attn_levels=cfg.attn_levels, num_res_blocks=cfg.num_res_blocks,
        num_mask_ch=cfg.num_mask_channels,
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if ckpt.get("ema") is not None:
        state = {k: v.to(device) for k, v in ckpt["ema"].items()}
        dtype = next(model.parameters()).dtype
        state = {k: v.to(dtype) for k, v in state.items()}
        model.load_state_dict(state)
        print(f"Loaded LDM EMA weights from {ckpt_path}")
    else:
        model.load_state_dict(ckpt["model"])
        print(f"Loaded LDM model weights from {ckpt_path}")

    model.eval()
    return model


def load_vae(ckpt_path, cfg, device):
    """Load VAE with EMA weights."""
    model = SpatioTemporalVAE(
        in_ch=cfg.vae_in_ch, base_ch=cfg.vae_base_ch, dec_base_ch=cfg.vae_dec_base_ch,
        latent_ch=cfg.vae_latent_ch, ch_mult=cfg.vae_ch_mult, temporal_hidden=cfg.vae_temporal_hidden,
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if ckpt.get("ema") is not None:
        state = {k: v.to(device) for k, v in ckpt["ema"].items()}
        dtype = next(model.parameters()).dtype
        state = {k: v.to(dtype) for k, v in state.items()}
        model.load_state_dict(state)
        print(f"Loaded VAE EMA weights from {ckpt_path}")
    else:
        model.load_state_dict(ckpt["model"])
        print(f"Loaded VAE model weights from {ckpt_path}")

    model.eval()
    return model


def normalize(x):
    """Clip to [0, 400] HU and normalize to [-1, 1]."""
    x = np.clip(x, 0.0, 400.0)
    return x / 200.0 - 1.0


def denormalize_to_hu(x):
    """Inverse: [-1, 1] -> [0, 400] HU."""
    return (x + 1.0) * 200.0


def soft_tissue_window(hu):
    """Map [0, 400] HU to [0, 1] for display."""
    return np.clip(hu / 400.0, 0, 1)


def get_conditions(cfg, subject_name, mode, custom_hr, custom_duration, custom_timestamps, device):
    """Get timestamps and heart_rate based on mode."""
    T = cfg.num_frames

    if mode == "original":
        df = pd.read_csv(cfg.metadata_csv)
        row = df[df["sample"] == subject_name].iloc[0]
        t_cols = [f"t{i}" for i in range(T)]
        timestamps = np.array([row[c] for c in t_cols], dtype=np.float32)
        hr = float(row["heart_rate_bpm"])
    else:
        hr = custom_hr
        if custom_timestamps is not None:
            timestamps = np.array(custom_timestamps, dtype=np.float32)
        else:
            timestamps = np.linspace(0, custom_duration, T, dtype=np.float32)

    timestamps_t = torch.from_numpy(timestamps).unsqueeze(0).to(device)
    heart_rate_t = torch.tensor([hr], dtype=torch.float32, device=device)
    return timestamps_t, heart_rate_t


def get_mask_tensor(cfg, subject_name, z_idx, y0, x0, patch_size, device):
    """Load or compute mask tensor for a single Z-slice.

    Tries pre-computed mask from latent_dir first, falls back to raw masks.
    Returns: (1, 5, 32, 32) tensor on device.
    """
    # Try pre-computed
    if cfg.latent_dir:
        mask_path = os.path.join(cfg.latent_dir, f"{subject_name}_z{z_idx:03d}_mask.npy")
        if os.path.exists(mask_path):
            mask = np.load(mask_path)
            return torch.from_numpy(mask).unsqueeze(0).to(device)

    # Fall back to computing from raw masks
    subj_mask_dir = os.path.join(cfg.mask_dir, subject_name)
    mask = downsample_masks(subj_mask_dir, z_idx, y0, x0, patch_size)
    return torch.from_numpy(mask).unsqueeze(0).to(device)


def get_crop_center(cfg, subject_name, Z, H, W, patch_size):
    """Get crop center from crop_centers.json or compute from masks."""
    if cfg.latent_dir:
        centers_path = os.path.join(cfg.latent_dir, "crop_centers.json")
        if os.path.exists(centers_path):
            with open(centers_path, "r") as f:
                centers = json.load(f)
            if subject_name in centers:
                info = centers[subject_name]
                return info["cy"], info["cx"], info["best_z"]

    subj_mask_dir = os.path.join(cfg.mask_dir, subject_name)
    return find_crop_center(subj_mask_dir, Z, H, W, patch_size)


@torch.no_grad()
def generate_sequence(ldm, vae, flow, first_frame_pixel, device, cfg,
                      timestamps, heart_rate, mask):
    """End-to-end: first frame pixel -> LDM generate -> VAE decode -> pixel sequence.

    Args:
        first_frame_pixel: (H, W) numpy array in HU
        mask: (1, 5, 32, 32) tensor on device
    Returns:
        generated_seq: (T=25, H, W) numpy array in HU
    """
    x = normalize(first_frame_pixel)
    x = torch.from_numpy(x[None, None, None, :, :]).to(device)
    mean, _ = vae.encode(x)
    z_cond = mean.squeeze(0).squeeze(1)

    shape = (1, cfg.latent_ch, cfg.num_frames, z_cond.shape[1], z_cond.shape[2])
    z_gen = flow.sample(ldm, z_cond.unsqueeze(0), shape, device,
                        timestamps=timestamps, heart_rate=heart_rate, mask=mask)

    z_refined = vae.temporal_refine(z_gen)
    recon = vae.decode(z_refined)

    recon_np = recon.cpu().numpy()[0, :, 0, :, :]
    return denormalize_to_hu(recon_np)


def plot_comparison(gt_seq, gen_seq, time_frames, fname, save_path):
    """Plot GT vs Generated for selected time frames."""
    n_t = len(time_frames)
    fig, axes = plt.subplots(2, n_t, figsize=(3 * n_t, 6))

    for col, t in enumerate(time_frames):
        gt_win = soft_tissue_window(gt_seq[t])
        gen_win = soft_tissue_window(gen_seq[t])

        axes[0, col].imshow(gt_win, cmap="gray", vmin=0, vmax=1)
        axes[0, col].set_axis_off()
        axes[0, col].set_title(f"GT t={t}", fontsize=9)

        axes[1, col].imshow(gen_win, cmap="gray", vmin=0, vmax=1)
        axes[1, col].set_axis_off()
        axes[1, col].set_title(f"Gen t={t}", fontsize=9)

    fig.suptitle(f"LDM Generation — {fname} (Window: 0-400 HU)", fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="LDM inference")
    parser.add_argument("--ldm_ckpt", type=str, required=True, help="LDM checkpoint")
    parser.add_argument("--subject_name", type=str, default=None)
    parser.add_argument("--subject", type=int, default=0, help="Val subject index")
    parser.add_argument("--z_slice", type=int, default=None, help="Z-slice index (default: best_z)")
    parser.add_argument("--frames", type=int, nargs="+", default=[0, 5, 10, 15, 20, 24])
    parser.add_argument("--out_dir", type=str, default="./LDM/samples")
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--mode", type=str, default="original", choices=["original", "custom"])
    parser.add_argument("--custom_hr", type=float, default=70.0)
    parser.add_argument("--custom_duration", type=float, default=36.0)
    parser.add_argument("--custom_timestamps", type=float, nargs="+", default=None)
    args = parser.parse_args()

    cfg = LDMConfig()
    if args.data_dir:
        cfg.raw_data_dir = args.data_dir

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ldm = load_ldm(args.ldm_ckpt, cfg, device)
    vae = load_vae(cfg.vae_ckpt, cfg, device)
    flow = FlowMatching(num_inference_steps=cfg.num_inference_steps)

    # Load val subject
    import random
    all_files = sorted([f for f in os.listdir(cfg.raw_data_dir) if f.endswith(".npy")])

    if args.subject_name is not None:
        fname = args.subject_name + ".npy"
        if fname not in all_files:
            raise FileNotFoundError(f"{fname} not found in {cfg.raw_data_dir}")
    else:
        rng = random.Random(cfg.seed)
        indices = list(range(len(all_files)))
        rng.shuffle(indices)
        n_train = int(len(all_files) * cfg.train_ratio)
        val_files = [all_files[i] for i in indices[n_train:]]
        fname = val_files[args.subject % len(val_files)]

    path = os.path.join(cfg.raw_data_dir, fname)
    subject_name = os.path.splitext(fname)[0]
    print(f"Loading subject: {fname}")

    data = np.load(path, mmap_mode="r")
    T, Z, H, W = data.shape
    ps = 256

    # Mask-centered crop
    cy, cx, best_z = get_crop_center(cfg, subject_name, Z, H, W, ps)
    y0 = crop_offset(cy, ps, H)
    x0 = crop_offset(cx, ps, W)

    z_idx = args.z_slice if args.z_slice is not None else best_z
    print(f"Z-slice: {z_idx} (best_z={best_z}, total Z={Z})")
    print(f"Crop center: cy={cy}, cx={cx}")

    gt_seq = np.array(data[:, z_idx, y0:y0 + ps, x0:x0 + ps], dtype=np.float32)

    # Get conditions
    timestamps, heart_rate = get_conditions(
        cfg, subject_name, args.mode, args.custom_hr, args.custom_duration,
        args.custom_timestamps, device)
    print(f"Mode: {args.mode}, HR: {heart_rate.item():.0f} bpm, "
          f"Duration: {timestamps[0, -1].item():.1f}s")

    # Get mask
    mask = get_mask_tensor(cfg, subject_name, z_idx, y0, x0, ps, device)

    # Generate from first frame
    first_frame = gt_seq[0]
    print("Generating sequence...")
    gen_seq = generate_sequence(ldm, vae, flow, first_frame, device, cfg,
                                timestamps, heart_rate, mask)

    # Plot
    os.makedirs(args.out_dir, exist_ok=True)
    base = os.path.splitext(fname)[0]
    plot_comparison(gt_seq, gen_seq, args.frames, fname,
                    os.path.join(args.out_dir, f"{base}_z{z_idx:03d}_comparison.png"))

    # Compute metrics
    mse = np.mean((gt_seq - gen_seq) ** 2)
    psnr = 10 * np.log10(400 ** 2 / max(mse, 1e-10))
    print(f"MSE: {mse:.1f} HU^2, PSNR: {psnr:.1f} dB")


if __name__ == "__main__":
    main()
