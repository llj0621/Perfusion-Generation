"""Visualize VAE reconstruction: middle 5 Z-slices, soft tissue window, multiple time frames."""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import random

import numpy as np
import torch
import matplotlib.pyplot as plt

from config import Config
from model.vae import SpatioTemporalVAE
from utils import EMA, load_checkpoint


def denormalize_to_hu(x):
    """Inverse of dataset normalization: [-1, 1] -> [0, 400] HU."""
    return (x + 1.0) * 200.0


def soft_tissue_window(hu):
    """Map [0, 400] HU to [0, 1] for display (matches training normalization window)."""
    return np.clip(hu / 400.0, 0, 1)


def load_model(ckpt_path, cfg, device):
    """Load model with EMA weights from checkpoint."""
    model = SpatioTemporalVAE(
        in_ch=cfg.in_ch, base_ch=cfg.base_ch, dec_base_ch=cfg.dec_base_ch,
        latent_ch=cfg.latent_ch, ch_mult=cfg.ch_mult, temporal_hidden=cfg.temporal_hidden,
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    # Prefer EMA weights if available
    if ckpt.get("ema") is not None:
        state = {k: v.to(device) for k, v in ckpt["ema"].items()}
        # Cast to model dtype
        dtype = next(model.parameters()).dtype
        state = {k: v.to(dtype) for k, v in state.items()}
        model.load_state_dict(state)
        print(f"Loaded EMA weights from {ckpt_path}")
    else:
        model.load_state_dict(ckpt["model"])
        print(f"Loaded model weights from {ckpt_path}")

    model.eval()
    return model


def load_subject_slice(cfg, subject_idx=0, z_indices=None):
    """Load raw data for a subject, return middle 5 Z-slices full temporal sequence.

    Returns: (n_slices, T=25, 1, H, W) normalized tensor, and raw HU array for display.
    """
    all_files = sorted([f for f in os.listdir(cfg.data_dir) if f.endswith(".npy")])

    # Use val split subjects (same split logic as dataset)
    rng = random.Random(cfg.seed)
    indices = list(range(len(all_files)))
    rng.shuffle(indices)
    n_train = int(len(all_files) * cfg.train_ratio)
    val_files = [all_files[i] for i in indices[n_train:]]

    fname = val_files[subject_idx % len(val_files)]
    path = os.path.join(cfg.data_dir, fname)
    print(f"Loading subject: {fname}")

    data = np.load(path, mmap_mode="r")  # (T=25, Z, H, W)
    T, Z, H, W = data.shape

    # Pick middle 5 Z-slices
    if z_indices is None:
        mid = Z // 2
        z_indices = list(range(mid - 2, mid + 3))
    print(f"Z-slices: {z_indices} (total Z={Z})")

    slices_norm = []
    slices_hu = []
    for z in z_indices:
        slc = np.array(data[:, z, :, :], dtype=np.float32)  # (T, H, W)
        # Normalize same as training: [0, 400] HU -> [-1, 1]
        slc_norm = np.clip(slc, 0.0, 400.0)
        slc_norm = slc_norm / 200.0 - 1.0
        slices_hu.append(np.clip(slc, 0.0, 400.0))
        slices_norm.append(slc_norm)

    # Stack: (n_slices, T, H, W) -> add channel -> (n_slices, T, 1, H, W)
    arr_norm = np.stack(slices_norm)[:, :, np.newaxis, :, :]
    arr_hu = np.stack(slices_hu)

    return torch.from_numpy(arr_norm.copy()), arr_hu, fname, z_indices


@torch.no_grad()
def reconstruct(model, x, device):
    """Run VAE reconstruction on full-resolution input.

    Args:
        x: (n_slices, T, 1, H, W) — each slice is a separate batch item
    Returns:
        recon_hu: (n_slices, T, H, W) in HU
    """
    recons = []
    for i in range(x.shape[0]):
        # Single slice: (1, T, 1, H, W)
        xi = x[i:i+1].to(device)
        recon, _, _ = model(xi, sample=False)
        recon_np = recon.cpu().numpy()[0, :, 0, :, :]  # (T, H, W)
        recons.append(denormalize_to_hu(recon_np))
    return np.stack(recons)  # (n_slices, T, H, W)


def plot_comparison(gt_hu, recon_hu, z_indices, time_frames, fname, save_path):
    """Plot GT vs Recon in soft tissue window.

    Rows: Z-slices (5)
    Columns: time frames, alternating GT / Recon
    """
    n_z = len(z_indices)
    n_t = len(time_frames)
    fig, axes = plt.subplots(n_z, n_t * 2, figsize=(3 * n_t * 2, 3 * n_z))

    for row, z in enumerate(range(n_z)):
        for col, t in enumerate(time_frames):
            gt_win = soft_tissue_window(gt_hu[z, t])
            recon_win = soft_tissue_window(recon_hu[z, t])

            # GT
            ax_gt = axes[row, col * 2]
            ax_gt.imshow(gt_win, cmap="gray", vmin=0, vmax=1)
            ax_gt.set_axis_off()
            if row == 0:
                ax_gt.set_title(f"GT t={t}", fontsize=9)
            if col == 0:
                ax_gt.set_ylabel(f"Z={z_indices[z]}", fontsize=9)

            # Recon
            ax_re = axes[row, col * 2 + 1]
            ax_re.imshow(recon_win, cmap="gray", vmin=0, vmax=1)
            ax_re.set_axis_off()
            if row == 0:
                ax_re.set_title(f"Recon t={t}", fontsize=9)

    fig.suptitle(f"VAE Reconstruction — {fname} (Window: 0-400 HU)", fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_difference(gt_hu, recon_hu, z_indices, time_frames, fname, save_path):
    """Plot absolute difference map (in HU) for each slice/frame."""
    n_z = len(z_indices)
    n_t = len(time_frames)
    fig, axes = plt.subplots(n_z, n_t, figsize=(3 * n_t, 3 * n_z))

    if n_z == 1:
        axes = axes[np.newaxis, :]
    if n_t == 1:
        axes = axes[:, np.newaxis]

    for row in range(n_z):
        for col, t in enumerate(time_frames):
            diff = np.abs(gt_hu[row, t] - recon_hu[row, t])
            ax = axes[row, col]
            im = ax.imshow(diff, cmap="hot", vmin=0, vmax=100)
            ax.set_axis_off()
            if row == 0:
                ax.set_title(f"t={t}", fontsize=9)
            if col == 0:
                ax.set_ylabel(f"Z={z_indices[row]}", fontsize=9)

    fig.suptitle(f"Absolute Error (HU) — {fname}", fontsize=11)
    fig.colorbar(im, ax=axes, shrink=0.6, label="HU")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize VAE reconstruction")
    parser.add_argument("--ckpt", type=str, default="./checkpoints/ckpt_final.pt", help="Checkpoint path")
    parser.add_argument("--subject", type=int, default=6, help="Val subject index")
    parser.add_argument("--frames", type=int, nargs="+", default=[0, 3, 6, 9, 12, 15, 18, 21, 24],
                        help="Time frames to visualize")
    parser.add_argument("--out_dir", type=str, default="./vis_final", help="Output directory")
    parser.add_argument("--data_dir", type=str, default="E:/CTP/data/rest", help="Override data_dir")
    args = parser.parse_args()

    cfg = Config()
    if args.data_dir:
        cfg.data_dir = args.data_dir

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = load_model(args.ckpt, cfg, device)

    # Load data
    x_norm, gt_hu, fname, z_indices = load_subject_slice(cfg, subject_idx=args.subject)

    # Reconstruct
    recon_hu = reconstruct(model, x_norm, device)

    # Clamp time frames to valid range
    T = gt_hu.shape[1]
    time_frames = [t for t in args.frames if t < T]

    # Plot
    os.makedirs(args.out_dir, exist_ok=True)
    base = os.path.splitext(fname)[0]

    plot_comparison(gt_hu, recon_hu, z_indices, time_frames, fname,
                    os.path.join(args.out_dir, f"{base}_comparison.png"))

    plot_difference(gt_hu, recon_hu, z_indices, time_frames, fname,
                    os.path.join(args.out_dir, f"{base}_error.png"))

    # Print per-slice PSNR
    for i, z in enumerate(z_indices):
        mse = np.mean((gt_hu[i] - recon_hu[i]) ** 2)
        psnr = 10 * np.log10(400 ** 2 / max(mse, 1e-10))  # HU range 400
        print(f"  Z={z}: MSE={mse:.1f} HU^2, PSNR={psnr:.1f} dB")


if __name__ == "__main__":
    main()
