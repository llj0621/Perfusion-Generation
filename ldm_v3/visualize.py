"""Visualize LDM v3 generation: first frame conditioned → full 25-frame sequence.

Two modes:
  --mode original : use timestamps + HR from CSV for the subject (default)
  --mode custom   : user specifies custom values

Outputs per Z-slice (each in its own subfolder):
  1. gt_full_seq.png      — GT 25-frame sequence (single row)
  2. gen_full_seq.png     — Generated 25-frame sequence (single row)
  3. comparison.png       — GT vs Gen for selected time frames (2 rows)
  4. tic.png              — Time-Intensity Curves per mask region

Usage:
    python ldm_v3/visualize.py --ldm_ckpt ldm_v3/checkpoints/ckpt_final.pt --subject_name SUBJ_02B9E42A
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import argparse
import json
import random

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ldm_v3.config import LDMConfig
from ldm_v3.model.unet3d import UNet3D
from ldm_v3.model.flow_matching import FlowMatching
from ldm_v3.mask_utils import MASK_NAMES, find_crop_center, crop_offset, downsample_masks
from model.vae import SpatioTemporalVAE


def denormalize_to_hu(x):
    return (x + 1.0) * 200.0


def normalize(x):
    x = np.clip(x, 0.0, 400.0)
    return x / 200.0 - 1.0


def soft_tissue_window(hu):
    return np.clip(hu / 400.0, 0, 1)


def load_ldm(ckpt_path, cfg, device):
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


def get_crop_center(cfg, subject_name, Z, H, W, patch_size):
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


def load_subject(cfg, subject_name=None, subject_idx=0, z_indices=None,
                 patch_size=256, n_slices=3, z_interval=3):
    """Load raw data for a subject, return multiple Z-slices with mask-centered crop.

    Returns:
        gt_hu: (n_slices, T=25, H, W) in HU [0, 400]
        fname: filename
        z_indices: list of Z indices used
        mask_slices: dict[z] -> dict[region_name] -> (H, W) bool array
        y0, x0: crop offsets
    """
    all_files = sorted([f for f in os.listdir(cfg.raw_data_dir) if f.endswith(".npy")])

    if subject_name is not None:
        fname = subject_name + ".npy"
        if fname not in all_files:
            raise FileNotFoundError(f"{fname} not found in {cfg.raw_data_dir}")
    else:
        rng = random.Random(cfg.seed)
        indices = list(range(len(all_files)))
        rng.shuffle(indices)
        n_train = int(len(all_files) * cfg.train_ratio)
        val_files = [all_files[i] for i in indices[n_train:]]
        fname = val_files[subject_idx % len(val_files)]

    path = os.path.join(cfg.raw_data_dir, fname)
    print(f"Loading subject: {fname}")

    data = np.load(path, mmap_mode="r")
    T, Z, H, W = data.shape

    sample = os.path.splitext(fname)[0]
    cy, cx, best_z = get_crop_center(cfg, sample, Z, H, W, patch_size)

    if z_indices is None:
        half_n = n_slices // 2
        z_indices = [best_z + (i - half_n) * z_interval for i in range(n_slices)]
        z_indices = [z for z in z_indices if cfg.z_trim <= z < Z - cfg.z_trim]
    print(f"Z-slices: {z_indices} (best_z={best_z}, total Z={Z})")
    print(f"Crop center: cy={cy}, cx={cx}")

    ps = patch_size
    y0 = crop_offset(cy, ps, H)
    x0 = crop_offset(cx, ps, W)

    slices_hu = []
    for z in z_indices:
        slc = np.array(data[:, z, y0:y0 + ps, x0:x0 + ps], dtype=np.float32)
        slices_hu.append(np.clip(slc, 0.0, 400.0))

    mask_slices = load_mask_slices(cfg.mask_dir, sample, z_indices, y0, x0, ps)

    return np.stack(slices_hu), fname, z_indices, mask_slices, y0, x0


def load_mask_slices(mask_dir, subject_name, z_indices, y0, x0, patch_size):
    """Load mask slices cropped to match data patches.

    Returns:
        dict[z] -> dict[region_name] -> (H, W) bool array
    """
    subj_mask_dir = os.path.join(mask_dir, subject_name)
    mask_slices = {}
    for z in z_indices:
        mask_slices[z] = {}
        for name in MASK_NAMES:
            p = Path(subj_mask_dir) / f"{name}.npy"
            if not p.exists():
                continue
            m = np.load(str(p), mmap_mode="r")
            crop = m[z, y0:y0 + patch_size, x0:x0 + patch_size]
            if crop.any():
                mask_slices[z][name] = crop > 0
    return mask_slices


def get_unet_mask(cfg, subject_name, z_idx, y0, x0, patch_size, device):
    """Get downsampled mask tensor for UNet input."""
    if cfg.latent_dir:
        mask_path = os.path.join(cfg.latent_dir, f"{subject_name}_z{z_idx:03d}_mask.npy")
        if os.path.exists(mask_path):
            mask = np.load(mask_path)
            return torch.from_numpy(mask).unsqueeze(0).to(device)

    subj_mask_dir = os.path.join(cfg.mask_dir, subject_name)
    mask = downsample_masks(subj_mask_dir, z_idx, y0, x0, patch_size)
    return torch.from_numpy(mask).unsqueeze(0).to(device)


def get_conditions(cfg, subject_name, mode, custom_hr, custom_duration, custom_timestamps, device):
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


@torch.no_grad()
def generate_from_first_frame(ldm, vae, flow, first_frame_hu, device, cfg,
                              timestamps, heart_rate, mask):
    """First frame pixel (H, W) in HU → LDM generate → VAE decode → (T, H, W) in HU."""
    x = normalize(first_frame_hu)
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


def plot_single_row_sequence(seq_hu, title, save_path):
    T = seq_hu.shape[0]
    fig, axes = plt.subplots(1, T, figsize=(1.5 * T, 1.8))

    for t in range(T):
        win = soft_tissue_window(seq_hu[t])
        axes[t].imshow(win, cmap="gray", vmin=0, vmax=1)
        axes[t].set_axis_off()
        axes[t].set_title(f"{t}", fontsize=7)

    fig.suptitle(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_comparison(gt_hu, gen_hu, time_frames, title, save_path):
    n_t = len(time_frames)
    fig, axes = plt.subplots(2, n_t, figsize=(3 * n_t, 6))

    for col, t in enumerate(time_frames):
        gt_win = soft_tissue_window(gt_hu[t])
        gen_win = soft_tissue_window(gen_hu[t])

        axes[0, col].imshow(gt_win, cmap="gray", vmin=0, vmax=1)
        axes[0, col].set_axis_off()
        axes[0, col].set_title(f"GT t={t}", fontsize=9)

        axes[1, col].imshow(gen_win, cmap="gray", vmin=0, vmax=1)
        axes[1, col].set_axis_off()
        axes[1, col].set_title(f"Gen t={t}", fontsize=9)

    fig.suptitle(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_tic(gt_hu, gen_hu, title, save_path, masks_z=None):
    T, H, W = gt_hu.shape
    time_axis = np.arange(T)

    roi_names = [n for n in MASK_NAMES if masks_z and n in masks_z]
    n_plots = len(roi_names) + 1
    ncols = 3
    nrows = (n_plots + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.array(axes).flatten()

    for idx, name in enumerate(roi_names):
        m = masks_z[name]
        gt_curve = gt_hu[:, m].mean(axis=1)
        gen_curve = gen_hu[:, m].mean(axis=1)

        ax = axes[idx]
        ax.plot(time_axis, gt_curve, "b-o", markersize=3, label="GT")
        ax.plot(time_axis, gen_curve, "r--s", markersize=3, label="Gen")
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("Frame")
        ax.set_ylabel("HU")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    ax = axes[len(roi_names)]
    gt_global = gt_hu.mean(axis=(1, 2))
    gen_global = gen_hu.mean(axis=(1, 2))
    ax.plot(time_axis, gt_global, "b-o", markersize=3, label="GT")
    ax.plot(time_axis, gen_global, "r--s", markersize=3, label="Gen")
    ax.set_title("Global Mean", fontsize=9)
    ax.set_xlabel("Frame")
    ax.set_ylabel("HU")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    for idx in range(len(roi_names) + 1, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize LDM v3 generation")
    parser.add_argument("--ldm_ckpt", type=str,
                        default=r"D:\Project\LiLJ\CTP_v2\ldm_v3\checkpoints\ckpt_final.pt")
    parser.add_argument("--subject_name", type=str, default="SUBJ_02B9E42A")
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--frames", type=int, nargs="+", default=[0, 5, 10, 15, 20, 24])
    parser.add_argument("--out_dir", type=str,
                        default=r"D:\Project\LiLJ\CTP_v2\ldm_v3\vis")
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--mode", type=str, default="original", choices=["original", "custom"])
    parser.add_argument("--custom_hr", type=float, default=70.0)
    parser.add_argument("--custom_duration", type=float, default=36.0)
    parser.add_argument("--custom_timestamps", type=float, nargs="+", default=None)
    args = parser.parse_args()

    cfg = LDMConfig()
    if args.data_dir:
        cfg.raw_data_dir = args.data_dir
    if args.steps:
        cfg.num_inference_steps = args.steps

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ldm = load_ldm(args.ldm_ckpt, cfg, device)
    vae = load_vae(cfg.vae_ckpt, cfg, device)
    flow = FlowMatching(num_inference_steps=cfg.num_inference_steps)

    gt_hu, fname, z_indices, mask_slices, y0, x0 = load_subject(
        cfg, subject_name=args.subject_name, subject_idx=args.subject)
    n_slices, T = gt_hu.shape[0], gt_hu.shape[1]
    base = os.path.splitext(fname)[0]
    time_frames = [t for t in args.frames if t < T]

    subject_name = base
    timestamps, heart_rate = get_conditions(
        cfg, subject_name, args.mode, args.custom_hr, args.custom_duration,
        args.custom_timestamps, device)
    print(f"Mode: {args.mode}, HR: {heart_rate.item():.0f} bpm, "
          f"Duration: {timestamps[0, -1].item():.1f}s")

    print(f"Generating {n_slices} slices...")
    for i, z in enumerate(z_indices):
        mask = get_unet_mask(cfg, subject_name, z, y0, x0, 256, device)
        first_frame = gt_hu[i, 0]
        gen = generate_from_first_frame(ldm, vae, flow, first_frame, device, cfg,
                                        timestamps, heart_rate, mask)
        print(f"  Z={z} done")

        sub_dir = os.path.join(args.out_dir, f"{base}_z{z:03d}")
        os.makedirs(sub_dir, exist_ok=True)

        plot_single_row_sequence(
            gt_hu[i], f"GT — {fname} Z={z}", os.path.join(sub_dir, "gt_full_seq.png"))

        plot_single_row_sequence(
            gen, f"Generated — {fname} Z={z}", os.path.join(sub_dir, "gen_full_seq.png"))

        plot_comparison(
            gt_hu[i], gen, time_frames, f"GT vs Gen — {fname} Z={z}",
            os.path.join(sub_dir, "comparison.png"))

        plot_tic(
            gt_hu[i], gen, f"TIC — {fname} Z={z}",
            os.path.join(sub_dir, "tic.png"), masks_z=mask_slices.get(z, {}))


if __name__ == "__main__":
    main()
