"""Visualize LDM v3 generation: volume-level TIC curves.

For selected subjects, generate full 25-frame sequences across multiple Z-slices,
then compute volume-averaged TIC per mask region (aorta, chambers) and plot GT vs Gen.

Usage:
    python ldm_v3/visualize_tic.py --subject_name SUBJ_02B9E42A
    python ldm_v3/visualize_tic.py --subject_name SUBJ_02B9E42A --mode custom --custom_hr 80
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import json
import sys
import argparse
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


def normalize(x):
    x = np.clip(x, 0.0, 400.0)
    return x / 200.0 - 1.0


def denormalize_to_hu(x):
    return (x + 1.0) * 200.0


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
    else:
        model.load_state_dict(ckpt["model"])
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
    else:
        model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def get_crop_center(cfg, subject_name, Z, H, W, patch_size=256):
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


def get_unet_mask(cfg, subject_name, z_idx, y0, x0, patch_size, device):
    if cfg.latent_dir:
        mask_path = os.path.join(cfg.latent_dir, f"{subject_name}_z{z_idx:03d}_mask.npy")
        if os.path.exists(mask_path):
            mask = np.load(mask_path)
            return torch.from_numpy(mask).unsqueeze(0).to(device)

    subj_mask_dir = os.path.join(cfg.mask_dir, subject_name)
    mask = downsample_masks(subj_mask_dir, z_idx, y0, x0, patch_size)
    return torch.from_numpy(mask).unsqueeze(0).to(device)


def load_subject_full(cfg, subject_name, patch_size=256, n_slices=5, z_interval=3):
    """Load subject data and masks for multiple Z-slices.

    Returns:
        gt_hu: (n_slices, T=25, H, W) in HU [0, 400]
        masks: dict[region_name] -> (n_slices, H, W) bool
        z_indices: list of Z indices
        fname: filename
        y0, x0: crop offsets
    """
    fname = subject_name + ".npy"
    path = os.path.join(cfg.raw_data_dir, fname)
    data = np.load(path, mmap_mode="r")  # (T=25, Z, H, W)
    T, Z, H, W = data.shape

    cy, cx, best_z = get_crop_center(cfg, subject_name, Z, H, W, patch_size)

    half_n = n_slices // 2
    z_indices = [best_z + (i - half_n) * z_interval for i in range(n_slices)]
    z_indices = [z for z in z_indices if cfg.z_trim <= z < Z - cfg.z_trim]

    ps = patch_size
    y0 = crop_offset(cy, ps, H)
    x0 = crop_offset(cx, ps, W)

    slices_hu = []
    for z in z_indices:
        slc = np.array(data[:, z, y0:y0 + ps, x0:x0 + ps], dtype=np.float32)
        slices_hu.append(np.clip(slc, 0.0, 400.0))

    # Load masks for all z_indices: dict[name] -> (n_slices, H, W) bool
    subj_mask_dir = os.path.join(cfg.mask_dir, subject_name)
    masks = {}
    for name in MASK_NAMES:
        p = Path(subj_mask_dir) / f"{name}.npy"
        if not p.exists():
            continue
        m = np.load(str(p), mmap_mode="r")
        m_slices = np.stack([m[z, y0:y0 + ps, x0:x0 + ps] for z in z_indices])
        if m_slices.any():
            masks[name] = m_slices > 0

    print(f"Subject: {fname}, best_z={best_z}, z_indices={z_indices}, crop=({y0},{x0})")
    return np.stack(slices_hu), masks, z_indices, fname, y0, x0


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
    x = normalize(first_frame_hu)
    x = torch.from_numpy(x[None, None, None, :, :]).to(device)
    mean, _ = vae.encode(x)
    z_cond = mean.squeeze(0).squeeze(1)
    shape = (1, cfg.latent_ch, cfg.num_frames, z_cond.shape[1], z_cond.shape[2])
    z_gen = flow.sample(ldm, z_cond.unsqueeze(0), shape, device,
                        timestamps=timestamps, heart_rate=heart_rate, mask=mask)
    z_refined = vae.temporal_refine(z_gen)
    recon = vae.decode(z_refined)
    return denormalize_to_hu(recon.cpu().numpy()[0, :, 0, :, :])


def plot_volume_tic(gt_curves, gen_curves, title, save_path):
    names = [n for n in MASK_NAMES if n in gt_curves] + ["global"]
    n_plots = len(names)
    ncols = 3
    nrows = (n_plots + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.array(axes).flatten()
    time_axis = np.arange(len(next(iter(gt_curves.values()))))

    for idx, name in enumerate(names):
        ax = axes[idx]
        ax.plot(time_axis, gt_curves[name], "b-o", markersize=3, label="GT")
        ax.plot(time_axis, gen_curves[name], "r--s", markersize=3, label="Gen")
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("Frame")
        ax.set_ylabel("HU")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    for idx in range(n_plots, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def main():
    with open(r'D:\Project\LiLJ\CTP_v2\ldm_v3\checkpoints\val_split.json', 'r') as f:
        data = json.load(f)

    for subj in data["subjects"]:
        print(subj)
        parser = argparse.ArgumentParser(description="Volume-level TIC visualization")
        parser.add_argument("--ldm_ckpt", type=str,
                            default=r"D:\Project\LiLJ\CTP_v2\ldm_v3\checkpoints\ckpt_final.pt")
        parser.add_argument("--subject_name", type=str, default=subj)
        parser.add_argument("--out_dir", type=str,
                            default=r"D:\Project\LiLJ\CTP_v2\ldm_v3\vis")
        parser.add_argument("--data_dir", type=str, default=None)
        parser.add_argument("--steps", type=int, default=None)
        parser.add_argument("--n_slices", type=int, default=50)
        parser.add_argument("--z_interval", type=int, default=3)
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

        gt_hu, masks, z_indices, fname, y0, x0 = load_subject_full(
            cfg, args.subject_name, n_slices=args.n_slices, z_interval=args.z_interval)
        base = os.path.splitext(fname)[0]

        timestamps, heart_rate = get_conditions(
            cfg, base, args.mode, args.custom_hr, args.custom_duration,
            args.custom_timestamps, device)
        print(f"Mode: {args.mode}, HR: {heart_rate.item():.0f} bpm, "
              f"Duration: {timestamps[0, -1].item():.1f}s")

        # Generate all slices
        n_slices = gt_hu.shape[0]
        gen_all = []
        for i, z in enumerate(z_indices):
            mask = get_unet_mask(cfg, base, z, y0, x0, 256, device)
            gen = generate_from_first_frame(ldm, vae, flow, gt_hu[i, 0], device, cfg,
                                            timestamps, heart_rate, mask)
            gen_all.append(gen)
            print(f"  Z={z} done ({i+1}/{n_slices})")
        gen_all = np.stack(gen_all)

        # Compute volume-averaged TIC per mask region
        gt_curves, gen_curves = {}, {}
        for name, m in masks.items():
            m_exp = m[:, np.newaxis, :, :]
            n_voxels = m.sum()
            gt_curves[name] = (gt_hu * m_exp).sum(axis=(0, 2, 3)) / n_voxels
            gen_curves[name] = (gen_all * m_exp).sum(axis=(0, 2, 3)) / n_voxels

        gt_curves["global"] = gt_hu.mean(axis=(0, 2, 3))
        gen_curves["global"] = gen_all.mean(axis=(0, 2, 3))

        os.makedirs(args.out_dir, exist_ok=True)
        save_path = os.path.join(args.out_dir, f"{base}_volume_tic.png")
        plot_volume_tic(gt_curves, gen_curves, f"Volume TIC — {base}", save_path)


if __name__ == "__main__":
    main()
