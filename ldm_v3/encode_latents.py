"""Pre-encode all CTP data to VAE latents + downsampled masks for LDM v3 training.

Key difference from v2: mask-centered cropping instead of image-center cropping.

Usage:
    python ldm_v3/encode_latents.py
    python ldm_v3/encode_latents.py --data_dir E:/CTP/data/rest --out_dir E:/CTP/data/rest_latent_v3
"""
import os
import sys
import json
import argparse

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import Config
from model.vae import SpatioTemporalVAE
from ldm_v3.mask_utils import find_crop_center, crop_offset, downsample_masks


def load_vae(ckpt_path, cfg, device):
    """Load VAE with EMA weights."""
    model = SpatioTemporalVAE(
        in_ch=cfg.in_ch, base_ch=cfg.base_ch, dec_base_ch=cfg.dec_base_ch,
        latent_ch=cfg.latent_ch, ch_mult=cfg.ch_mult, temporal_hidden=cfg.temporal_hidden,
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if ckpt.get("ema") is not None:
        state = {k: v.to(device) for k, v in ckpt["ema"].items()}
        dtype = next(model.parameters()).dtype
        state = {k: v.to(dtype) for k, v in state.items()}
        model.load_state_dict(state)
        print(f"Loaded EMA weights from {ckpt_path}")
    else:
        model.load_state_dict(ckpt["model"])
        print(f"Loaded model weights from {ckpt_path}")

    model.eval()
    return model


def normalize(x):
    """Clip to [0, 400] HU and normalize to [-1, 1]."""
    x = np.clip(x, 0.0, 400.0)
    return x / 200.0 - 1.0


@torch.no_grad()
def encode_slice(vae, slc, device, batch_t=5):
    """Encode a single Z-slice (T=25 frames) to latent space.

    Args:
        slc: (T, 1, H, W) normalized numpy array
        batch_t: number of frames per forward pass
    Returns:
        latent: (C, T, h, w) numpy float16
    """
    T = slc.shape[0]
    latents = []
    for t0 in range(0, T, batch_t):
        t1 = min(t0 + batch_t, T)
        x = torch.from_numpy(slc[t0:t1]).unsqueeze(0).to(device)
        mean, _ = vae.encode(x)
        latents.append(mean.squeeze(0).cpu())

    latent = torch.cat(latents, dim=1)
    return latent.numpy().astype(np.float16)


def main():
    parser = argparse.ArgumentParser(description="Pre-encode CTP data with mask-centered crop")
    parser.add_argument("--data_dir", type=str, default=r"E:\CTP\data\rest")
    parser.add_argument("--out_dir", type=str, default=r"E:\CTP\data\rest_latent_v3")
    parser.add_argument("--mask_dir", type=str, default=r"E:\CTP\data\heartchambers_seg_npy")
    parser.add_argument("--vae_ckpt", type=str, default=r"D:\Project\LiLJ\CTP_v2\checkpoints\ckpt_final.pt")
    parser.add_argument("--z_trim", type=int, default=20)
    parser.add_argument("--patch_size", type=int, default=256)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = Config()
    vae = load_vae(args.vae_ckpt, cfg, device)

    all_files = sorted([f for f in os.listdir(args.data_dir) if f.endswith(".npy")])
    print(f"Found {len(all_files)} subjects")

    ps = args.patch_size
    crop_centers = {}
    total_slices = 0
    skipped = 0

    for fname in tqdm(all_files, desc="Subjects"):
        path = os.path.join(args.data_dir, fname)
        data = np.load(path, mmap_mode="r")  # (T=25, Z, H, W)
        T, Z, H, W = data.shape
        subject_name = os.path.splitext(fname)[0]

        # Mask-centered crop
        subj_mask_dir = os.path.join(args.mask_dir, subject_name)
        cy, cx, best_z = find_crop_center(subj_mask_dir, Z, H, W, ps)
        y0 = crop_offset(cy, ps, H)
        x0 = crop_offset(cx, ps, W)
        crop_centers[subject_name] = {"cy": cy, "cx": cx, "best_z": best_z, "y0": y0, "x0": x0}

        for z in range(args.z_trim, Z - args.z_trim):
            latent_name = f"{subject_name}_z{z:03d}.npy"
            mask_name = f"{subject_name}_z{z:03d}_mask.npy"
            latent_path = os.path.join(args.out_dir, latent_name)
            mask_path = os.path.join(args.out_dir, mask_name)

            if os.path.exists(latent_path) and os.path.exists(mask_path):
                skipped += 1
                continue

            # Encode latent
            slc = np.array(data[:, z, y0:y0 + ps, x0:x0 + ps], dtype=np.float32)
            slc = normalize(slc)
            slc = slc[:, np.newaxis, :, :]  # (T, 1, H, W)
            latent = encode_slice(vae, slc, device)
            np.save(latent_path, latent)

            # Downsample masks
            mask = downsample_masks(subj_mask_dir, z, y0, x0, ps)
            np.save(mask_path, mask)

            total_slices += 1

    # Save crop centers
    centers_path = os.path.join(args.out_dir, "crop_centers.json")
    with open(centers_path, "w", encoding="utf-8") as f:
        json.dump(crop_centers, f, indent=2, ensure_ascii=False)

    print(f"Done. Encoded {total_slices} slices, skipped {skipped} existing.")
    print(f"Crop centers saved to {centers_path}")


if __name__ == "__main__":
    main()
