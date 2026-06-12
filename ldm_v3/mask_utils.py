"""Shared mask utilities for ldm_v3."""
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


MASK_NAMES = ("aorta", "heart_atrium_left", "heart_atrium_right",
              "heart_ventricle_left", "heart_ventricle_right")


def find_crop_center(mask_dir, Z, H, W, patch_size=256):
    """Find the Z-slice with max combined mask area and return crop center.

    Args:
        mask_dir: path to subject's mask folder (contains aorta.npy, etc.)
        Z, H, W: volume dimensions
        patch_size: crop size

    Returns:
        (cy, cx, best_z) — centroid coordinates and best Z-slice index.
        Falls back to image center if no masks found.
    """
    if not os.path.isdir(mask_dir):
        return H // 2, W // 2, Z // 2

    combined = np.zeros((Z, H, W), dtype=np.float32)
    for p in sorted(Path(mask_dir).glob("*.npy")):
        combined += np.load(str(p), mmap_mode="r")

    if combined.max() == 0:
        return H // 2, W // 2, Z // 2

    area_per_z = combined.sum(axis=(1, 2))
    best_z = int(np.argmax(area_per_z))

    mask_slice = combined[best_z]
    ys, xs = np.where(mask_slice > 0)
    cy = int(ys.mean())
    cx = int(xs.mean())

    return cy, cx, best_z


def crop_offset(center, patch_size, dim_size):
    """Compute crop start coordinate, clamped to valid range."""
    return max(0, min(center - patch_size // 2, dim_size - patch_size))


def downsample_masks(mask_dir, z_idx, y0, x0, patch_size=256, latent_size=32):
    """Load, crop, and downsample all region masks for a single Z-slice.

    Returns:
        (num_masks, latent_size, latent_size) float32 array with soft values in [0, 1].
        Zeros for any missing mask file.
    """
    n = len(MASK_NAMES)
    result = np.zeros((n, latent_size, latent_size), dtype=np.float32)
    scale = patch_size // latent_size  # 8

    for i, name in enumerate(MASK_NAMES):
        p = Path(mask_dir) / f"{name}.npy"
        if not p.exists():
            continue
        m = np.load(str(p), mmap_mode="r")  # (Z, H, W)
        crop = m[z_idx, y0:y0 + patch_size, x0:x0 + patch_size].astype(np.float32)
        # avg_pool via torch for clean downsampling
        t = torch.from_numpy(crop).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        pooled = F.avg_pool2d(t, kernel_size=scale, stride=scale)  # (1, 1, h, w)
        result[i] = pooled.squeeze().numpy()

    return result
