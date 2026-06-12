"""Pre-encode all CTP data to VAE latents for LDM training.

Usage:
    python encode_latents.py --data_dir /data/CTP/rest --out_dir /data/CTP/latents --vae_ckpt ./checkpoints/ckpt_final.pt
"""
import os
import argparse

import numpy as np
import torch
from tqdm import tqdm

from config import Config
from model.vae import SpatioTemporalVAE


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
        batch_t: number of frames per forward pass (to control memory)
    Returns:
        latent: (C, T, h, w) numpy float16
    """
    T = slc.shape[0]
    latents = []
    for t0 in range(0, T, batch_t):
        t1 = min(t0 + batch_t, T)
        # (batch_t, 1, H, W) -> (1, batch_t, 1, H, W)
        x = torch.from_numpy(slc[t0:t1]).unsqueeze(0).to(device)
        mean, _ = vae.encode(x)  # (1, C, batch_t, h, w)
        latents.append(mean.squeeze(0).cpu())  # (C, batch_t, h, w)

    # Concat along time: (C, T, h, w)
    latent = torch.cat(latents, dim=1)
    return latent.numpy().astype(np.float16)


def main():
    parser = argparse.ArgumentParser(description="Pre-encode CTP data to VAE latents")
    parser.add_argument("--data_dir", type=str, default=r"E:\CTP\data\rest", help="Raw CTP .npy directory")
    parser.add_argument("--out_dir", type=str, default=r"E:\CTP\data\rest_latent", help="Output latent directory")
    parser.add_argument("--vae_ckpt", type=str, default=r"D:\Project\LiLJ\CTP_v2\checkpoints\ckpt_final.pt")
    parser.add_argument("--z_trim", type=int, default=20, help="Trim first/last N Z-slices")
    parser.add_argument("--patch_size", type=int, default=256, help="Center crop size")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = Config()
    vae = load_vae(args.vae_ckpt, cfg, device)

    all_files = sorted([f for f in os.listdir(args.data_dir) if f.endswith(".npy")])
    print(f"Found {len(all_files)} subjects")

    total_slices = 0
    skipped = 0

    for fname in tqdm(all_files, desc="Subjects"):
        path = os.path.join(args.data_dir, fname)
        data = np.load(path, mmap_mode="r")  # (T=25, Z, H, W)
        T, Z, H, W = data.shape
        ps = args.patch_size

        # Center crop coordinates
        cy, cx = (H - ps) // 2, (W - ps) // 2
        subject_name = os.path.splitext(fname)[0]

        for z in range(args.z_trim, Z - args.z_trim):
            out_name = f"{subject_name}_z{z:03d}.npy"
            out_path = os.path.join(args.out_dir, out_name)

            # Skip if already encoded
            if os.path.exists(out_path):
                skipped += 1
                continue

            # Extract slice: (T, H, W) -> center crop -> normalize -> add channel
            slc = np.array(data[:, z, cy:cy + ps, cx:cx + ps], dtype=np.float32)
            slc = normalize(slc)
            slc = slc[:, np.newaxis, :, :]  # (T, 1, H, W)

            # Encode
            latent = encode_slice(vae, slc, device)  # (C, T, h, w)
            np.save(out_path, latent)
            total_slices += 1

    print(f"Done. Encoded {total_slices} slices, skipped {skipped} existing.")


if __name__ == "__main__":
    main()
