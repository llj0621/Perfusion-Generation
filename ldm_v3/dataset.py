import json
import os
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class LatentCTPDataset(Dataset):
    """Dataset for pre-encoded VAE latents with masks and metadata.

    Each sample = one Z-slice's full 25-frame latent + region mask + per-subject metadata.

    Training: random horizontal flip (applied to both latent and mask).
    Validation: no augmentation.
    """

    def __init__(self, cfg, split="train"):
        super().__init__()
        self.cfg = cfg
        self.split = split
        self.is_train = split == "train"

        all_files = sorted([f for f in os.listdir(cfg.latent_dir)
                            if f.endswith(".npy") and not f.endswith("_mask.npy")])

        # Subject-level split (same seed/ratio as VAE for consistency)
        subjects = sorted(set(f.rsplit("_z", 1)[0] for f in all_files))
        rng = random.Random(cfg.seed)
        indices = list(range(len(subjects)))
        rng.shuffle(indices)
        n_train = int(len(subjects) * cfg.train_ratio)

        train_subjects = sorted(subjects[i] for i in indices[:n_train])
        val_subjects = sorted(subjects[i] for i in indices[n_train:])

        if self.is_train:
            keep = set(train_subjects)
        else:
            keep = set(val_subjects)

        self.subjects = train_subjects if self.is_train else val_subjects
        self.files = [f for f in all_files if f.rsplit("_z", 1)[0] in keep]
        self._paths = [os.path.join(cfg.latent_dir, f) for f in self.files]

        # Load metadata: timestamps + heart rate per subject
        self._load_metadata(cfg.metadata_csv)

        # Lazy mmap cache (opened per-worker after fork)
        self._mmaps = {}

    def _load_metadata(self, csv_path):
        """Build lookup: subject_name -> (timestamps[25], heart_rate)."""
        df = pd.read_csv(csv_path)
        self.metadata = {}
        t_cols = [f"t{i}" for i in range(self.cfg.num_frames)]
        for _, row in df.iterrows():
            subj = row["sample"]
            timestamps = np.array([row[c] for c in t_cols], dtype=np.float32)
            hr = float(row["heart_rate_bpm"])
            self.metadata[subj] = (timestamps, hr)

    def save_split(self, output_dir):
        """Save train/val split info to JSON."""
        split_info = {
            "split": self.split,
            "subjects": self.subjects,
            "files": self.files,
            "n_subjects": len(self.subjects),
            "n_samples": len(self.files),
        }
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f"{self.split}_split.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(split_info, f, indent=2, ensure_ascii=False)
        return path

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # Lazy mmap open per-worker
        if idx not in self._mmaps:
            self._mmaps[idx] = np.load(self._paths[idx], mmap_mode="r")
        data = self._mmaps[idx]  # (C=16, T=25, h=32, w=32), float16

        z_full = np.array(data, dtype=np.float32)  # copy from mmap

        # Load mask
        mask_path = self._paths[idx].replace(".npy", "_mask.npy")
        mask = np.load(mask_path)  # (5, 32, 32) float32

        # Random horizontal flip (same decision for latent and mask)
        if self.is_train and random.random() > 0.5:
            z_full = z_full[:, :, :, ::-1].copy()
            mask = mask[:, :, ::-1].copy()

        z_cond = z_full[:, 0, :, :]  # (C, h, w) — first frame

        # Get metadata for this subject
        subject = self.files[idx].rsplit("_z", 1)[0]
        timestamps, heart_rate = self.metadata[subject]

        return {
            "z_full": torch.from_numpy(np.ascontiguousarray(z_full)),
            "z_cond": torch.from_numpy(np.ascontiguousarray(z_cond)),
            "timestamps": torch.from_numpy(timestamps),
            "heart_rate": torch.tensor(heart_rate, dtype=torch.float32),
            "mask": torch.from_numpy(np.ascontiguousarray(mask)),  # (5, 32, 32)
        }
