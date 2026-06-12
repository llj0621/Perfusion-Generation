import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset


class CTPSliceDataset(Dataset):
    """CTP dynamic sequence dataset. Each sample = one Z-slice's temporal sequence.

    Training: random Z-slice, random temporal window, random spatial crop, random flip.
    Validation: sequential Z-slices, full temporal sequence, center crop.
    """

    def __init__(self, cfg, split="train"):
        super().__init__()
        self.cfg = cfg
        self.split = split
        self.is_train = split == "train"

        # List all subject files
        all_files = sorted([f for f in os.listdir(cfg.data_dir) if f.endswith(".npy")])

        # Subject-level split
        rng = random.Random(cfg.seed)
        indices = list(range(len(all_files)))
        rng.shuffle(indices)
        n_train = int(len(all_files) * cfg.train_ratio)

        if self.is_train:
            self.files = [all_files[i] for i in indices[:n_train]]
        else:
            self.files = [all_files[i] for i in indices[n_train:]]

        # Build index: (file_idx, z_idx) pairs
        # Use mmap for data access — Linux page cache will keep 182GB in 565GB RAM
        # All ranks/workers share the same OS page cache, no memory duplication
        self.samples = []
        self._paths = []
        for fi, fname in enumerate(self.files):
            path = os.path.join(cfg.data_dir, fname)
            self._paths.append(path)
            arr = np.load(path, mmap_mode="r")
            n_z = arr.shape[1]
            z_start = cfg.z_trim
            z_end = n_z - cfg.z_trim
            for z in range(z_start, z_end):
                self.samples.append((fi, z))

        # Lazy mmap cache — opened per-worker after fork (mmap can't pickle)
        self._mmaps = {}

    def _normalize(self, x):
        """Clip to soft-tissue window [0, 400] HU and normalize to [-1, 1].

        CTP signal of interest (brain parenchyma, arterial/venous enhancement)
        falls within ~0-400 HU. Narrowing the window from [-1000, 2000] to
        [0, 400] gives ~7.5x better intensity resolution for the relevant signal.
        """
        x = np.clip(x, 0.0, 400.0)
        return x / 200.0 - 1.0  # [0, 400] -> [-1, 1]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_idx, z_idx = self.samples[idx]

        # Lazy mmap open per-worker
        if file_idx not in self._mmaps:
            self._mmaps[file_idx] = np.load(self._paths[file_idx], mmap_mode="r")
        data = self._mmaps[file_idx]

        # Extract slice: (T=25, H=512, W=512)
        slc = np.array(data[:, z_idx, :, :], dtype=np.float32)
        slc = self._normalize(slc)

        T, H, W = slc.shape
        tw = self.cfg.temporal_window
        ps = self.cfg.patch_size

        if self.is_train:
            # Random temporal window
            t0 = random.randint(0, T - tw)
            slc = slc[t0: t0 + tw]

            # Random spatial crop
            y0 = random.randint(0, H - ps)
            x0 = random.randint(0, W - ps)
            slc = slc[:, y0: y0 + ps, x0: x0 + ps]

            # Random horizontal flip
            if random.random() > 0.5:
                slc = slc[:, :, ::-1].copy()
        else:
            # Fixed temporal window (center), center crop
            t0 = (T - tw) // 2
            slc = slc[t0: t0 + tw]
            cy, cx = (H - ps) // 2, (W - ps) // 2
            slc = slc[:, cy: cy + ps, cx: cx + ps]

        # Add channel dim: (T, H, W) -> (T, 1, H, W)
        slc = slc[:, np.newaxis, :, :]
        return torch.from_numpy(np.ascontiguousarray(slc))
