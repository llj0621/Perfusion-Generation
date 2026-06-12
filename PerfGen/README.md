# PerfGen

Two-stage CTP sequence generation: VAE spatial compression → LDM latent flow matching.

```
PerfGen/
├── utils.py              # shared EMA, save/load checkpoint
├── VAE/                  # Stage 1: spatiotemporal VAE
│   ├── config.py
│   ├── dataset.py        # CTPSliceDataset (reads raw .npy)
│   ├── losses.py
│   ├── train.py
│   └── model/            # SpatioTemporalVAE, Encoder2D, Decoder2D, TemporalConv1D
└── LDM/                  # Stage 2: latent diffusion (flow matching)
    ├── config.py
    ├── dataset.py        # LatentCTPDataset (reads pre-encoded latents)
    ├── losses.py
    ├── mask_utils.py
    ├── encode_latents.py # bridge: VAE-encode raw data → LDM training inputs
    ├── sample.py         # inference: first frame → full 25-frame sequence
    └── model/            # UNet3D, FlowMatching, RoPE
```

---

## Data layout expected

```
data\
├── raw_npy\                          # raw CTP volumes, one file per subject
│   ├── subject_001.npy            # shape (T=25, Z, H, W), dtype float32, unit HU
│   └── ...
├── heartchambers_seg_npy\         # cardiac segmentation masks
│   ├── subject_001\
│   │   ├── aorta.npy              # shape (Z, H, W), binary
│   │   ├── heart_atrium_left.npy
│   │   ├── heart_atrium_right.npy
│   │   ├── heart_ventricle_left.npy
│   │   └── heart_ventricle_right.npy
│   └── ...
├── latent\                # produced by LDM/encode_latents.py
│   ├── subject_001_z020.npy       # shape (C=16, T=25, h=32, w=32), float16
│   ├── subject_001_z020_mask.npy  # shape (5, 32, 32), float32
│   ├── crop_centers.json
│   └── ...
└── timestamps_heartrate.csv       # columns: sample, t0..t24, heart_rate_bpm
```

All scripts are run from the **PerfGen root directory**.

---

## Step 1 — Train VAE

Edit `VAE/config.py` to set `data_dir`, `output_dir`, `log_file`, then:

```bash
# single GPU
python VAE/train.py

# multi-GPU (e.g. 4 GPUs)
torchrun --nproc_per_node=4 VAE/train.py
```

Key config defaults (`VAE/config.py`):

| field | default | note |
|---|---|---|
| `data_dir` | `` | raw CTP .npy directory |
| `patch_size` | `256` | spatial crop size |
| `temporal_window` | `5` | frames per VAE forward pass during training |
| `batch_size` | `8` | per-GPU |
| `epochs` | `100` | |
| `lr` | `2e-4` | scaled by world_size at runtime |
| `latent_ch` | `16` | latent channels → spatial 8× downsample: 256→32 |
| `output_dir` | `./checkpoints` | checkpoint save path |

Checkpoints saved as `ckpt_epoch{N:04d}.pt` and `ckpt_final.pt`.

---

## Step 2 — Encode latents (VAE → LDM inputs)

Requires a trained VAE checkpoint. Produces latents + masks + `crop_centers.json` for LDM training.

```bash
python LDM/encode_latents.py \
    --vae_ckpt  VAE/checkpoints/ckpt_final.pt \
    --data_dir  data/raw_npy \
    --mask_dir  data/heartchambers_seg_npy \
    --out_dir   data/latent \
    --z_trim    20 \
    --patch_size 256
```

`--z_trim 20` skips the first and last 20 Z-slices (typically outside the heart). Existing files are skipped automatically, so the script is safe to re-run.

---

## Step 3 — Train LDM

Edit `LDM/config.py` to set `latent_dir`, `metadata_csv`, `mask_dir`, `output_dir`, `log_file`, then:

```bash
# single GPU
python LDM/train.py

# multi-GPU
torchrun --nproc_per_node=4 LDM/train.py
```

Key config defaults (`LDM/config.py`):

| field | default | note |
|---|---|---|
| `latent_dir` | `""` | **must set** — path to rest_latent_v3 |
| `metadata_csv` | `data\timestamps_heartrate.csv` | |
| `mask_dir` | `data\heartchambers_seg_npy` | |
| `vae_ckpt` | `VAE\checkpoints\ckpt_final.pt` | update to your VAE checkpoint |
| `num_frames` | `25` | |
| `batch_size` | `16` | per-GPU |
| `epochs` | `200` | |
| `lr` | `5e-5` | scaled by sqrt(world_size) |
| `latent_ch` | `16` | must match VAE |
| `num_mask_channels` | `5` | aorta + 4 cardiac chambers |
| `num_inference_steps` | `20` | ODE steps at sampling time |
| `output_dir` | `""` | **must set** |

---

## Step 4 — Sample (inference)

Generate a 25-frame sequence from the first frame of a validation subject.

```bash
# use original timestamps + HR from CSV (default)
python LDM/sample.py \
    --ldm_ckpt LDM/checkpoints/ckpt_final.pt \
    --subject  0

# specify a subject by name
python LDM/sample.py \
    --ldm_ckpt      LDM/checkpoints/ckpt_final.pt \
    --subject_name  subject_001 \
    --z_slice       35

# custom HR and duration
python LDM/sample.py \
    --ldm_ckpt       LDM/checkpoints/ckpt_final.pt \
    --subject        0 \
    --mode           custom \
    --custom_hr      80 \
    --custom_duration 36.0
```

Key arguments:

| argument | default | note |
|---|---|---|
| `--ldm_ckpt` | required | LDM checkpoint path |
| `--subject` | `0` | index into val set (ignored if `--subject_name` given) |
| `--subject_name` | `None` | subject filename stem, e.g. `subject_001` |
| `--z_slice` | `None` | Z index; defaults to `best_z` from crop_centers.json |
| `--mode` | `original` | `original` uses CSV timestamps/HR; `custom` uses CLI values |
| `--custom_hr` | `70.0` | heart rate in bpm (custom mode) |
| `--custom_duration` | `36.0` | total scan duration in seconds (custom mode) |
| `--frames` | `0 5 10 15 20 24` | time frames to include in comparison plot |
| `--out_dir` | `./LDM/samples` | output directory for PNG plots |

Output: a side-by-side GT vs Generated PNG saved to `--out_dir`, plus MSE and PSNR printed to stdout.

---

## Tensor shapes reference

| stage | tensor | shape |
|---|---|---|
| VAE input | raw CTP slice | `(B, T=25, 1, H=256, W=256)` |
| VAE latent | mean / z | `(B, C=16, T, h=32, w=32)` |
| LDM input | noisy latent + cond | `(B, 32, T=25, 32, 32)` |
| LDM mask | cardiac regions | `(B, 5, 32, 32)` |
| LDM output | predicted velocity | `(B, 16, T=25, 32, 32)` |

---

## Checkpoint format

All checkpoints saved by `utils.save_checkpoint` contain:

```python
{
    "epoch":     int,
    "model":     model.state_dict(),
    "ema":       ema.state_dict(),   # None if not rank 0
    "optimizer": optimizer.state_dict(),
    "scheduler": scheduler.state_dict(),
}
```

At inference, EMA weights are used when present (`ema` key is not None).
