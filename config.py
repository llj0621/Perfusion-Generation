from dataclasses import dataclass


@dataclass
class Config:
    # Data
    data_dir: str = "../data/rest"
    patch_size: int = 256
    temporal_window: int = 5
    train_ratio: float = 0.85  # 110 train, 20 val
    z_trim: int = 20  # trim first/last N slices

    # Model
    in_ch: int = 1
    base_ch: int = 64
    dec_base_ch: int = 128
    latent_ch: int = 16
    ch_mult: tuple = (1, 2, 4)
    temporal_hidden: int = 256

    # Training (per-GPU batch size, effective = batch_size * world_size)
    batch_size: int = 8
    lr: float = 2e-4  # base lr, scaled by world_size at runtime
    weight_decay: float = 1e-5
    epochs: int = 100
    grad_clip: float = 1.0
    use_amp: bool = True

    # Loss weights
    w_recon: float = 1.0
    w_ssim: float = 0.1
    w_temporal: float = 0.1
    w_kl: float = 1e-4

    # EMA
    ema_decay: float = 0.99

    # Scheduler
    warmup_epochs: int = 3
    min_lr: float = 1e-6

    # Misc
    seed: int = 42
    num_workers: int = 4
    output_dir: str = "./checkpoints"
    log_file: str = "./log.txt"
    save_interval: int = 10
    val_interval: int = 5
    log_interval: int = 20
