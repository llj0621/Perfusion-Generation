from dataclasses import dataclass


@dataclass
class LDMConfig:
    # Data
    latent_dir: str = ""
    raw_data_dir: str = r""
    metadata_csv: str = r""
    mask_dir: str = r""
    num_frames: int = 25
    train_ratio: float = 0.85
    seed: int = 42
    z_trim: int = 20

    # Mask conditioning
    num_mask_channels: int = 5
    mask_names: tuple = ("aorta", "heart_atrium_left", "heart_atrium_right",
                         "heart_ventricle_left", "heart_ventricle_right")

    # VAE (for encode/decode)
    vae_ckpt: str = ""
    vae_in_ch: int = 1
    vae_base_ch: int = 64
    vae_dec_base_ch: int = 128
    vae_latent_ch: int = 16
    vae_ch_mult: tuple = (1, 2, 4)
    vae_temporal_hidden: int = 256

    # UNet3D
    latent_ch: int = 16
    base_ch: int = 128
    ch_mult: tuple = (1, 2, 4)        # -> [128, 256, 512]
    t_emb_dim: int = 256
    attn_levels: tuple = (2,)         # attention at level 2 + bottleneck
    num_res_blocks: int = 2

    # Flow Matching
    num_inference_steps: int = 20

    # Training
    batch_size: int = 16
    lr: float = 5e-5
    weight_decay: float = 1e-4
    epochs: int = 200
    grad_clip: float = 1.0
    use_amp: bool = True
    ema_decay: float = 0.999
    warmup_epochs: int = 5
    min_lr: float = 1e-6

    # Misc
    num_workers: int = 4
    output_dir: str = "./checkpoints"
    log_file: str = "./log.txt"
    save_interval: int = 20
    val_interval: int = 5
    log_interval: int = 20
