import torch
import torch.nn as nn
import torch.nn.functional as F


def gaussian_window(window_size, sigma=1.5):
    """Create 1D Gaussian kernel."""
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    return g / g.sum()


def create_ssim_window(window_size, channel):
    """Create 2D Gaussian window for SSIM."""
    w1d = gaussian_window(window_size)
    w2d = w1d.unsqueeze(1) * w1d.unsqueeze(0)
    return w2d.unsqueeze(0).unsqueeze(0).expand(channel, 1, -1, -1).contiguous()


def ssim(x, y, window_size=7, C1=(0.01 * 2) ** 2, C2=(0.03 * 2) ** 2):
    """Compute SSIM between x and y (both in [-1, 1]).

    C1/C2 use L=2 (dynamic range of [-1, 1]) per the SSIM formula:
    C1 = (K1*L)^2, C2 = (K2*L)^2 where K1=0.01, K2=0.03.

    Args:
        x, y: (N, 1, H, W)
    Returns:
        scalar SSIM value (mean over batch)
    """
    ch = x.size(1)
    window = create_ssim_window(window_size, ch).to(x.device, x.dtype)
    pad = window_size // 2

    mu_x = F.conv2d(x, window, padding=pad, groups=ch)
    mu_y = F.conv2d(y, window, padding=pad, groups=ch)
    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = F.conv2d(x * x, window, padding=pad, groups=ch) - mu_x2
    sigma_y2 = F.conv2d(y * y, window, padding=pad, groups=ch) - mu_y2
    sigma_xy = F.conv2d(x * y, window, padding=pad, groups=ch) - mu_xy

    num = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
    den = (mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2)
    return (num / den).mean()


def compute_loss(recon, target, mean, logvar, w_recon=1.0, w_ssim=0.1, w_temporal=0.5, w_kl=1e-6):
    """Compute total VAE loss.

    Args:
        recon:   (B, T, 1, H, W) — reconstructed frames
        target:  (B, T, 1, H, W) — original frames
        mean:    (B, C, T, h, w) — latent mean
        logvar:  (B, C, T, h, w) — latent log-variance
    Returns:
        total_loss, loss_dict
    """
    B, T = recon.shape[:2]

    # 1. Reconstruction loss (SmoothL1)
    loss_recon = F.smooth_l1_loss(recon, target, beta=0.01)

    # 2. SSIM loss (per-frame 2D)
    recon_flat = recon.view(B * T, *recon.shape[2:])
    target_flat = target.view(B * T, *target.shape[2:])
    loss_ssim = 1.0 - ssim(recon_flat, target_flat)

    # 3. Temporal consistency loss
    recon_diff = recon[:, 1:] - recon[:, :-1]
    target_diff = target[:, 1:] - target[:, :-1]
    loss_temporal = F.l1_loss(recon_diff, target_diff)

    # 4. KL divergence
    loss_kl = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())

    # Total
    total = w_recon * loss_recon + w_ssim * loss_ssim + w_temporal * loss_temporal + w_kl * loss_kl

    loss_dict = {
        "total": total,
        "recon": loss_recon,
        "ssim": loss_ssim,
        "temporal": loss_temporal,
        "kl": loss_kl,
    }
    return total, loss_dict
