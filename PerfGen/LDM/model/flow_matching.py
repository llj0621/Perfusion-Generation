import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_embedding(t, dim):
    """Sinusoidal positional embedding for continuous timestep t.

    Args:
        t: (B,) timestep values in [0, 1]
        dim: embedding dimension (must be even)
    Returns:
        (B, dim) embedding
    """
    assert dim % 2 == 0
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = t[:, None].float() * freqs[None, :]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class FlowMatching(nn.Module):
    """Flow matching for latent diffusion.

    Training: linear interpolation + velocity prediction.
    Inference: Euler ODE stepping from t=1 (noise) to t=0 (clean).
    """

    def __init__(self, num_inference_steps=20):
        super().__init__()
        self.num_inference_steps = num_inference_steps

    @staticmethod
    def add_noise(z_0, noise, t):
        """Linear interpolation: z_t = (1-t)*z_0 + t*noise.

        Args:
            z_0: (B, C, T, H, W) clean latent
            noise: (B, C, T, H, W) standard Gaussian
            t: (B,) timestep in [0, 1]
        Returns:
            z_t: (B, C, T, H, W)
        """
        t = t[:, None, None, None, None]  # (B, 1, 1, 1, 1)
        return (1 - t) * z_0 + t * noise

    @staticmethod
    def velocity_target(z_0, noise):
        """Target velocity field: v = dz_t/dt = noise - z_0.

        Args:
            z_0: (B, C, T, H, W) clean latent
            noise: (B, C, T, H, W) standard Gaussian
        Returns:
            v: (B, C, T, H, W)
        """
        return noise - z_0

    @torch.no_grad()
    def sample(self, model, cond, shape, device, timestamps=None, heart_rate=None, mask=None):
        """Generate latent sequence via Euler ODE from t=1 to t=0.

        Args:
            model: UNet3D, takes (model_input, t, timestamps, heart_rate, mask) -> v_pred
            cond: (B, C, H, W) first frame latent
            shape: target shape (B, C, T, H, W)
            device: torch device
            timestamps: (B, T) real frame timestamps in seconds, or None
            heart_rate: (B,) heart rate in bpm, or None
            mask: (B, M, H, W) region masks, or None
        Returns:
            z_0: (B, C, T, H, W) denoised latent
        """
        z_t = torch.randn(shape, device=device)
        dt = 1.0 / self.num_inference_steps

        cond_exp = cond.unsqueeze(2).expand(shape)  # (B, C, T, H, W)

        for i in range(self.num_inference_steps):
            t_val = 1.0 - i * dt
            t_batch = torch.full((shape[0],), t_val, device=device)

            model_input = torch.cat([z_t, cond_exp], dim=1)  # (B, 2C, T, H, W)
            v_pred = model(model_input, t_batch, timestamps=timestamps,
                           heart_rate=heart_rate, mask=mask)

            z_t = z_t - dt * v_pred

        return z_t
