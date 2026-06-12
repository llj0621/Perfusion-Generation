import torch
import torch.nn as nn

from .encoder import Encoder2D
from .decoder import Decoder2D
from .temporal import TemporalConv1D


class SpatioTemporalVAE(nn.Module):
    """Spatiotemporal VAE for CTP dynamic sequences.

    Architecture:
        1. 2D Spatial Encoder (base_ch=64, 3-level 8× downsample, mid attention)
        2. Temporal Module (multi-scale dilated 1D conv in latent space)
        3. 2D Spatial Decoder (dec_base_ch=128, asymmetric wider, mid attention)

    Input:  (B, T, 1, H, W)
    Output: (B, T, 1, H, W), mean (B, C, T, H/8, W/8), logvar (B, C, T, H/8, W/8)
    """

    def __init__(self, in_ch=1, base_ch=64, dec_base_ch=128, latent_ch=16,
                 ch_mult=(1, 2, 4), temporal_hidden=256):
        super().__init__()
        self.latent_ch = latent_ch

        self.encoder = Encoder2D(in_ch, base_ch, latent_ch, ch_mult)
        self.temporal = TemporalConv1D(latent_ch, temporal_hidden)
        self.decoder = Decoder2D(in_ch, dec_base_ch, latent_ch, ch_mult)

    def encode(self, x):
        """Encode frames to latent space.

        Args:
            x: (B, T, 1, H, W)
        Returns:
            mean: (B, C, T, h, w)
            logvar: (B, C, T, h, w)
        """
        B, T = x.shape[:2]
        # Flatten batch and time: (B*T, 1, H, W)
        x_flat = x.view(B * T, *x.shape[2:])

        # Spatial encode per-frame
        mean, logvar = self.encoder(x_flat)  # (B*T, C, h, w) each
        C, h, w = mean.shape[1:]

        # Reshape to (B, T, C, h, w) then permute to (B, C, T, h, w)
        mean = mean.view(B, T, C, h, w).permute(0, 2, 1, 3, 4)
        logvar = logvar.view(B, T, C, h, w).permute(0, 2, 1, 3, 4)

        return mean, logvar

    def reparameterize(self, mean, logvar):
        """VAE reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def temporal_refine(self, z):
        """Apply temporal conv to refine latent across time.

        Args:
            z: (B, C, T, h, w)
        Returns:
            z: (B, C, T, h, w)
        """
        return self.temporal(z)

    def decode(self, z):
        """Decode latent to frames.

        Args:
            z: (B, C, T, h, w)
        Returns:
            recon: (B, T, 1, H, W)
        """
        B, C, T, h, w = z.shape
        # Flatten: (B, C, T, h, w) -> (B*T, C, h, w)
        z_flat = z.permute(0, 2, 1, 3, 4).reshape(B * T, C, h, w)

        # Spatial decode per-frame
        recon = self.decoder(z_flat)  # (B*T, 1, H, W)

        # Reshape back: (B, T, 1, H, W)
        return recon.view(B, T, *recon.shape[1:])

    def forward(self, x, sample=True):
        """Full forward pass.

        Args:
            x: (B, T, 1, H, W) — input frames
            sample: if True, use reparameterization; if False, use mean
        Returns:
            recon: (B, T, 1, H, W)
            mean: (B, C, T, h, w)
            logvar: (B, C, T, h, w)
        """
        # Step 1: Spatial encode
        mean, logvar = self.encode(x)

        # Step 2: Reparameterize
        z = self.reparameterize(mean, logvar) if sample else mean

        # Step 3: Temporal refine
        z = self.temporal_refine(z)

        # Step 4: Spatial decode
        recon = self.decode(z)

        return recon, mean, logvar
