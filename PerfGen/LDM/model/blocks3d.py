import torch
import torch.nn as nn
import torch.nn.functional as F

from .rope import temporal_rope


class ResBlock3D(nn.Module):
    """3D Residual Block with time embedding injection (AdaGN).

    GN -> SiLU -> Conv3d -> [+ time_emb via AdaGN] -> GN -> SiLU -> Conv3d + Skip.
    """

    def __init__(self, in_ch, out_ch, t_emb_dim=256, num_groups=8):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(num_groups, in_ch), in_ch)
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(min(num_groups, out_ch), out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv3d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.act = nn.SiLU(inplace=True)

        # Time embedding -> (scale, shift) for AdaGN
        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(t_emb_dim, out_ch * 2),
        )

    def forward(self, x, t_emb):
        """
        Args:
            x: (B, C, T, H, W)
            t_emb: (B, t_emb_dim)
        """
        h = self.act(self.norm1(x))
        h = self.conv1(h)

        # AdaGN: modulate after norm2 with time embedding
        scale_shift = self.time_proj(t_emb)  # (B, out_ch*2)
        scale, shift = scale_shift.chunk(2, dim=1)
        scale = scale[:, :, None, None, None]  # (B, C, 1, 1, 1)
        shift = shift[:, :, None, None, None]
        h = self.norm2(h) * (1 + scale) + shift

        h = self.act(h)
        h = self.conv2(h)
        return h + self.skip(x)


class Downsample3D(nn.Module):
    """Spatial-only strided conv downsample (2x). Keeps T unchanged."""

    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv3d(ch, ch, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))

    def forward(self, x):
        return self.conv(x)


class Upsample3D(nn.Module):
    """Spatial-only trilinear upsample (2x) + conv. Keeps T unchanged."""

    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv3d(ch, ch, kernel_size=(1, 3, 3), padding=(0, 1, 1))

    def forward(self, x):
        x = F.interpolate(x, scale_factor=(1, 2, 2), mode="trilinear", align_corners=False)
        return self.conv(x)


class SelfAttention3D(nn.Module):
    """Single-head self-attention over flattened (T, H, W) tokens.

    Supports optional temporal RoPE via timestamps.
    """

    def __init__(self, ch, num_groups=8):
        super().__init__()
        self.ch = ch
        self.norm = nn.GroupNorm(min(num_groups, ch), ch)
        self.qkv = nn.Conv3d(ch, ch * 3, 1)
        self.proj = nn.Conv3d(ch, ch, 1)
        # Zero-init for stable training
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x, timestamps=None):
        """
        Args:
            x: (B, C, T, H, W)
            timestamps: (B, T) real timestamps in seconds, or None
        """
        identity = x
        B, C, T, H, W = x.shape
        h = self.norm(x)

        qkv = self.qkv(h)  # (B, 3C, T, H, W)
        qkv = qkv.reshape(B, 3, C, T * H * W).permute(1, 0, 3, 2)  # (3, B, THW, C)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each (B, THW, C)

        # Apply temporal RoPE if timestamps provided
        if timestamps is not None:
            q, k = temporal_rope(q, k, timestamps, T, H * W)

        # (B, 1, THW, C) for scaled_dot_product_attention
        q = q.unsqueeze(1)
        k = k.unsqueeze(1)
        v = v.unsqueeze(1)

        attn_out = F.scaled_dot_product_attention(q, k, v)  # (B, 1, THW, C)
        attn_out = attn_out.squeeze(1).permute(0, 2, 1).reshape(B, C, T, H, W)

        out = self.proj(attn_out)
        return out + identity
