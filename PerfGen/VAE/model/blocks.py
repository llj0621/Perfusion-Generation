import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock2D(nn.Module):
    """2D Residual Block: GroupNorm -> SiLU -> Conv -> GroupNorm -> SiLU -> Conv + Skip."""

    def __init__(self, in_ch, out_ch, num_groups=8):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(num_groups, in_ch), in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(min(num_groups, out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        h = self.act(self.norm1(x))
        h = self.conv1(h)
        h = self.act(self.norm2(h))
        h = self.conv2(h)
        return h + self.skip(x)


class Downsample2D(nn.Module):
    """Strided convolution downsample (2x)."""

    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample2D(nn.Module):
    """Bilinear upsample (2x) + conv."""

    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return self.conv(x)


class SelfAttention2D(nn.Module):
    """Single-head self-attention for spatial features. Inspired by Wan VAE."""

    def __init__(self, ch, num_groups=8):
        super().__init__()
        self.ch = ch
        self.norm = nn.GroupNorm(min(num_groups, ch), ch)
        self.qkv = nn.Conv2d(ch, ch * 3, 1)
        self.proj = nn.Conv2d(ch, ch, 1)
        # Zero-init projection for stable training
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        identity = x
        B, C, H, W = x.shape
        h = self.norm(x)

        # Compute Q, K, V
        qkv = self.qkv(h)  # (B, 3C, H, W)
        qkv = qkv.reshape(B, 3, C, H * W).permute(1, 0, 3, 2)  # (3, B, HW, C)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each (B, HW, C)

        # Scaled dot-product attention
        q = q.unsqueeze(1)  # (B, 1, HW, C)
        k = k.unsqueeze(1)  # (B, 1, HW, C)
        v = v.unsqueeze(1)  # (B, 1, HW, C)

        attn_out = F.scaled_dot_product_attention(q, k, v)  # (B, 1, HW, C)
        attn_out = attn_out.squeeze(1).permute(0, 2, 1).reshape(B, C, H, W)

        # Project and residual
        out = self.proj(attn_out)
        return out + identity
