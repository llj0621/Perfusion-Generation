import torch
import torch.nn as nn

from .blocks3d import ResBlock3D, Downsample3D, Upsample3D, SelfAttention3D
from .flow_matching import sinusoidal_embedding


class UNet3D(nn.Module):
    """3D UNet denoiser for flow matching in latent space.

    Spatial-only downsampling (T is preserved at all levels).
    Time embedding + heart rate injected via AdaGN in every ResBlock3D.
    Temporal RoPE applied in SelfAttention3D using real timestamps.
    First-frame condition concatenated along channel dim at input.
    Region masks (optional) concatenated along channel dim at input.

    Input:  (B, latent_ch*2 + num_mask_ch, T, H, W)
    Output: (B, latent_ch, T, H, W) — predicted velocity
    """

    def __init__(self, latent_ch=16, base_ch=128, ch_mult=(1, 2, 4),
                 t_emb_dim=256, attn_levels=(2,), num_res_blocks=2,
                 num_mask_ch=0):
        super().__init__()
        self.latent_ch = latent_ch
        self.t_emb_dim = t_emb_dim
        self.num_levels = len(ch_mult)
        self.attn_levels = set(attn_levels)
        self.num_mask_ch = num_mask_ch

        chs = [base_ch * m for m in ch_mult]
        in_ch = latent_ch * 2 + num_mask_ch

        # Time embedding MLP (diffusion timestep)
        self.time_mlp = nn.Sequential(
            nn.Linear(t_emb_dim, t_emb_dim),
            nn.SiLU(),
            nn.Linear(t_emb_dim, t_emb_dim),
        )

        # Heart rate embedding MLP (scalar -> t_emb_dim, added to time embedding)
        self.hr_mlp = nn.Sequential(
            nn.Linear(1, t_emb_dim),
            nn.SiLU(),
            nn.Linear(t_emb_dim, t_emb_dim),
        )

        # Input conv
        self.conv_in = nn.Conv3d(in_ch, chs[0], 3, padding=1)

        # Encoder
        self.enc_blocks = nn.ModuleList()
        self.enc_downs = nn.ModuleList()
        prev_ch = chs[0]
        for level, ch in enumerate(chs):
            blocks = nn.ModuleList()
            for i in range(num_res_blocks):
                blocks.append(ResBlock3D(prev_ch if i == 0 else ch, ch, t_emb_dim))
            if level in self.attn_levels:
                blocks.append(SelfAttention3D(ch))
            self.enc_blocks.append(blocks)
            self.enc_downs.append(Downsample3D(ch))
            prev_ch = ch

        # Bottleneck
        self.mid = nn.ModuleList([
            ResBlock3D(chs[-1], chs[-1], t_emb_dim),
            SelfAttention3D(chs[-1]),
            ResBlock3D(chs[-1], chs[-1], t_emb_dim),
        ])

        # Decoder (reverse order, with skip connections)
        self.dec_ups = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        prev_ch = chs[-1]
        for level in reversed(range(self.num_levels)):
            ch = chs[level]
            skip_ch = ch

            self.dec_ups.append(Upsample3D(prev_ch))

            blocks = nn.ModuleList()
            for i in range(num_res_blocks):
                block_in = prev_ch + skip_ch if i == 0 else ch
                blocks.append(ResBlock3D(block_in, ch, t_emb_dim))
            if level in self.attn_levels:
                blocks.append(SelfAttention3D(ch))
            self.dec_blocks.append(blocks)
            prev_ch = ch

        # Output conv
        self.norm_out = nn.GroupNorm(8, chs[0])
        self.act_out = nn.SiLU(inplace=True)
        self.conv_out = nn.Conv3d(chs[0], latent_ch, 3, padding=1)

        # Zero-init output for stable training
        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)

    def forward(self, x, t, timestamps=None, heart_rate=None, mask=None):
        """
        Args:
            x: (B, latent_ch*2, T, H, W) — concat of noisy latent and condition
            t: (B,) diffusion timestep in [0, 1]
            timestamps: (B, T) real frame timestamps in seconds, or None
            heart_rate: (B,) heart rate in bpm, or None
            mask: (B, num_mask_ch, H, W) region masks, or None
        Returns:
            v: (B, latent_ch, T, H, W) — predicted velocity
        """
        # Concat mask along channel dim if provided
        if mask is not None and self.num_mask_ch > 0:
            T = x.shape[2]
            mask_exp = mask.unsqueeze(2).expand(-1, -1, T, -1, -1)  # (B, M, T, H, W)
            x = torch.cat([x, mask_exp], dim=1)

        # Time embedding (diffusion step)
        t_emb = sinusoidal_embedding(t, self.t_emb_dim)
        t_emb = self.time_mlp(t_emb)  # (B, t_emb_dim)

        # Heart rate conditioning: normalize and add to time embedding
        if heart_rate is not None:
            hr_norm = (heart_rate.float() - 70.0) / 20.0  # (B,)
            hr_emb = self.hr_mlp(hr_norm.unsqueeze(-1))  # (B, t_emb_dim)
            t_emb = t_emb + hr_emb

        # Input
        h = self.conv_in(x)

        # Encoder — collect skip connections
        skips = []
        for level in range(self.num_levels):
            for block in self.enc_blocks[level]:
                if isinstance(block, ResBlock3D):
                    h = block(h, t_emb)
                else:
                    h = block(h, timestamps)
            skips.append(h)
            h = self.enc_downs[level](h)

        # Bottleneck
        for block in self.mid:
            if isinstance(block, ResBlock3D):
                h = block(h, t_emb)
            else:
                h = block(h, timestamps)

        # Decoder — consume skip connections in reverse
        for level in range(self.num_levels):
            skip = skips.pop()
            h = self.dec_ups[level](h)
            h = torch.cat([h, skip], dim=1)
            for block in self.dec_blocks[level]:
                if isinstance(block, ResBlock3D):
                    h = block(h, t_emb)
                else:
                    h = block(h, timestamps)

        # Output
        h = self.act_out(self.norm_out(h))
        return self.conv_out(h)
