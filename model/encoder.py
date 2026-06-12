import torch.nn as nn

from .blocks import ResBlock2D, Downsample2D, SelfAttention2D


class Encoder2D(nn.Module):
    """2D Spatial Encoder. Processes each frame independently.

    Input:  (N, 1, H, W)
    Output: mean (N, latent_ch, H/8, W/8), logvar (N, latent_ch, H/8, W/8)
    """

    def __init__(self, in_ch=1, base_ch=64, latent_ch=16, ch_mult=(1, 2, 4)):
        super().__init__()
        self.conv_in = nn.Conv2d(in_ch, base_ch * ch_mult[0], 3, padding=1)

        # Downsampling levels
        self.down_blocks = nn.ModuleList()
        chs = [base_ch * m for m in ch_mult]
        in_c = chs[0]
        for out_c in chs:
            self.down_blocks.append(nn.ModuleList([
                ResBlock2D(in_c, out_c),
                ResBlock2D(out_c, out_c),
                Downsample2D(out_c),
            ]))
            in_c = out_c

        # Mid (ResBlock + Attention + ResBlock, like Wan VAE)
        self.mid = nn.Sequential(
            ResBlock2D(in_c, in_c),
            SelfAttention2D(in_c),
            ResBlock2D(in_c, in_c),
        )

        # Output
        self.norm_out = nn.GroupNorm(8, in_c)
        self.act_out = nn.SiLU(inplace=True)
        self.conv_out = nn.Conv2d(in_c, latent_ch * 2, 3, padding=1)

    def forward(self, x):
        h = self.conv_in(x)
        for res1, res2, down in self.down_blocks:
            h = res1(h)
            h = res2(h)
            h = down(h)
        h = self.mid(h)
        h = self.act_out(self.norm_out(h))
        h = self.conv_out(h)
        mean, logvar = h.chunk(2, dim=1)
        return mean, logvar
