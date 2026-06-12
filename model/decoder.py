import torch
import torch.nn as nn

from .blocks import ResBlock2D, Upsample2D, SelfAttention2D


class Decoder2D(nn.Module):
    """2D Spatial Decoder. Wider than Encoder (asymmetric design from Wan VAE).

    Input:  (N, latent_ch, h, w)
    Output: (N, 1, 8h, 8w)  with tanh activation -> [-1, 1]
    """

    def __init__(self, out_ch=1, base_ch=128, latent_ch=16, ch_mult=(1, 2, 4)):
        super().__init__()
        chs = [base_ch * m for m in ch_mult]
        top_ch = chs[-1]

        self.conv_in = nn.Conv2d(latent_ch, top_ch, 3, padding=1)

        # Mid (ResBlock + Attention + ResBlock)
        self.mid = nn.Sequential(
            ResBlock2D(top_ch, top_ch),
            SelfAttention2D(top_ch),
            ResBlock2D(top_ch, top_ch),
        )

        # Upsampling levels (reverse order)
        self.up_blocks = nn.ModuleList()
        in_c = top_ch
        for out_c in reversed(chs):
            self.up_blocks.append(nn.ModuleList([
                ResBlock2D(in_c, out_c),
                ResBlock2D(out_c, out_c),
                Upsample2D(out_c),
            ]))
            in_c = out_c

        # Output
        self.norm_out = nn.GroupNorm(8, in_c)
        self.act_out = nn.SiLU(inplace=True)
        self.conv_out = nn.Conv2d(in_c, out_ch, 3, padding=1)

    def forward(self, z):
        h = self.conv_in(z)
        h = self.mid(h)
        for res1, res2, up in self.up_blocks:
            h = res1(h)
            h = res2(h)
            h = up(h)
        h = self.act_out(self.norm_out(h))
        h = self.conv_out(h)
        return torch.tanh(h)
