import torch.nn as nn


class TemporalConv1D(nn.Module):
    """Multi-scale dilated 1D conv along time axis in latent space.

    Uses increasing kernel sizes and dilation rates to capture both
    local frame transitions and long-range TIC dynamics.
    Effective receptive field: 35 frames (covers full T=25 sequence).

    Operates independently at each spatial position.
    Bidirectional (non-causal) with global residual connection.

    Input:  (B, C, T, h, w)
    Output: (B, C, T, h, w)
    """

    def __init__(self, in_ch=4, hidden_ch=128):
        super().__init__()
        # Block 1: k=3, d=1, pad=1  — local (RF=3)
        # Block 2: k=5, d=2, pad=4  — medium (RF=11)
        # Block 3: k=7, d=4, pad=12 — global (RF=35)
        self.blocks = nn.Sequential(
            nn.Conv1d(in_ch, hidden_ch, kernel_size=3, dilation=1, padding=1),
            nn.GroupNorm(8, hidden_ch),
            nn.SiLU(inplace=True),

            nn.Conv1d(hidden_ch, hidden_ch, kernel_size=5, dilation=2, padding=4),
            nn.GroupNorm(8, hidden_ch),
            nn.SiLU(inplace=True),

            nn.Conv1d(hidden_ch, hidden_ch, kernel_size=7, dilation=4, padding=12),
            nn.GroupNorm(8, hidden_ch),
            nn.SiLU(inplace=True),
        )
        self.proj_out = nn.Conv1d(hidden_ch, in_ch, kernel_size=1)

    def forward(self, z):
        B, C, T, h, w = z.shape
        # (B, C, T, h, w) -> (B*h*w, C, T)
        x = z.permute(0, 3, 4, 1, 2).reshape(B * h * w, C, T)
        # Multi-scale temporal conv with global residual
        out = self.proj_out(self.blocks(x)) + x
        # (B*h*w, C, T) -> (B, C, T, h, w)
        return out.reshape(B, h, w, C, T).permute(0, 3, 4, 1, 2)
