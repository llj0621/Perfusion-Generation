import torch


def temporal_rope(q, k, timestamps, T, HW):
    """Apply RoPE rotation to q, k along temporal dimension only.

    Same rotation for all H*W spatial positions within a frame —
    RoPE only encodes temporal distance (real seconds).

    Args:
        q, k: (B, N, C) where N = T*HW, C = channel dim (must be even)
        timestamps: (B, T) real timestamps in seconds
        T: number of time frames
        HW: number of spatial tokens per frame (H*W)
    Returns:
        q_rot, k_rot: same shape as input
    """
    B, N, C = q.shape
    half = C // 2
    freqs = 1.0 / (10000 ** (torch.arange(half, device=q.device).float() / half))

    # (B, T, 1) * (1, 1, half) -> (B, T, half)
    angles = timestamps.unsqueeze(-1) * freqs.unsqueeze(0).unsqueeze(0)
    cos_t = angles.cos()  # (B, T, half)
    sin_t = angles.sin()  # (B, T, half)

    # Expand to all spatial positions: (B, T, half) -> (B, T*HW, half)
    cos_t = cos_t.unsqueeze(2).expand(B, T, HW, half).reshape(B, N, half)
    sin_t = sin_t.unsqueeze(2).expand(B, T, HW, half).reshape(B, N, half)

    def rotate(x, cos, sin):
        x1, x2 = x[..., :half], x[..., half:]
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

    return rotate(q, cos_t, sin_t), rotate(k, cos_t, sin_t)
