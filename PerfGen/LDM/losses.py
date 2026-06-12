import torch.nn.functional as F


def velocity_loss(v_pred, v_target):
    """MSE loss between predicted and target velocity.

    Args:
        v_pred: (B, C, T, H, W) predicted velocity
        v_target: (B, C, T, H, W) target velocity (noise - clean)
    Returns:
        scalar loss
    """
    return F.mse_loss(v_pred, v_target)
