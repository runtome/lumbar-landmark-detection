import torch


def per_landmark_mae(pred, gt):
    """
    pred, gt: [B, N, 2]
    Returns: [N] MAE in pixels
    """
    return torch.mean(torch.norm(pred - gt, dim=-1), dim=0)
