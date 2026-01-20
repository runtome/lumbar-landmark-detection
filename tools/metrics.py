import torch
import numpy as np

def pixel_error_per_level(pred, gt, img_size):
    """
    pred, gt: [B, N, 2] normalized (0–1)
    returns: dict[level] -> list[pixel_error]
    """
    B, N, _ = pred.shape
    errors = {i: [] for i in range(N)}

    pred_px = pred * img_size
    gt_px = gt * img_size

    for b in range(B):
        for i in range(N):
            dx = pred_px[b, i, 0] - gt_px[b, i, 0]
            dy = pred_px[b, i, 1] - gt_px[b, i, 1]
            err = torch.sqrt(dx ** 2 + dy ** 2)
            errors[i].append(err.item())

    return errors



def per_landmark_mae(pred, gt):
    """
    pred, gt: [B, N, 2]
    Returns: [N] MAE in pixels
    """
    return torch.mean(torch.norm(pred - gt, dim=-1), dim=0)