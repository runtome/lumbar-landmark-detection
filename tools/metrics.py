import torch
import numpy as np

def pixel_error_per_level(pred, gt):
    errors = {i: [] for i in range(pred.shape[1])}

    diff = pred - gt                  # already pixels
    dist = torch.norm(diff, dim=2)    # (B, N)

    for i in range(pred.shape[1]):
        errors[i].extend(dist[:, i].cpu().numpy().tolist())

    return errors




def per_landmark_mae(pred, gt):
    """
    pred, gt: [B, N, 2]
    Returns: [N] MAE in pixels
    """
    return torch.mean(torch.norm(pred - gt, dim=-1), dim=0)