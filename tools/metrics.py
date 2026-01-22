# tools\metrics.py

import torch
import numpy as np

def pixel_error_per_level(pred, gt, img_size):
    """
    pred, gt: [B, N, 2] normalized
    """
    H, W = img_size
    pred_px = pred.clone()
    gt_px = gt.clone()

    pred_px[..., 0] *= W
    pred_px[..., 1] *= H
    gt_px[..., 0] *= W
    gt_px[..., 1] *= H

    errors = {i: [] for i in range(pred.shape[1])}
    dist = torch.norm(pred_px - gt_px, dim=2)

    for i in range(pred.shape[1]):
        errors[i].extend(dist[:, i].cpu().numpy().tolist())

    return errors





def per_landmark_mae(pred, gt, img_size):
    """
    pred, gt: [B, N, 2] normalized
    """
    H, W = img_size
    pred_px = pred.clone()
    gt_px = gt.clone()

    pred_px[..., 0] *= W
    pred_px[..., 1] *= H
    gt_px[..., 0] *= W
    gt_px[..., 1] *= H

    return torch.mean(torch.norm(pred_px - gt_px, dim=-1), dim=0)
