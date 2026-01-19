# models/losses.py
import torch
import torch.nn as nn
import math


class NormalizedL2Loss(nn.Module):
    def __init__(self, img_size):
        """
        img_size: [H, W]
        """
        super().__init__()
        h, w = img_size
        self.norm = math.sqrt(h * h + w * w)

    def forward(self, pred, target):
        """
        pred:   [B, K, 2]
        target:[B, K, 2]
        """
        diff = pred - target
        dist = torch.norm(diff, dim=-1)          # [B, K]
        dist = dist / self.norm
        return dist.mean()
