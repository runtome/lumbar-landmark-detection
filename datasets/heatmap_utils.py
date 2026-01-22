# datasets/heatmap_utils.py
import torch
import math


def generate_gaussian_heatmap(h, w, cx, cy, sigma=2):
    y = torch.arange(0, h).view(h, 1)
    x = torch.arange(0, w).view(1, w)

    heatmap = torch.exp(
        -((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma ** 2)
    )
    return heatmap
