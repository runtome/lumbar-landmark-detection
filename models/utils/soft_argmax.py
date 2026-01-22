import torch
import torch.nn.functional as F


def soft_argmax_2d(heatmaps):
    """
    heatmaps: [B, N, H, W]
    return:   [B, N, 2]  in normalized [0,1]
    """
    B, N, H, W = heatmaps.shape

    heatmaps = heatmaps.view(B, N, -1)
    heatmaps = F.softmax(heatmaps, dim=-1)

    xs = torch.linspace(0, 1, W, device=heatmaps.device)
    ys = torch.linspace(0, 1, H, device=heatmaps.device)

    xs = xs.repeat(H)
    ys = ys.repeat_interleave(W)

    x = torch.sum(heatmaps * xs, dim=-1)
    y = torch.sum(heatmaps * ys, dim=-1)

    return torch.stack([x, y], dim=-1)
