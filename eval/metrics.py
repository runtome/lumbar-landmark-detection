import torch
import pandas as pd
import numpy as np

LEVELS = ["L1/L2", "L2/L3", "L3/L4", "L4/L5", "L5/S1"]

def compute_errors(pred, gt, img_size):
    H, W = img_size
    diag = (H**2 + W**2) ** 0.5

    pixel_err = torch.norm(pred - gt, dim=-1)
    norm_err = pixel_err / diag

    return pixel_err.cpu().numpy(), norm_err.cpu().numpy()

def export_error_csv(preds, gts, img_size, out_csv):
    rows = []

    for p, g in zip(preds, gts):
        pe, ne = compute_errors(p, g, img_size)
        for i, lvl in enumerate(LEVELS):
            rows.append({
                "level": lvl,
                "pixel_error": pe[i],
                "norm_error": ne[i],
            })

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    return df
