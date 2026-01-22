# tools\visualization_utils.py
import numpy as np
import cv2
import torch


def draw_landmarks(
    image,            # [1, H, W] tensor
    gt=None,           # [N, 2] normalized
    pred=None,         # [N, 2] normalized
    radius=4
):
    img = image.squeeze().cpu().numpy()
    img = (img * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    H, W = img.shape[:2]

    def to_px(coords):
        coords = coords.clone()
        coords[:, 0] *= W
        coords[:, 1] *= H
        return coords

    if gt is not None:
        gt = to_px(gt)
        for (x, y) in gt:
            cv2.circle(img, (int(x), int(y)), radius, (0, 255, 0), -1)

    if pred is not None:
        pred = to_px(pred)
        for (x, y) in pred:
            cv2.circle(img, (int(x), int(y)), radius, (255, 0, 0), -1)

    return img
