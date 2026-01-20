import numpy as np
import cv2
import torch


def draw_landmarks(
    image,            # [1, H, W] tensor
    gt=None,           # [N, 2]
    pred=None,         # [N, 2]
    radius=4
):
    """
    Returns RGB image with GT (green) and Pred (red)
    """
    img = image.squeeze().cpu().numpy()
    img = (img * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    if gt is not None:
        for (x, y) in gt:
            cv2.circle(img, (int(x), int(y)), radius, (0, 255, 0), -1)

    if pred is not None:
        for (x, y) in pred:
            cv2.circle(img, (int(x), int(y)), radius, (255, 0, 0), -1)

    return img
