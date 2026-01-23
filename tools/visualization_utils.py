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


def draw_heatmaps_on_image(
    image,
    heatmaps,
    alpha=0.4,
    colormap=cv2.COLORMAP_JET,
):
    """
    image:    Tensor [C,H,W] or ndarray [H,W] or [H,W,C]
    heatmaps: Tensor [N,H,W]
    return:   ndarray [H,W,3] uint8
    """

    # ------------------
    # Image → numpy
    # ------------------
    if torch.is_tensor(image):
        image = image.detach().cpu().numpy()
        if image.ndim == 3:          # CHW
            image = np.transpose(image, (1, 2, 0))
    else:
        image = image.copy()

    # normalize image
    image = image.astype(np.float32)
    image = (image - image.min()) / (image.max() - image.min() + 1e-6)
    image = (image * 255).astype(np.uint8)

    # ensure 3-channel
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    H, W, _ = image.shape

    # ------------------
    # Heatmaps → numpy
    # ------------------
    if torch.is_tensor(heatmaps):
        heatmaps = heatmaps.detach().cpu().numpy()

    heatmap = heatmaps.max(axis=0)  # [H,W]

    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-6)
    heatmap = (heatmap * 255).astype(np.uint8)

    heatmap_color = cv2.applyColorMap(heatmap, colormap)

    # ------------------
    # Overlay
    # ------------------
    overlay = cv2.addWeighted(
        image, 1 - alpha,
        heatmap_color, alpha,
        0
    )

    return overlay
