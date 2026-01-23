# tools\visualization_utils.py
import numpy as np
import cv2
import torch

def draw_landmarks(
    image,            # torch [1,H,W]
    gt=None,           # [N,2] normalized (torch or numpy)
    pred=None,         # [N,2] normalized (torch or numpy)
    radius=4
):
    img = image.squeeze().cpu().numpy()
    img = (img * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    H, W = img.shape[:2]

    def to_px(coords):
        if torch.is_tensor(coords):
            coords = coords.detach().cpu().clone().numpy()
        else:
            coords = coords.copy()

        coords[:, 0] *= W
        coords[:, 1] *= H
        return coords

    if gt is not None:
        gt = to_px(gt)
        for x, y in gt:
            cv2.circle(img, (int(x), int(y)), radius, (0, 255, 0), -1)

    if pred is not None:
        pred = to_px(pred)
        for x, y in pred:
            cv2.circle(img, (int(x), int(y)), radius, (255, 0, 0), -1)

    return img

# def draw_landmarks(
#     image,            # [1, H, W] tensor
#     gt=None,           # [N, 2] normalized
#     pred=None,         # [N, 2] normalized
#     radius=4
# ):
#     img = image.squeeze().cpu().numpy()
#     img = (img * 255).astype(np.uint8)
#     img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

#     H, W = img.shape[:2]

#     def to_px(coords):
#         coords = coords.clone()
#         coords[:, 0] *= W
#         coords[:, 1] *= H
#         return coords

#     if gt is not None:
#         gt = to_px(gt)
#         for (x, y) in gt:
#             cv2.circle(img, (int(x), int(y)), radius, (0, 255, 0), -1)

#     if pred is not None:
#         pred = to_px(pred)
#         for (x, y) in pred:
#             cv2.circle(img, (int(x), int(y)), radius, (255, 0, 0), -1)

#     return img


LEVEL_ORDER = ["L1/L2", "L2/L3", "L3/L4", "L4/L5", "L5/S1"]

def draw_heatmaps_on_image(
    image,          # torch [1,H,W] or numpy [H,W]
    heatmaps,       # torch [5,h,w]
    alpha=0.4
):
    # -------------------------
    # IMAGE → numpy BGR
    # -------------------------
    if torch.is_tensor(image):
        image = image.squeeze().cpu().numpy()

    image = (image * 255).astype(np.uint8)

    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    H, W = image.shape[:2]
    overlay = image.copy()

    # -------------------------
    # HEATMAPS
    # -------------------------
    if torch.is_tensor(heatmaps):
        heatmaps = heatmaps.cpu().numpy()

    for i, level in enumerate(LEVEL_ORDER):
        hm = heatmaps[i]

        # resize heatmap → image size
        hm = cv2.resize(hm, (W, H))

        # normalize to [0,255]
        hm = hm - hm.min()
        hm = hm / (hm.max() + 1e-6)
        hm = (hm * 255).astype(np.uint8)

        # apply colormap
        hm_color = cv2.applyColorMap(hm, cv2.COLORMAP_JET)

        # blend
        overlay = cv2.addWeighted(overlay, 1.0, hm_color, alpha, 0)

    return overlay