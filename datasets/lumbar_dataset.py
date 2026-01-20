import os
import cv2
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset

LEVEL_ORDER = ["L1/L2", "L2/L3", "L3/L4", "L4/L5", "L5/S1"]


class LumbarDataset(Dataset):
    def __init__(
        self,
        csv_path,
        image_root,
        img_size=(224, 224),
        mode="coord",          # "coord" | "heatmap"
        sigma=4,
        transform=None,
    ):
        self.df = pd.read_csv(csv_path)
        self.image_root = image_root
        self.img_size = img_size
        self.mode = mode
        self.sigma = sigma
        self.transform = transform

        # group by image
        self.samples = list(self.df.groupby("filename"))

    def __len__(self):
        return len(self.samples)

    def _load_image(self, filename):
        path = os.path.join(self.image_root, filename)

        if not os.path.exists(path):
            raise FileNotFoundError(f"Image not found: {path}")

        # grayscale image
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            raise ValueError(f"Failed to load image: {path}")

        return img  # H x W (uint8)

    def _make_heatmap(self, x, y):
        H, W = self.img_size
        xx, yy = np.meshgrid(np.arange(W), np.arange(H))
        return np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * self.sigma ** 2))

    def __getitem__(self, idx):
        filename, rows = self.samples[idx]

        # ---- image ----
        img = self._load_image(filename)

        # apply transform (Resize, ToTensor, Normalize)
        if self.transform is not None:
            img = self.transform(img)     # [1, H, W]
        else:
            img = torch.from_numpy(img).unsqueeze(0).float() / 255.0

        # ---- landmarks ----
        coords = []
        heatmaps = []

        for level in LEVEL_ORDER:
            r = rows[rows.level == level]

            if len(r) == 0:
                coords.append([0.0, 0.0])
                heatmaps.append(np.zeros(self.img_size, dtype=np.float32))
            else:
                x = r.relative_x.values[0] * self.img_size[1]
                y = r.relative_y.values[0] * self.img_size[0]
                coords.append([x, y])
                heatmaps.append(self._make_heatmap(x, y))

        # FAST tensor conversion (fixes warning)
        coords = torch.from_numpy(np.array(coords, dtype=np.float32))
        heatmaps = torch.from_numpy(np.stack(heatmaps)).float()

        if self.mode == "coord":
            return img, coords
        else:
            return img, heatmaps
