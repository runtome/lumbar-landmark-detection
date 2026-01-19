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

        self.samples = self.df.groupby("filename")

    def __len__(self):
        return len(self.samples)

    def _load_image(self, row):
        path = os.path.join(self.image_root, row.filename)

        if not os.path.exists(path):
            raise FileNotFoundError(f"Image not found: {path}")

        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, self.img_size)
        img = img.astype(np.float32) / 255.0
        return img[None, ...]

    def _make_heatmap(self, x, y):
        H, W = self.img_size
        xx, yy = np.meshgrid(np.arange(W), np.arange(H))
        return np.exp(-((xx - x)**2 + (yy - y)**2) / (2 * self.sigma**2))

    def __getitem__(self, idx):
        _, rows = list(self.samples)[idx]
        row0 = rows.iloc[0]

        img = self._load_image(row0)

        coords = []
        heatmaps = []

        for level in LEVEL_ORDER:
            r = rows[rows.level == level]
            if len(r) == 0:
                coords.append([0.0, 0.0])
                heatmaps.append(np.zeros(self.img_size))
            else:
                x = r.relative_x.values[0] * self.img_size[1]
                y = r.relative_y.values[0] * self.img_size[0]
                coords.append([x, y])
                heatmaps.append(self._make_heatmap(x, y))

        coords = torch.tensor(coords, dtype=torch.float32)
        heatmaps = torch.tensor(heatmaps, dtype=torch.float32)

        img = torch.tensor(img, dtype=torch.float32)

        if self.mode == "coord":
            return img, coords
        else:
            return img, heatmaps
