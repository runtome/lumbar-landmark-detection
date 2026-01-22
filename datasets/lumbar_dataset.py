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
        heatmap_size=(56, 56),
        mode="coord",          # "coord" | "heatmap"
        sigma=2,
        transform=None,
    ):
        self.df = pd.read_csv(csv_path)
        self.image_root = image_root
        
        self.img_size = img_size
        self.mode = mode
        self.sigma = sigma
        self.transform = transform
        self.heatmap_size = heatmap_size
        # group annotations by image
        self.samples = list(self.df.groupby("filename"))

    def __len__(self):
        return len(self.samples)

    def _load_image(self, filename):
        path = os.path.join(self.image_root, filename)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(path)
        return img

    def _make_heatmap(self, x, y):
        H, W = self.heatmap_size
        xx, yy = np.meshgrid(np.arange(W), np.arange(H))
        heatmap = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * self.sigma ** 2))
        return heatmap.astype(np.float32)

    def __getitem__(self, idx):
        filename, rows = self.samples[idx]

        # --------------------
        # IMAGE
        # --------------------
        img = self._load_image(filename)

        if self.transform:
            img = self.transform(img)      # [1, H, W]
        else:
            img = torch.from_numpy(img).unsqueeze(0).float() / 255.0

        H, W = self.img_size

        # --------------------
        # LANDMARKS
        # --------------------
        coords = []
        heatmaps = []

        for level in LEVEL_ORDER:
            r = rows[rows.level == level]

            if len(r) == 0:
                coords.append([0.0, 0.0])
                heatmaps.append(np.zeros((H, W), dtype=np.float32))
            else:
                # normalized GT from CSV
                x_norm = float(r.relative_x.values[0])
                y_norm = float(r.relative_y.values[0])

                # clamp safety
                x_norm = np.clip(x_norm, 0.0, 1.0)
                y_norm = np.clip(y_norm, 0.0, 1.0)

                coords.append([x_norm, y_norm])

                # 🔥 convert to HEATMAP resolution (NOT image resolution)
                x_pix = x_norm * self.heatmap_size
                y_pix = y_norm * self.heatmap_size
                heatmaps.append(self._make_heatmap(x_pix, y_pix))

        coords = torch.tensor(coords, dtype=torch.float32)
        heatmaps = torch.tensor(np.stack(heatmaps), dtype=torch.float32)

        if self.mode == "coord":
            return img, coords
        if self.mode == "heatmap":
             return img, heatmaps
        else:
            return img, heatmaps




# FAST tensor conversion (fixes warning)
# coords = torch.from_numpy(np.array(coords, dtype=np.float32))
# heatmaps = torch.from_numpy(np.stack(heatmaps)).float()