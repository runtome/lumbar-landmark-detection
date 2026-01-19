# models/vit_coord.py
import torch
import torch.nn as nn
import timm


class ViTCoordRegressor(nn.Module):
    def __init__(
        self,
        backbone="vit_base_patch16_224",
        num_landmarks=5,
        pretrained=True,
        in_chans=1,
    ):
        super().__init__()

        self.num_landmarks = num_landmarks

        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            in_chans=in_chans,
            num_classes=0,  # remove classifier
        )

        feat_dim = self.backbone.num_features

        self.head = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, 2 * num_landmarks),
        )

    def forward(self, x):
        """
        x: [B, 1, H, W]
        return: [B, 5, 2]  (x, y)
        """
        feat = self.backbone(x)          # [B, C]
        out = self.head(feat)            # [B, 10]
        out = out.view(-1, self.num_landmarks, 2)
        return out
