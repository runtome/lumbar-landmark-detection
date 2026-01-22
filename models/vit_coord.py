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
        freeze_backbone=True,   # 🔧 NEW
    ):
        super().__init__()

        self.num_landmarks = num_landmarks

        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            in_chans=in_chans,
            num_classes=0,
        )

        # 🔒 Freeze backbone (optional but recommended)
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

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
        return: [B, N, 2] in [0, 1]
        """
        feat = self.backbone(x)          # [B, C]
        out = self.head(feat)            # [B, 2N]
        out = out.view(-1, self.num_landmarks, 2)
        out = torch.sigmoid(out)         # ✅ FORCE [0,1]
        return out
