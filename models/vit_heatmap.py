import torch
import torch.nn as nn
import timm


class ViTHeatmap(nn.Module):
    def __init__(
        self,
        backbone="vit_base_patch16_224",
        num_landmarks=5,
        heatmap_size=56,
        pretrained=True,
        in_chans=1,
    ):
        super().__init__()

        self.num_landmarks = num_landmarks
        self.heatmap_size = heatmap_size

        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            in_chans=in_chans,
            num_classes=0,
        )

        feat_dim = self.backbone.num_features

        self.head = nn.Sequential(
            nn.Linear(feat_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(
                1024,
                num_landmarks * heatmap_size * heatmap_size
            )
        )

    def forward(self, x):
        B = x.size(0)
        feat = self.backbone(x)
        out = self.head(feat)

        out = out.view(
            B,
            self.num_landmarks,
            self.heatmap_size,
            self.heatmap_size
        )
        return out
