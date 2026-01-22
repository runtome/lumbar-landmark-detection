import torch
import torch.nn as nn
import timm

class ViTHeatmap(nn.Module):
    def __init__(self, backbone="vit_base_patch16_224", num_landmarks=5):
        super().__init__()

        self.backbone = timm.create_model(
            backbone, pretrained=True, in_chans=1, num_classes=0
        )

        self.embed_dim = self.backbone.num_features
        self.num_landmarks = num_landmarks

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(self.embed_dim, 256, 2, 2),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 2, 2),
            nn.ReLU(),
            nn.Conv2d(128, num_landmarks, 1),
        )

    def forward(self, x):
        B = x.size(0)
        tokens = self.backbone.patch_embed(x)
        h = w = int(tokens.shape[1] ** 0.5)

        tokens = tokens.transpose(1, 2).reshape(B, -1, h, w)
        return self.decoder(tokens)


class ViTHeatmapRegressor(nn.Module):
    def __init__(
        self,
        backbone="vit_base_patch16_224",
        num_landmarks=5,
        pretrained=True,
        in_chans=1,
        img_size=224,
    ):
        super().__init__()

        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            in_chans=in_chans,
            num_classes=0,
        )

        C = self.backbone.num_features

        self.head = nn.Sequential(
            nn.Linear(C, 512),
            nn.ReLU(),
            nn.Linear(512, num_landmarks * img_size * img_size),
        )

        self.num_landmarks = num_landmarks
        self.img_size = img_size

    def forward(self, x):
        B = x.size(0)
        feat = self.backbone(x)
        out = self.head(feat)
        return out.view(B, self.num_landmarks, self.img_size, self.img_size)
