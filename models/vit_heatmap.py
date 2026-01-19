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
