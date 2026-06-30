"""ViT-B/16 visual encoder for RGB and depth branches."""

import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights

VIT_FEAT_DIM = 768


class ViTEncoder(nn.Module):
    """ViT-B/16 backbone with identity classification head.

    Input:  x [B, 3, H, W]  (224x224 after external transforms)
    Output: feat [B, 768]
    """

    def __init__(self, device: torch.device, pretrained: bool = True):
        super().__init__()
        if pretrained:
            weights = ViT_B_16_Weights.DEFAULT
            self.vit = vit_b_16(weights=weights)
        else:
            self.vit = vit_b_16(weights=None)
        self.vit.heads = nn.Identity()
        self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.vit(x)


def build_visual_encoder(device: torch.device, pretrained: bool = True):
    """Return (encoder, per_view_feature_dim) for RGB/depth ViT branches."""
    encoder = ViTEncoder(device, pretrained=pretrained)
    return encoder, VIT_FEAT_DIM
