"""Auto-extracted from notebooks/Crop_y_Discovery.ipynb (code cell 9).
This file mirrors notebook logic for script-style orchestration.
"""

# ---------------------------------------------------------------------------
# Satellite encoder: pretrained ResNet18 + temporal pooling.
# ---------------------------------------------------------------------------
import torch.nn as nn
import torchvision.models as tv_models


class SatelliteEncoder(nn.Module):
    """
    Encodes a sequence of satellite images into a single feature vector.

    Input  shape: (B, T, 3, 224, 224)
    Output shape: (B, embed_dim=512)

    Process:
      1. Reshape (B, T, 3, H, W) -> (B*T, 3, H, W) so we can run all frames
         through the CNN in one batched forward pass.
      2. ResNet18 produces (B*T, 512) embeddings.
      3. Reshape back to (B, T, 512) and mean-pool over T.

    Mean-pooling over time is the simplest temporal aggregation. It assumes
    each timestep contributes equally to the final yield, which isn't quite
    right (mid-season matters more than late winter), but it's a strong
    starting point. To upgrade: replace mean with a learned attention weight
    per timestep, or pass the (B, T, 512) sequence through a small GRU.
    """

    def __init__(self, embed_dim: int = 512, pretrained: bool = True):
        super().__init__()
        # ResNet18 is small (~11M params), well-suited to small datasets.
        # weights="DEFAULT" pulls the ImageNet-pretrained weights from torchvision.
        weights = tv_models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = tv_models.resnet18(weights=weights)

        # Replace the final FC layer with Identity so we get the 512-dim
        # feature vector instead of 1000-class logits.
        # (resnet18.fc is the last layer; .in_features = 512.)
        self.feature_dim = backbone.fc.in_features  # 512
        backbone.fc = nn.Identity()
        self.backbone = backbone

        # Sanity assertion: caller's expected embed_dim must match ResNet's.
        assert embed_dim == self.feature_dim, (
            f"SatelliteEncoder expects embed_dim={self.feature_dim} "
            f"(ResNet18's penultimate dim) but got {embed_dim}."
        )

    def forward(self, x):
        # x: (B, T, 3, 224, 224)
        B, T, C, H, W = x.shape

        # Flatten batch and time so all frames go through ResNet at once.
        # This is much faster than looping over T.
        x = x.view(B * T, C, H, W)

        # ResNet forward: (B*T, 3, 224, 224) -> (B*T, 512)
        feats = self.backbone(x)

        # Restore time axis: (B, T, 512), then mean-pool over T -> (B, 512).
        feats = feats.view(B, T, self.feature_dim)
        feats = feats.mean(dim=1)
        return feats


# Quick smoke test.
_enc = SatelliteEncoder()
_dummy = torch.randn(2, 4, 3, 224, 224)  # batch=2, T=4 frames
with torch.no_grad():
    _out = _enc(_dummy)
print(f"SatelliteEncoder output shape: {tuple(_out.shape)}  (expect (2, 512))")
