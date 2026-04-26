"""Auto-extracted from notebooks/Crop_y_Discovery.ipynb (code cell 11).
This file mirrors notebook logic for script-style orchestration.
"""

# ---------------------------------------------------------------------------
# Multimodal fusion: combine satellite + weather embeddings -> yield prediction.
# ---------------------------------------------------------------------------
class MultiModalYieldNet(nn.Module):
    """
    The full model. Takes (img_seq, weather) and predicts yield (z-score).

    Input shapes:
      img_seq : (B, T, 3, 224, 224)
      weather : (B, weather_dim)
    Output shape:
      yield_pred: (B,)  — the predicted z-score normalized yield.

    Architecture:
      img_seq  --> SatelliteEncoder --> (B, 512) ─┐
                                                   ├─> concat -> (B, 576) -> MLP head -> (B,)
      weather  --> WeatherMLP -------> (B,  64) ──┘

    The fusion strategy here is "late fusion" — each modality is encoded
    independently, then the final embeddings are concatenated. This is the
    most common multimodal approach because it lets each branch specialize.
    Alternatives (cross-attention, gating) might help but add complexity
    and need more data.
    """

    def __init__(self, weather_dim: int,
                 sat_embed_dim: int = 512,
                 weather_embed_dim: int = 64,
                 head_dropout: float = 0.3):
        super().__init__()
        self.satellite_enc = SatelliteEncoder(embed_dim=sat_embed_dim)
        self.weather_enc   = WeatherMLP(in_dim=weather_dim,
                                         out_dim=weather_embed_dim)

        # Fusion head: concat -> linear -> ReLU -> dropout -> linear -> scalar.
        fused_dim = sat_embed_dim + weather_embed_dim
        self.head = nn.Sequential(
            nn.Linear(fused_dim, 128),
            nn.ReLU(),
            nn.Dropout(head_dropout),
            nn.Linear(128, 1),
        )

    def forward(self, img_seq, weather):
        sat_emb     = self.satellite_enc(img_seq)   # (B, 512)
        weather_emb = self.weather_enc(weather)      # (B, 64)
        fused = torch.cat([sat_emb, weather_emb], dim=1)  # (B, 576)
        # Squeeze the trailing dim of the (B, 1) output to give (B,).
        # This matches the shape of the target tensor for MSE loss.
        return self.head(fused).squeeze(-1)


# Build the model and report parameter count.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MultiModalYieldNet(weather_dim=full_train_ds.weather_dim).to(device)

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Device: {device}")
print(f"Total trainable parameters: {n_params:,}")
print()
print(model)
