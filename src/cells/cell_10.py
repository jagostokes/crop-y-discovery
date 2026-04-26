"""Auto-extracted from notebooks/Crop_y_Discovery.ipynb (code cell 10).
This file mirrors notebook logic for script-style orchestration.
"""

# ---------------------------------------------------------------------------
# Weather MLP: encodes the HRRR feature summary into a 64-dim embedding.
# ---------------------------------------------------------------------------
class WeatherMLP(nn.Module):
    """
    Multi-layer perceptron over the seasonal weather feature vector.

    Input  shape: (B, in_dim)
    Output shape: (B, out_dim=64)

    Design notes:
      - Two hidden layers with BatchNorm + ReLU + Dropout. Dropout helps
        because the weather vector is small (~16 features) so the MLP would
        otherwise overfit easily on our small dataset.
      - Final dim is 64, smaller than the satellite embedding (512). This
        rough proportionality matches the relative information content:
        imagery has way more entropy than a short weather summary.
    """

    def __init__(self, in_dim: int, out_dim: int = 64, dropout: float = 0.3):
        super().__init__()
        # If in_dim is very small, scale hidden layers proportionally.
        h1 = max(64, in_dim * 4)
        h2 = max(64, in_dim * 2)
        self.net = nn.Sequential(
            nn.Linear(in_dim, h1),
            nn.BatchNorm1d(h1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h1, h2),
            nn.BatchNorm1d(h2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h2, out_dim),
        )

    def forward(self, w):
        return self.net(w)


# Smoke test.
_wmlp = WeatherMLP(in_dim=full_train_ds.weather_dim)
_dummy_w = torch.randn(2, full_train_ds.weather_dim)
with torch.no_grad():
    _out_w = _wmlp(_dummy_w)
print(f"WeatherMLP output shape: {tuple(_out_w.shape)}  (expect (2, 64))")
