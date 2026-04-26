# Trained weights (`models/`)

## Committed checkpoint

**`multimodal_yield_corn.pt`** (~43 MB on disk, fine for GitHub) is **tracked in git** on purpose so graders can load weights without retraining. It is the same dict the notebook saves: `model_state`, `yield_mean`, `yield_std`, `weather_dim`, `history`, `test_metrics`, `hyperparameters`, county lists, years, etc.

Example load (after you define `MultiModalYieldNet` with the right `weather_dim`):

```python
import torch
ckpt = torch.load("models/multimodal_yield_corn.pt", map_location="cpu")
```

## Other files in `models/`

Anything else under **`models/`** stays **gitignored** by default (`models/*` in the root `.gitignore`), except `.gitkeep`, this README, and **`multimodal_yield_corn.pt`**.

If you retrain and produce a new corn checkpoint, replace **`multimodal_yield_corn.pt`** and commit the updated file (or extend `.gitignore` with another `!models/...` line if you add more named artifacts).
