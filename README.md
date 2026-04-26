# Crop Yield Discovery

County-level corn yield prediction from public satellite and weather data. I took the CropNet setup (Sentinel-2 + HRRR + USDA labels), kept the temporal dimension instead of averaging it away, added a real weather branch and fusion head on top of a pretrained ResNet18 backbone, and compared everything against a simple mean-yield baseline so the numbers actually mean something.

## What it Does

The notebook trains a multimodal model: satellite sequences go through ResNet18 (ImageNet weights), weather vectors through a small MLP, then both get fused and regressed to yield in bushels per acre. Data are split by county so test counties never appear in training — which is harder than random row splits and is why a dumb baseline matters. Training uses early stopping, LR reduction on plateau, and light augmentation on the image side. Everything runs top-to-end in one Colab-style notebook with the CropNet retriever for pulling the public dataset.

## Quick Start

1. Read **[SETUP.md](SETUP.md)** for environment options (local vs Colab), CropNet access, and GPU notes.
2. Open **`notebooks/Crop_y_Discovery.ipynb`** and run cells in order from the top. The first cells install system deps (eccodes) where needed for HRRR.
3. Point the notebook at your CropNet data path (see SETUP) and run training through evaluation.

There is no separate CLI entrypoint yet — the notebook *is* the runnable project.

## Video Links

Recordings live on YouTube (you can also drop `.mp4` files in `videos/` and link those instead). Swap in your real URLs:

- **Project demo** (3–5 min): `https://www.youtube.com/watch?v=YOUR_DEMO_VIDEO_ID`
- **Technical walkthrough** (5–10 min): `https://www.youtube.com/watch?v=YOUR_TECH_VIDEO_ID`

## Evaluation

Numbers below are from a completed run already saved in the notebook outputs (held-out test counties, global-mean baseline). Your exact figures may shift slightly with seed/hardware, but should be in the same ballpark if you follow the same split and hyperparameters.

| Metric | Model | Baseline (constant train mean) |
|--------|-------|--------------------------------|
| RMSE (BU/acre) | 10.78 | 18.53 |
| MAE (BU/acre) | 9.15 | 15.63 |
| R² | 0.37 | −0.87 |
| Skill vs baseline | ~42% lower RMSE | — |

Qualitatively: predictions track actual yields better than guessing the same number for every county-year, and the learning curves in the notebook show validation RMSE improving before early stopping kicks in. R² is modest — small geographic sample, hard transfer to unseen counties — which is exactly why I report the baseline side by side.

## Individual Contributions

Solo project — design, implementation, experiments, and write-ups are all mine.

---

**Also see:** [SETUP.md](SETUP.md) · [ATTRIBUTION.md](ATTRIBUTION.md)
