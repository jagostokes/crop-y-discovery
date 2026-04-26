# Crop Yield Discovery

County-level corn yield prediction from public satellite and weather data. I took the CropNet setup (Sentinel-2 + HRRR + USDA labels), kept the temporal dimension instead of averaging it away, added a real weather branch and fusion head on top of a pretrained ResNet18 backbone, and compared everything against a simple mean-yield baseline so the numbers actually mean something.

## What it Does

The notebook trains a multimodal model: satellite sequences go through ResNet18 (ImageNet weights), weather vectors through a small MLP, then both get fused and regressed to yield in bushels per acre. Data are split by county so test counties never appear in training — which is harder than random row splits and is why a dumb baseline matters. Training uses early stopping, LR reduction on plateau, and light augmentation on the image side. Everything runs top-to-end in one Colab-style notebook with the CropNet retriever for pulling the public dataset.

## Quick Start

1. Read **[SETUP.md](SETUP.md)** for environment options (local vs Colab), CropNet access, and GPU notes.
2. Open **`notebooks/Crop_y_Discovery.ipynb`** and run cells in order from the top. The first cells install system deps (eccodes) where needed for HRRR.
3. Point the notebook at your CropNet data path (see SETUP) and run training through evaluation.

If you want script-style execution, the same notebook code has been mirrored into `src/cells/` (one file per code cell), with entrypoints:

- `python src/get-data.py`
- `python src/train-model.py`
- `python src/test-model.py`

## Video Links

Recordings live on YouTube (you can also drop `.mp4` files in `videos/` and link those instead). Swap in your real URLs:

- **Project demo** (3–5 min): `https://www.youtube.com/watch?v=YOUR_DEMO_VIDEO_ID`
- **Technical walkthrough** (5–10 min): `https://www.youtube.com/watch?v=YOUR_TECH_VIDEO_ID`

## Evaluation

Numbers below are from a completed run already saved in the notebook outputs (held-out test counties). Your exact figures may shift slightly with seed/hardware, but should be in the same ballpark if you follow the same split and hyperparameters.

The notebook uses **two different baselines**, and the README used to blur them together:

- **Global train-mean baseline:** one constant (mean training yield in BU/acre) predicted for every test sample. That is what the printed “skill score” and the middle column below refer to.
- **County historical-mean baseline:** for each test county-year, predict that county’s historical average yield (from `test_ds.hist_yields` / USDA history), which is a stronger sanity check than a single global number.

| Metric         | Model | Global train-mean baseline | County historical-mean baseline |
| -------------- | ----- | -------------------------- | ------------------------------- |
| RMSE (BU/acre) | 10.78 | 18.53                      | 17.42                           |
| MAE (BU/acre)  | 9.15  | 15.63                      | 14.55                           |
| R²             | 0.37  | −0.87                      | 0.18                            |

Skill vs the **global** baseline is about **42% lower RMSE**; the model also beats the **per-county historical** baseline on RMSE in this run (about **6.6 BU/acre** better).

Qualitatively: the model does better than both trivial predictors on this split, and the learning curves in the notebook show validation RMSE improving before early stopping. R² versus the global constant looks better than versus county history because the historical predictor already explains some geographic structure.

## Individual Contributions

**Angad Miglani**

- Discovered and implemented the HuggingFace-based Sentinel-2 download workaround, reverse-engineering the correct repo path structure after the official `DataDownloader` silently failed without Sentinel Hub credentials
- Diagnosed and patched the pandas 3.x incompatibility in the CropNet library via a `pd.read_json` monkey-patch so the pipeline runs on modern environments without touching the library source
- Built `MultimodalCropDataset` — assembles all three modalities (satellite sequences, weather vectors, USDA labels) into batched tensors, including variable-T truncation to handle differing observation counts across counties
- Implemented county-level 80/20 train/test split to prevent data leakage across geographic regions
- Ran all experiments, interpreted results, and created both all presentations required.

**Jago Stokes**

- Designed the multimodal model architecture: `SatelliteEncoder` (pretrained ResNet18 with temporal mean-pooling), `WeatherMLP`, and `MultiModalYieldNet` fusion head
- Implemented the training loop with early stopping, `ReduceLROnPlateau` LR scheduling, gradient clipping, and best-checkpoint restoration
- Set up `SequenceFlipAugment` for temporally-consistent data augmentation
- Built `MultimodalCropDataset` — assembles all three modalities (satellite sequences, weather vectors, USDA labels) into batched tensors, including variable-T truncation to handle differing observation counts across counties
- Established the two-baseline evaluation framework (global mean + county historical mean) and skill score comparison
- Implemented experiment logging system writing structured Markdown from checkpoint metadata
- Set up repository structure and video script.

---

**Also see:** [SETUP.md](SETUP.md) · [ATTRIBUTION.md](ATTRIBUTION.md)
