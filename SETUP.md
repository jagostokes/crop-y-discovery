# Setup

## What you need

- Python 3.10+ (3.11 is fine)
- A GPU strongly recommended for ResNet18 + data loading at sane batch sizes (Google Colab T4 works; local CUDA is better if you have it)
- Disk space for CropNet (imagery + weather + labels — plan for tens of GB depending on how much you download)

## Dependencies

From the repo root:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

If you prefer conda, you can `conda install` the same packages or adapt this into an `environment.yml` later.

## CropNet data

This project uses **[CropNet](https://github.com/fudong03/CropNet)** (Sentinel-2, HRRR, USDA yields). It is **CC BY-NC, research-only** — read their license before redistributing anything.

1. Follow the CropNet repo instructions to obtain credentials / paths for their data retriever.
2. In `notebooks/Crop_y_Discovery.ipynb`, set the paths or drive mounts where the notebook expects the dataset (the notebook cells that build `MultimodalCropDataset` are the source of truth).

Put any local copies of raw or processed files under **`data/`** (gitignored if large) so graders know where to look. Do not commit multi-gigabyte binaries to GitHub.

## System packages (HRRR / GRIB)

The notebook installs **eccodes** (via `apt` on Colab / Debian-style systems) so Python can read HRRR GRIB. On macOS with Homebrew:

```bash
brew install eccodes
```

If `cfgrib` / `xarray` throws import errors, the fix is almost always “install eccodes for your OS, then reinstall the Python stack.”

## Trained weights

Checkpoints should be written under **`models/`** (or Google Drive if you use the Colab persistence path in the notebook). Add `models/` patterns to `.gitignore` if files are huge; for grading, either include a small exported checkpoint or document how to reproduce training in the technical video.

## For graders

1. Clone the repo.
2. Install deps and system libraries above.
3. Obtain CropNet access per upstream docs.
4. Open `notebooks/Crop_y_Discovery.ipynb`, adjust data paths in the early configuration cells, run all.

If anything fails on your machine, the error message from the first failing cell plus your Python version is enough for me to help debug — paste those into course Ed or email.

---

[Back to README](README.md) · [Attribution](ATTRIBUTION.md)
