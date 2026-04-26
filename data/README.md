# Data (not in this repo)

The CropNet stack here is **satellite time series + HRRR weather + USDA county yields**. That is **too large** to ship inside a GitHub repo (often tens of GB depending on counties and years), so this folder stays **empty in git** on purpose. You download material locally or on Colab and point the code at it.

## Where the code expects data

The notebook and the mirrored modules under `src/cells/` use a single base path, **`TARGET_DIR`**, set in the configuration cell (see `src/cells/cell_04.py` and the same block in `notebooks/Crop_y_Discovery.ipynb`). By default that path is aimed at Colab (`/content/cropnet_data`). For your machine, set it to something under this directory if you want, e.g. `data/cropnet_raw`, or any absolute path where you stored the CropNet tree.

## How it gets there

1. **`cropnet.DataDownloader`** — pulls the packaged CropNet files into `TARGET_DIR` (WRF-HRRR layout, Sentinel layout, etc.), per the upstream CropNet instructions and credentials.
2. **`cropnet.DataRetriever`** — reads from that same `base_dir` (same as `TARGET_DIR`) when the dataset class and USDA/history steps need files.

So: **nothing magic in `data/`** except this note; the **source of truth** for paths is still that config cell. After a successful download, your tree should look like CropNet’s expected folders under whatever you set as `TARGET_DIR`.

Licensing: CropNet is **CC BY-NC** — keep their terms in mind if you redistribute derivatives.

See **[SETUP.md](../SETUP.md)** for install, `eccodes`, and how graders can wire paths on their side.
