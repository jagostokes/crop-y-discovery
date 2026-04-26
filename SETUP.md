# Setup

This is the path I actually use end-to-end: Python venv, system **eccodes** for GRIB, `pip install -r requirements.txt`, CropNet + Hugging Face data under **`data/cropnet_data/`** (configurable), checkpoints under **`models/`**.

---

## What you need

Honestly just run all on colab: https://colab.research.google.com/drive/1Lgc-mj6RCm8FWGDDBrCRVia_jKj3xoTV?usp=sharing, but if not do this: 

- **Python 3.10+** (3.11 is what I tested with; avoid 3.13+ unless you enjoy being the first person to hit wheel bugs).
- **GPU** strongly recommended for ResNet18 training at a sane batch size (Colab T4 is fine; local CUDA is nicer if you have it). CPU will run for debugging but training is slow.
- **Disk**: CropNet slices for the counties/years in the notebook are large — plan for **tens of GB** depending on how much you pull from Hugging Face.
- **Network**: first run downloads USDA tables, HRRR CSVs, and Sentinel `.h5` shards from **`CropNet/CropNet`** on Hugging Face (dataset is public; no token required for read in normal cases).

---

## 1) System libraries (GRIB / eccodes)

HRRR paths in this project go through **GRIB** tooling. You need **eccodes** on the machine, not only Python wheels.

**macOS (Homebrew):**

```bash
brew install eccodes
```

**Debian / Ubuntu / Colab:**

```bash
sudo apt-get update && sudo apt-get install -y libeccodes-dev
```

**If `cfgrib` / `xarray` still cannot open GRIB:** install the Python stack *after* eccodes is on the PATH, then retry. On some platforms the CropNet docs suggest `ecmwflibs` as a fallback; only add that if you hit import errors you cannot fix with the system library.

---

## 2) Python environment

From the **repository root**:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt` includes **torch**, **torchvision**, **cropnet**, **pygrib**, **huggingface_hub**, and the usual numeric/plotting stack so you are not relying on commented-out `%pip` lines in the notebook.

**Note:** `pygrib` compiles against eccodes — if `pip install pygrib` fails, fix eccodes first, then rerun `pip install -r requirements.txt`.

---

## 3) Where data lands (important)

The notebook and `src/cells/cell_04.py` set:

- **`TARGET_DIR`** = `CROPY_TARGET_DIR` environment variable, **or** by default **`<repo>/data/cropnet_data`** (the directory is created if missing).

So on a fresh clone, downloads go under **`data/cropnet_data/`** as long as you start Jupyter or your shell from the repo root (or from `notebooks/` — the config cell resolves the repo root the same way as for checkpoints).

**Colab** (optional): if you prefer the old path:

```python
import os
os.environ["CROPY_TARGET_DIR"] = "/content/cropnet_data"
```

…in the first cell, *before* the download/config cell runs, or export it in the shell before `python src/get-data.py`.

CropNet is **CC BY-NC** — keep their license in mind. More context: **[data/README.md](data/README.md)**.

---

## 4) Where checkpoints land

Training saves to **`models/multimodal_yield_<crop>.pt`** (for corn: **`models/multimodal_yield_corn.pt`**), unless you set:

```bash
export CROPY_CKPT_DIR=/absolute/path/to/some/folder
```

**Browser “(2)” downloads:** if you save a duplicate like `multimodal_yield_corn (2).pt`, rename it to **`multimodal_yield_corn.pt`** before committing so it matches the training cell and this repo’s tracked file.

This repo **includes** **`models/multimodal_yield_corn.pt`** (~43 MB) in git for grading. Other ad-hoc `.pt` files under **`models/`** are still ignored unless you add another `!models/...` rule — see **[models/README.md](models/README.md)**.

---

## 5) Run the project

**Primary path (recommended):** open **`notebooks/Crop_y_Discovery.ipynb`**, choose a GPU runtime if you can, then **Run All** from the top. The first code cell still shows Colab `apt` / `%pip` lines for reference; on a local machine you already satisfied those via sections 1–2.

**Optional scripts** (same logic as the notebook, sequential cell modules):

```bash
python src/get-data.py
python src/train-model.py
python src/test-model.py
```

Notebook magics (`!`, `%`) are commented inside `src/cells/cell_01.py` — do not expect the scripts to `apt-get` for you.

---

## For graders

1. Clone the repo.
2. Install **eccodes** (section 1), then create the venv and **`pip install -r requirements.txt`** (section 2).
3. Open **`notebooks/Crop_y_Discovery.ipynb`**, run all. First full run downloads data into **`data/cropnet_data/`** and can take a long time.
4. If you cannot download data, ask the submitter for a **hosted checkpoint** or run their technical video; with a `.pt` in **`models/`**, you can still inspect `torch.load` keys and match shapes without retraining.

If a cell fails, send the **traceback + Python version + whether eccodes installed** — that is enough to narrow it down.

---

[Back to README](README.md) · [Attribution](ATTRIBUTION.md)
