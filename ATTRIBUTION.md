# Attribution

## AI-assisted code and prose

Per course policy, generative AI was allowed on the final project. The notes below document where AI assistance was used in this repository.

| Area | Tool / model (if known) | What it helped with |
|------|-------------------------|---------------------|
| Repository docs (`README.md`, `SETUP.md`, this file) | Cursor | Structure, wording, checklist alignment with course handout |
| Code comments across `src/` and the notebook | Cursor / ChatGPT | Generated and refined explanatory comments/docstrings |
| Dataloader subsection (`src/` dataloader code) | Cursor | Drafted part of the dataloader implementation, then reviewed/edited by me |
| Notebook-to-module extraction (`src/cells/*.py`, pipeline wrappers) | Cursor | Converted each notebook code cell into a module and wired `src/get-data.py`, `src/train-model.py`, `src/test-model.py` |

You are responsible for correctness of all submitted work regardless of how it was drafted.

## External codebases

- **[CropNet](https://github.com/fudong03/CropNet)** — dataset, retrieval utilities, and baseline framing I built on top of. License: CC BY-NC (research / non-commercial).

## Python libraries

Core stack (see `requirements.txt` for pinned or minimum versions):

- **PyTorch** & **torchvision** — model, training loop, pretrained ResNet18  
- **NumPy**, **pandas** — arrays and tabular handling  
- **scikit-learn** — metrics (RMSE, MAE, R², etc.)  
- **matplotlib** — plots in the notebook  
- **xarray**, **cfgrib** (if used per notebook) — weather / GRIB reading  

Add any other imports you introduce (e.g. **tqdm**, **seaborn**) here when you add them to the project.

## Data

- **CropNet** — Sentinel-2 surface reflectance time series, HRRR-derived weather features, USDA county-level corn yield labels. Citation and terms of use: follow the official CropNet repository and any associated paper they specify.

## Pretrained weights

- **ResNet18** trained on **ImageNet** via `torchvision.models` — used as a frozen or fine-tuned visual backbone depending on notebook settings.

---

[README](README.md) · [Setup](SETUP.md)
