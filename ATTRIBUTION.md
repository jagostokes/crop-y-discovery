# Attribution

## AI-assisted code and prose

| Area                                                                 | Tool / model (if known) | What it helped with                                                                                           |
| -------------------------------------------------------------------- | ----------------------- | ------------------------------------------------------------------------------------------------------------- |
| Repository docs (`README.md`, `SETUP.md`, this file)                 | Cursor                  | Structure, wording, checklist alignment with course handout                                                   |
| Pandas 3.x compatibility patch (`pd.read_json` monkey-patch)         | Claude                  | Identifying that pandas 3.x no longer accepts raw JSON strings and suggesting the `io.StringIO` wrapper fix   |
| DataLoader construction                                              | Claude                  | Minor guidance on `random_split` usage and `worker_init_fn` for reproducible shuffling                        |
| Weather-to-embedding conversion (`WeatherMLP`, `_weather_to_vector`) | Claude                  | Minor assistance on structuring the MLP layers and padding/truncating the weather vector to a fixed dimension |

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

- **CropNet** — Sentinel-2 surface reflectance time series, HRRR-derived weather features, USDA county-level corn yield labels. License: CC BY-NC (research / non-commercial).

```bibtex
@misc{cropnet_dataset_2024,
    author    = { CropNet Dataset },
    title     = { CropNet (Revision 1dcfbe9) },
    year      = 2024,
    url       = { https://huggingface.co/datasets/CropNet/CropNet },
    doi       = { 10.57967/hf/3514 },
    publisher = { Hugging Face }
}
```

## Pretrained weights

- **ResNet18** trained on **ImageNet** via `torchvision.models` — used as a frozen or fine-tuned visual backbone depending on notebook settings.

---

[README](README.md) · [Setup](SETUP.md)
