"""Auto-extracted from notebooks/Crop_y_Discovery.ipynb (code cell 5).
This file mirrors notebook logic for script-style orchestration.
"""

# ---------------------------------------------------------------------------
# MultimodalCropDataset
# ---------------------------------------------------------------------------
# A PyTorch Dataset that returns (image_sequence, weather_features, yield)
# for each sample. Now includes a Historical Yield Baseline parameter!
# ---------------------------------------------------------------------------
import numpy as np
import torch
import pandas as pd
import os
import glob
from torch.utils.data import Dataset
from cropnet.data_retriever import DataRetriever


def _retrieve_weather_custom(base_dir, fips_codes, years):
    """
    Custom weather retriever that reads the pre-computed daily summaries directly
    from the WRF-HRRR Computed Dataset directory we downloaded them to.
    """
    weather_dict = {}
    state_map = {"17": "IL", "18": "IN", "19": "IA", "26": "MI", "39": "OH"}

    for fips in fips_codes:
        sc = fips[:2]
        if sc not in state_map:
            continue
        abbr = state_map[sc]

        weather_dict[fips] = {}
        for year in years:
            year_dfs = []
            for month in range(1, 13):
                fname = f"HRRR_{sc}_{abbr}_{year}-{month:02d}.csv"
                path = os.path.join(base_dir, "WRF-HRRR Computed Dataset", "data", year, abbr, fname)
                if os.path.exists(path):
                    try:
                        df = pd.read_csv(path)
                        # Filter for the specific county FIPS
                        county_df = df[df["county_ansi"].astype(str).str.zfill(3) == fips[2:]]
                        if not county_df.empty:
                            # Drop metadata columns to keep only features
                            feat_cols = [c for c in county_df.columns if c not in ("fips", "year", "state_ansi", "county_ansi", "Unnamed: 0")]
                            year_dfs.append(county_df[feat_cols].mean(axis=0).to_numpy(dtype=np.float32))
                    except Exception as e:
                        pass
            if year_dfs:
                # Average the monthly means to get a yearly summary feature vector
                weather_dict[fips][year] = np.mean(year_dfs, axis=0)

    return weather_dict


def _weather_to_vector(weather_data, fips, year, expected_dim=None):
    """
    Convert weather data for one (fips, year) into a fixed-size float vector.
    """
    vec = None

    if fips in weather_data and year in weather_data[fips]:
        vec = weather_data[fips][year]

    # Fallback: zeros (lets the model still train as satellite-only).
    if vec is None:
        vec = np.zeros(expected_dim or 16, dtype=np.float32)

    # Standardize length across samples by pad-or-truncate.
    if expected_dim is not None and len(vec) != expected_dim:
        if len(vec) < expected_dim:
            vec = np.pad(vec, (0, expected_dim - len(vec)))
        else:
            vec = vec[:expected_dim]
    return vec


class MultimodalCropDataset(Dataset):
    """
    Multimodal CropNet dataset returning (image_sequence, weather+history, yield).
    """

    def __init__(self, fips_codes, years, base_dir, crop,
                 transform=None,
                 yield_mean=None, yield_std=None,
                 weather_dim=None):

        self.transform = transform
        retriever = DataRetriever(base_dir=base_dir)

        # Pull all three modalities from disk.
        usda_df  = retriever.retrieve_USDA(crop, fips_codes, years)
        s2_dict  = retriever.retrieve_Sentinel2(fips_codes, years, image_type="AG")
        weather_data = _retrieve_weather_custom(base_dir, fips_codes, years)

        # --- NEW: Get historical yield data ---
        hist_years = [str(y) for y in range(2010, int(years[0]))]
        hist_df = retriever.retrieve_USDA(crop, fips_codes, hist_years)
        self.hist_yields = {}
        if hist_df is not None and not hist_df.empty:
            for i, row in hist_df.iterrows():
                f = str(row["state_ansi"]).zfill(2) + str(row["county_ansi"]).zfill(3)
                if f not in self.hist_yields:
                    self.hist_yields[f] = []
                self.hist_yields[f].append(float(row["YIELD, MEASURED IN BU / ACRE"]))

        # Average it out
        for f in self.hist_yields:
            self.hist_yields[f] = np.mean(self.hist_yields[f])

        global_hist_mean = np.mean(list(self.hist_yields.values())) if self.hist_yields else 150.0
        # --------------------------------------

        raw = []
        seen_weather_lens = []

        # Find the minimum sequence length (T) across all counties
        available_T = [arr.shape[0] for arr in s2_dict.values() if len(arr.shape) == 5]
        min_t = min(available_T) if available_T else 0
        print(f"Truncating all satellite sequences to T={min_t} frames to ensure equal sizing.")

        for _, row in usda_df.iterrows():
            fips = str(row["state_ansi"]).zfill(2) + str(row["county_ansi"]).zfill(3)
            yield_val = float(row["YIELD, MEASURED IN BU / ACRE"])
            year_val  = str(row.get("year", years[0]))

            if fips not in s2_dict:
                continue

            arr = s2_dict[fips].astype(np.float32) / 255.0
            # Enforce identical sequence length by truncating
            arr = arr[:min_t]

            # Calculate the expected dimension for just the weather portion
            # (total weather_dim minus the 1 historical slot)
            w_expected = weather_dim - 1 if weather_dim is not None else None
            wvec = _weather_to_vector(weather_data, fips, year_val, expected_dim=w_expected)

            # --- NEW: Append historical yield vector to weather features ---
            hist_y = self.hist_yields.get(fips, global_hist_mean)
            # Scale by 1/100 to keep it roughly in the range of other features
            hist_feat = np.array([hist_y / 100.0], dtype=np.float32)

            wvec = np.concatenate([wvec, hist_feat])
            # ------------------------------------------------------------

            seen_weather_lens.append(len(wvec))

            for p in range(arr.shape[1]):
                img_seq = torch.from_numpy(arr[:, p]).permute(0, 3, 1, 2)
                w_tensor = torch.from_numpy(wvec).float()
                raw.append((img_seq, w_tensor, yield_val))

        if len(raw) == 0:
            raise RuntimeError("No samples loaded. Check that downloads completed.")

        yields = np.array([y for _, _, y in raw], dtype=np.float32)
        self.yield_mean = float(yields.mean()) if yield_mean is None else yield_mean
        self.yield_std  = float(yields.std())  if yield_std  is None else yield_std
        if self.yield_std < 1e-6:
            self.yield_std = 1.0

        self.weather_dim = max(seen_weather_lens) if seen_weather_lens else 21

        self.samples = [
            (
                img_seq,
                wvec,
                torch.tensor((y - self.yield_mean) / self.yield_std, dtype=torch.float32)
            )
            for img_seq, wvec, y in raw
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_seq, weather, yld = self.samples[idx]
        if self.transform is not None:
            img_seq = self.transform(img_seq)
        return img_seq, weather, yld


print("MultimodalCropDataset defined with Historical Yield Baseline.")
