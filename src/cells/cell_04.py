"""Auto-extracted from notebooks/Crop_y_Discovery.ipynb (code cell 4).
This file mirrors notebook logic for script-style orchestration.
"""

# ---------------------------------------------------------------------------
# Configuration: which counties, which years, which crop.
# ---------------------------------------------------------------------------
from cropnet.data_downloader import DataDownloader
from huggingface_hub import hf_hub_download
import os, shutil

TARGET_DIR = "/content/cropnet_data"
CROP       = "Corn"

# Train counties span 5 states in the US Corn Belt.
FIPS_TRAIN = ["17019", "17043", "19013", "19015", "26065", "18019", "39003"]
# Test counties: held-out geography in IL, IA, OH.
FIPS_TEST  = ["17115", "19163", "39101"]
FIPS_ALL   = FIPS_TRAIN + FIPS_TEST
YEARS      = ["2021", "2022"]
HISTORICAL_YEARS = [str(y) for y in range(2010, int(YEARS[0]))]

# --- USDA yield records ------------------------------------------------------
downloader = DataDownloader(target_dir=TARGET_DIR)
downloader.download_USDA(CROP, fips_codes=FIPS_ALL, years=YEARS)

print("Downloading historical USDA data...")
for y in HISTORICAL_YEARS:
    try:
        downloader.download_USDA(CROP, fips_codes=FIPS_ALL, years=[y])
    except TypeError:
        print(f"Skipping year {y}: not supported by cropnet USDA mapping.")

# --- HRRR weather (Pre-computed from Hugging Face) ---------------------------
# Instead of downloading raw hourly GRIB files, we download the daily
# pre-computed summaries directly from the CropNet Hugging Face repo.
STATE_MAP = {"17": "IL", "18": "IN", "19": "IA", "26": "MI", "39": "OH"}
state_codes = sorted({f[:2] for f in FIPS_ALL})

print("Checking pre-computed weather downloads...")
MONTHS = [f"{m:02d}" for m in range(1, 13)]
for sc in state_codes:
    abbr = STATE_MAP[sc]
    for year in YEARS:
        for month in MONTHS:
            fname = f"HRRR_{sc}_{abbr}_{year}-{month}.csv"
            hf_path = f"WRF-HRRR Computed Dataset/data/{year}/{abbr}/{fname}"
            local_path = os.path.join(TARGET_DIR, "WRF-HRRR Computed Dataset", "data", year, abbr, fname)
            if not os.path.exists(local_path):
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                try:
                    tmp = hf_hub_download(
                        repo_id="CropNet/CropNet", repo_type="dataset",
                        filename=hf_path, local_dir="/tmp/hf_weather"
                    )
                    shutil.move(tmp, local_path)
                    print(f"Downloaded pre-computed weather: {fname}")
                except Exception as e:
                    # Some files might genuinely be missing on the repo
                    pass

# --- Sentinel-2 imagery (from HuggingFace) -----------------------------------
QUARTERS  = [("01-01", "03-31"), ("04-01", "06-30"),
             ("07-01", "09-30"), ("10-01", "12-31")]

total = len(state_codes) * len(YEARS) * len(QUARTERS)
done = 0

for sc in state_codes:
    abbr = STATE_MAP[sc]
    for year in YEARS:
        for qs, qe in QUARTERS:
            fname      = f"Agriculture_{sc}_{abbr}_{year}-{qs}_{year}-{qe}.h5"
            hf_path    = f"Sentinel-2 Imagery/data/AG/{year}/{abbr}/{fname}"
            local_path = os.path.join(TARGET_DIR, "Sentinel", "data", "AG", year, abbr, fname)
            done += 1
            if os.path.exists(local_path):
                print(f"[{done}/{total}] already exists: {fname}")
                continue
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            try:
                tmp = hf_hub_download(
                    repo_id="CropNet/CropNet", repo_type="dataset",
                    filename=hf_path, local_dir="/tmp/hf_s2"
                )
                shutil.move(tmp, local_path)
                print(f"[{done}/{total}] downloaded: {fname}")
            except Exception as e:
                print(f"[{done}/{total}] FAILED {fname}: {e}")

print("Download complete:", TARGET_DIR)
