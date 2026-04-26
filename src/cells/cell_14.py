"""Auto-extracted from notebooks/Crop_y_Discovery.ipynb (code cell 14).
This file mirrors notebook logic for script-style orchestration.
"""

from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np
from cropnet.data_retriever import DataRetriever

# ---------------------------------------------------------------------------
# Evaluate "County Historical Mean" Baseline
# ---------------------------------------------------------------------------
# Instead of a global mean, what if we just predict each test county's
# 10-year historical average for the 2021 and 2022 seasons?
# ---------------------------------------------------------------------------

retriever = DataRetriever(base_dir=TARGET_DIR)
test_df = retriever.retrieve_USDA(CROP, FIPS_TEST, YEARS)

historical_preds = []
actuals = []

for _, row in test_df.iterrows():
    fips = str(row["state_ansi"]).zfill(2) + str(row["county_ansi"]).zfill(3)
    actual_yield = float(row["YIELD, MEASURED IN BU / ACRE"])

    # Predict the historical average for this specific county
    # (test_ds.hist_yields contains the historical means for the test counties)
    if fips in test_ds.hist_yields:
        pred_yield = test_ds.hist_yields[fips]
    else:
        pred_yield = baseline_pred  # fallback to global mean if no history

    historical_preds.append(pred_yield)
    actuals.append(actual_yield)

historical_preds = np.array(historical_preds)
actuals = np.array(actuals)

hist_rmse = np.sqrt(np.mean((historical_preds - actuals) ** 2))
hist_mae  = mean_absolute_error(actuals, historical_preds)
hist_r2   = r2_score(actuals, historical_preds)

print("── County Historical Mean Baseline ──")
print(f"  RMSE : {hist_rmse:.2f} BU/acre")
print(f"  MAE  : {hist_mae:.2f} BU/acre")
print(f"  R²   : {hist_r2:.4f}")
print("─" * 37)
print(f"  Model RMSE : {test_rmse:.2f} BU/acre")
print()

if test_rmse < hist_rmse:
    improvement = hist_rmse - test_rmse
    print(f"✅ The multimodal model beats the historical county average by {improvement:.2f} BU/acre!")
else:
    worse_by = test_rmse - hist_rmse
    print(f"❌ The model does NOT beat the historical county average (worse by {worse_by:.2f} BU/acre).")
