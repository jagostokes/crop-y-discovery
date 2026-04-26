"""Auto-extracted from notebooks/Crop_y_Discovery.ipynb (code cell 8).
This file mirrors notebook logic for script-style orchestration.
"""

# ---------------------------------------------------------------------------
# Naive baseline: predict the global training-mean yield for every test sample.
# ---------------------------------------------------------------------------
# Why this baseline matters:
#   - It establishes the FLOOR. Any model worth keeping must do better than this.
#   - It exposes whether your task has any signal at all in the features.
#     If your CNN's RMSE matches the mean baseline, your features are useless.
#   - It gives a "skill score" reference: how much variance does the model
#     explain ABOVE the trivial predictor?
#
# We could also try a county-level mean (predict each county's own historical
# average), but our train/test split is by county, so test counties have NO
# history in train — the global mean is the right baseline here.
# ---------------------------------------------------------------------------
from sklearn.metrics import mean_absolute_error, r2_score

# Pull all training yields in original (BU/acre) units.
# Note: the random_split Subset stores indices into full_train_ds.samples.
train_yields_normalized = np.array([
    full_train_ds.samples[i][2].item() for i in train_ds.indices
])
# Denormalize back to BU/acre.
train_yields_bu = (train_yields_normalized
                    * full_train_ds.yield_std + full_train_ds.yield_mean)

baseline_pred = float(train_yields_bu.mean())

# Test yields in original units.
test_yields_normalized = np.array([s[2].item() for s in test_ds.samples])
test_yields_bu = (test_yields_normalized
                   * test_ds.yield_std + test_ds.yield_mean)

# Predict baseline_pred for every test sample.
y_pred_baseline = np.full_like(test_yields_bu, baseline_pred)

baseline_rmse = float(np.sqrt(np.mean((y_pred_baseline - test_yields_bu) ** 2)))
baseline_mae  = float(mean_absolute_error(test_yields_bu, y_pred_baseline))
# R^2 of a constant predictor is, by definition, <= 0. We compute it anyway
# because seeing R^2 = 0 (or negative) makes the comparison concrete.
baseline_r2   = float(r2_score(test_yields_bu, y_pred_baseline))

print("── Baseline (predict training mean) ──")
print(f"  Predicted yield (constant): {baseline_pred:.2f} BU/acre")
print(f"  RMSE : {baseline_rmse:.2f} BU/acre")
print(f"  MAE  : {baseline_mae:.2f} BU/acre")
print(f"  R²   : {baseline_r2:.4f}")
print()
print("Goal: the multimodal model should beat these numbers significantly.")
