"""Auto-extracted from notebooks/Crop_y_Discovery.ipynb (code cell 13).
This file mirrors notebook logic for script-style orchestration.
"""

# ---------------------------------------------------------------------------
# Evaluate the best model on the held-out test set.
# ---------------------------------------------------------------------------
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error

model.eval()
all_preds, all_targets = [], []

with torch.no_grad():
    for img_seq, weather, z in test_loader:
        img_seq = img_seq.to(device)
        weather = weather.to(device)
        preds = model(img_seq, weather).cpu().numpy()
        all_preds.append(preds)
        all_targets.append(z.numpy())

# Denormalize predictions and targets back to BU/acre.
mu, sigma = full_train_ds.yield_mean, full_train_ds.yield_std
y_pred = np.concatenate(all_preds)   * sigma + mu
y_true = np.concatenate(all_targets) * sigma + mu

# ---- Metrics ----
test_rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
test_mae  = float(mean_absolute_error(y_true, y_pred))
test_r2   = float(r2_score(y_true, y_pred))
# Guard MAPE against division by zero — happens when y_true contains 0.
nonzero_mask = y_true != 0
test_mape = float(
    np.mean(np.abs((y_true[nonzero_mask] - y_pred[nonzero_mask])
                    / y_true[nonzero_mask])) * 100
) if nonzero_mask.any() else float("nan")

# Skill score: improvement over baseline (% reduction in RMSE).
skill_pct = (1 - test_rmse / baseline_rmse) * 100

print("── Test Set Metrics ─────────────────────────")
print(f"  RMSE : {test_rmse:.2f} BU/acre   (baseline: {baseline_rmse:.2f})")
print(f"  MAE  : {test_mae:.2f} BU/acre   (baseline: {baseline_mae:.2f})")
print(f"  R²   : {test_r2:.4f}             (baseline: {baseline_r2:.4f})")
print(f"  MAPE : {test_mape:.2f}%")
print()
print(f"  Skill score: {skill_pct:+.1f}% RMSE reduction vs. baseline")
print("─" * 45)
