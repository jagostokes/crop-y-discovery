"""Auto-extracted from notebooks/Crop_y_Discovery.ipynb (code cell 15).
This file mirrors notebook logic for script-style orchestration.
"""

# ---------------------------------------------------------------------------
# Diagnostic plots.
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

# (1) Learning curves — train vs val RMSE per epoch.
# A growing gap between the two curves signals overfitting.
ax = axes[0]
epochs = range(1, len(history["train_rmse"]) + 1)
ax.plot(epochs, history["train_rmse"], label="train", marker="o", markersize=3)
ax.plot(epochs, history["val_rmse"],   label="val",   marker="s", markersize=3)
ax.set_xlabel("Epoch")
ax.set_ylabel("RMSE (BU/acre)")
ax.set_title("Learning Curves")
ax.legend()
ax.grid(alpha=0.3)

# (2) Predicted vs actual scatter on test set.
# Points along the diagonal = perfect predictions.
# A flat horizontal cloud = model collapsed to predicting the mean.
ax = axes[1]
ax.scatter(y_true, y_pred, alpha=0.4, edgecolors="k", linewidths=0.3)
lims = [min(y_true.min(), y_pred.min()) - 5,
        max(y_true.max(), y_pred.max()) + 5]
ax.plot(lims, lims, "r--", linewidth=1, label="perfect")
# Also draw the baseline (constant predictor) as a horizontal line.
ax.axhline(baseline_pred, color="gray", linestyle=":", linewidth=1,
            label=f"baseline ({baseline_pred:.0f})")
ax.set_xlabel("Actual yield (BU/acre)")
ax.set_ylabel("Predicted yield (BU/acre)")
ax.set_title(f"Predicted vs Actual  (R²={test_r2:.3f})")
ax.legend()
ax.grid(alpha=0.3)

# (3) Residual histogram — errors should be roughly Gaussian and centered at 0.
# Skewed residuals = systematic bias (over- or under-predicting).
ax = axes[2]
residuals = y_pred - y_true
ax.hist(residuals, bins=30, edgecolor="k", alpha=0.7)
ax.axvline(0, color="r", linestyle="--", linewidth=1)
ax.set_xlabel("Residual (Predicted - Actual)  [BU/acre]")
ax.set_ylabel("Count")
ax.set_title(f"Residuals  (mean={residuals.mean():.2f}, std={residuals.std():.2f})")
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()
