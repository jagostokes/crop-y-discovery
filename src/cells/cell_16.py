"""Auto-extracted from notebooks/Crop_y_Discovery.ipynb (code cell 16).
This file mirrors notebook logic for script-style orchestration.
"""

# ---------------------------------------------------------------------------
# Persist the trained model to Google Drive (not /content, which is ephemeral).
# ---------------------------------------------------------------------------
# What we save:
#   - model_state:      the actual weights (load with model.load_state_dict)
#   - yield_mean/std:   needed to denormalize predictions at inference time
#   - weather_dim:      needed to instantiate the right-sized model on load
#   - history:          training curves for later analysis
#   - test_metrics:     so we have the eval numbers without re-running
#   - hyperparameters:  model configuration so we remember what settings we used
#
# We DON'T save the optimizer state — fine for inference. Save it too if you
# plan to resume training later.
# ---------------------------------------------------------------------------
import os

# Mount Drive (uncomment when running in Colab).
# from google.colab import drive
# drive.mount("/content/drive")
# CKPT_DIR = "/content/drive/MyDrive/CropNet"

# Fallback for non-Colab environments.
CKPT_DIR = "/content/drive/MyDrive/CropNet" if os.path.exists("/content/drive") \
           else "/content"
os.makedirs(CKPT_DIR, exist_ok=True)

ckpt_path = os.path.join(CKPT_DIR, f"multimodal_yield_{CROP.lower()}.pt")
torch.save({
    "model_state":  model.state_dict(),
    "crop":         CROP,
    "yield_mean":   full_train_ds.yield_mean,
    "yield_std":    full_train_ds.yield_std,
    "weather_dim":  full_train_ds.weather_dim,
    "history":      history,
    "test_metrics": {
        "rmse":  test_rmse,
        "mae":   test_mae,
        "r2":    test_r2,
        "mape":  test_mape,
        "baseline_rmse": baseline_rmse,
        "baseline_mae": baseline_mae,
        "baseline_r2": baseline_r2,
    },
    "hyperparameters": {
        "NUM_EPOCHS": NUM_EPOCHS,
        "LEARNING_RATE": LEARNING_RATE,
        "WEIGHT_DECAY": WEIGHT_DECAY,
        "EARLY_STOP_PAT": EARLY_STOP_PAT,
        "LR_PATIENCE": LR_PATIENCE,
        "BATCH_SIZE": BATCH_SIZE
    },
    "fips_train": FIPS_TRAIN,
    "fips_test":  FIPS_TEST,
    "years":      YEARS,
}, ckpt_path)

print(f"Saved checkpoint to: {ckpt_path}")
print(f"Test RMSE: {test_rmse:.2f} BU/acre  |  R²: {test_r2:.4f}")
