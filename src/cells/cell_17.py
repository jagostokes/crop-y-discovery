"""Auto-extracted from notebooks/Crop_y_Discovery.ipynb (code cell 17).
This file mirrors notebook logic for script-style orchestration.
"""

import os
from datetime import datetime
import pandas as pd
import torch

def log_experiment_to_md(checkpoint_path, log_file="training_log.md"):
    """
    Reads a saved PyTorch checkpoint and appends its metadata and
    training history to a Markdown file.
    """
    ckpt = torch.load(checkpoint_path, map_location=device)

    # Prepare history dataframe
    history_df = pd.DataFrame(ckpt.get("history", {}))
    history_df.index.name = "Epoch"
    history_df.index += 1

    metrics = ckpt.get("test_metrics", {})
    hparams = ckpt.get("hyperparameters", {})
    crop = ckpt.get("crop", "Unknown")
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Open the file in append mode ('a')
    with open(log_file, "a") as f:
        f.write(f"## Experiment: {crop} Yield Prediction - {now}\n\n")

        f.write("### Configuration\n")
        f.write(f"- **Crop**: {crop}\n")
        f.write(f"- **Train Counties FIPS**: {ckpt.get('fips_train', 'N/A')}\n")
        f.write(f"- **Test Counties FIPS**: {ckpt.get('fips_test', 'N/A')}\n")
        f.write(f"- **Years**: {ckpt.get('years', 'N/A')}\n\n")

        if hparams:
            f.write("### Hyperparameters\n")
            for k, v in hparams.items():
                f.write(f"- **{k}**: {v}\n")
            f.write("\n")

        f.write("### Test Results (Model vs Baseline)\n")
        for k, v in metrics.items():
            if isinstance(v, float):
                f.write(f"- **{k}**: {v:.4f}\n")
            else:
                f.write(f"- **{k}**: {v}\n")
        f.write("\n")

        f.write("### Epoch History\n")
        # to_markdown() creates a perfectly formatted Markdown table
        f.write(history_df.to_markdown())
        f.write("\n\n---\n\n")

    print(f"Successfully appended log to {log_file}")

# Let's log the checkpoint we saved earlier to a file in our Drive/working directory
log_path = os.path.join(CKPT_DIR, "training_log.md")
log_experiment_to_md(ckpt_path, log_path)

# Peek at the first few lines of the file to verify
with open(log_path, "r") as f:
    print("\n--- File Preview ---")
    print(f.read()[:800] + "\n... [File Continues]")
