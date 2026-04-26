"""Auto-extracted from notebooks/Crop_y_Discovery.ipynb (code cell 12).
This file mirrors notebook logic for script-style orchestration.
"""

# ---------------------------------------------------------------------------
# Training loop: validation, LR scheduling, early stopping, best-checkpoint.
# ---------------------------------------------------------------------------
import math
import copy

# Hyperparameters.
NUM_EPOCHS      = 50         # max epochs (early stopping will likely cut this short)
LEARNING_RATE   = 1e-3       # AdamW with weight decay 1e-4 is a good default
WEIGHT_DECAY    = 1e-4
EARLY_STOP_PAT  = 7          # stop if no val improvement for this many epochs
LR_PATIENCE     = 3          # halve LR if no val improvement for this many epochs

optimizer = torch.optim.AdamW(model.parameters(),
                               lr=LEARNING_RATE,
                               weight_decay=WEIGHT_DECAY)
criterion = torch.nn.MSELoss()

# ReduceLROnPlateau monitors val loss and reduces LR when it stops improving.
# factor=0.5 -> halve LR; patience=LR_PATIENCE epochs of no improvement before reducing.
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=LR_PATIENCE
)


def run_epoch(loader, *, train: bool):
    """
    Runs one pass through `loader`. Returns RMSE in original (BU/acre) units.

    Args:
        loader: DataLoader to iterate over.
        train:  If True, run backward pass and optimizer step. If False,
                wraps in no_grad for efficiency and skips weight updates.
    """
    # model.train(True) enables BatchNorm/Dropout's training behavior.
    # model.train(False) (i.e., .eval()) freezes BN running stats and
    # disables Dropout. CRITICAL to flip this correctly for val/test —
    # the #1 source of "train metrics good, val metrics terrible" bugs.
    model.train(train)

    total_sq_err = 0.0
    n_seen = 0

    # set_grad_enabled(False) during val saves memory and speeds eval.
    with torch.set_grad_enabled(train):
        for img_seq, weather, z in loader:
            img_seq = img_seq.to(device, non_blocking=True)
            weather = weather.to(device, non_blocking=True)
            z       = z.to(device, non_blocking=True)

            z_hat = model(img_seq, weather)
            loss  = criterion(z_hat, z)

            if train:
                optimizer.zero_grad()
                loss.backward()
                # Gradient clipping prevents the rare exploding gradient that
                # can destabilize training, especially with batch norm + small batches.
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            # MSE loss returns mean-of-squared-errors, so multiply by len(z)
            # to recover sum-of-squared-errors for proper aggregation.
            total_sq_err += loss.item() * len(z)
            n_seen += len(z)

    # MSE -> RMSE (in z-score space) -> denormalize to BU/acre.
    rmse_z = math.sqrt(total_sq_err / n_seen)
    rmse_bu = rmse_z * full_train_ds.yield_std
    return rmse_bu


# Disable augmentation during validation by swapping the underlying transform.
# random_split returns Subsets that share the parent dataset, so changing the
# parent's transform affects both train_ds and val_ds. We toggle it at runtime.
def set_augment(on: bool):
    full_train_ds.transform = train_augment if on else None


# ---- Training loop ----
history = {"train_rmse": [], "val_rmse": [], "lr": []}
best_val_rmse = float("inf")
best_state    = None
epochs_no_improve = 0

print(f"Training on {device} for up to {NUM_EPOCHS} epochs "
       f"(early stop patience = {EARLY_STOP_PAT}, LR patience = {LR_PATIENCE})")
print("-" * 72)

for epoch in range(1, NUM_EPOCHS + 1):
    # Train pass — augmentation ON.
    set_augment(True)
    train_rmse = run_epoch(train_loader, train=True)

    # Val pass — augmentation OFF (we want a deterministic eval).
    set_augment(False)
    val_rmse = run_epoch(val_loader, train=False)
    set_augment(True)  # restore for next epoch

    # Update LR scheduler based on val loss.
    scheduler.step(val_rmse)
    current_lr = optimizer.param_groups[0]["lr"]

    history["train_rmse"].append(train_rmse)
    history["val_rmse"].append(val_rmse)
    history["lr"].append(current_lr)

    # Track best model. deepcopy is needed because state_dict() returns
    # references — without copy, our "best" state would mutate as training continues.
    if val_rmse < best_val_rmse:
        best_val_rmse = val_rmse
        best_state = copy.deepcopy(model.state_dict())
        epochs_no_improve = 0
        marker = " ← best"
    else:
        epochs_no_improve += 1
        marker = ""

    print(f"Epoch {epoch:2d}/{NUM_EPOCHS}  "
           f"train RMSE={train_rmse:6.2f}  val RMSE={val_rmse:6.2f}  "
           f"lr={current_lr:.1e}{marker}")

    # Early stop.
    if epochs_no_improve >= EARLY_STOP_PAT:
        print(f"\nEarly stopping triggered (no val improvement for "
               f"{EARLY_STOP_PAT} epochs).")
        break

# Restore the best weights (not whatever we ended on).
if best_state is not None:
    model.load_state_dict(best_state)
    print(f"\nRestored best model: val RMSE = {best_val_rmse:.2f} BU/acre")
