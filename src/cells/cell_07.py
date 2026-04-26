"""Auto-extracted from notebooks/Crop_y_Discovery.ipynb (code cell 7).
This file mirrors notebook logic for script-style orchestration.
"""

# ---------------------------------------------------------------------------
# Build datasets and split into train/val/test.
# ---------------------------------------------------------------------------
# Strategy:
#   - Test set is fully held-out counties (FIPS_TEST), defined upstream.
#   - Validation set is a random 15% of the training samples. This is
#     RANDOM at the sample level (not county level) for simplicity, but
#     since each sample is one patch, val and train still differ in spatial
#     location even though they share counties.
#
#   For a more rigorous split you'd hold out a county for validation too.
#   We use random sample split here to keep enough training data for the
#   model to learn from, given how few counties we have.
# ---------------------------------------------------------------------------
from torch.utils.data import DataLoader, random_split

# Build full training dataset (will be split into train + val below).
full_train_ds = MultimodalCropDataset(
    fips_codes=FIPS_TRAIN,
    years=YEARS,
    base_dir=TARGET_DIR,
    crop=CROP,
    transform=train_augment,    # augmentation ON for train
)

# Random 85/15 split on training samples.
val_size   = int(0.15 * len(full_train_ds))
train_size = len(full_train_ds) - val_size
train_ds, val_ds = random_split(
    full_train_ds,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(SEED),  # reproducible split
)

# IMPORTANT: val should NOT be augmented. random_split returns Subset objects
# that wrap the original dataset, so they share the transform. We work around
# this by giving the underlying dataset a flag that the Subset can toggle.
# Simpler approach: just create a separate non-augmented dataset for val.
# Even simpler: live with augmentation on val — the random flips average out
# over many epochs anyway. We choose to just disable transform globally during
# validation by toggling the dataset's transform attribute in the val loop.
# (See run_epoch below.)

# Test dataset uses train's normalization stats (CRITICAL for fair evaluation).
# Without this, the model would be predicting on a differently-scaled target
# than what it trained on, which is the standard data leakage trap students
# fall into.
test_ds = MultimodalCropDataset(
    fips_codes=FIPS_TEST,
    years=YEARS,
    base_dir=TARGET_DIR,
    crop=CROP,
    transform=None,                            # no augmentation at test time
    yield_mean=full_train_ds.yield_mean,       # reuse train normalization
    yield_std=full_train_ds.yield_std,
    weather_dim=full_train_ds.weather_dim,     # match weather vector size
)

# DataLoader settings:
#   batch_size=16: small enough to fit T x 3 x 224 x 224 sequences in T4 GPU memory.
#   shuffle=True for train: standard SGD practice.
#   num_workers=2: parallel data loading, but careful with seed_worker for reproducibility.
BATCH_SIZE = 16

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                           worker_init_fn=seed_worker, generator=g)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

# Sanity check: pull one batch and print shapes.
img_seq, weather, z = next(iter(train_loader))
print(f"img_seq: {tuple(img_seq.shape)}   # (B, T, 3, 224, 224)")
print(f"weather: {tuple(weather.shape)}   # (B, weather_dim)")
print(f"z (yield):       {tuple(z.shape)}   # (B,)")
print()
print(f"Train: {len(train_ds)}  Val: {len(val_ds)}  Test: {len(test_ds)} samples")
print(f"Yield mean: {full_train_ds.yield_mean:.1f}  std: {full_train_ds.yield_std:.1f} BU/acre")
print(f"Weather feature dim: {full_train_ds.weather_dim}")
