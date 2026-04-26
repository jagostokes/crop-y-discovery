"""Auto-extracted from notebooks/Crop_y_Discovery.ipynb (code cell 2).
This file mirrors notebook logic for script-style orchestration.
"""

# ---------------------------------------------------------------------------
# Reproducibility: seed everything that has a random component.
# ---------------------------------------------------------------------------
# ML experiments are pseudo-random — they use random number generators for
# weight initialization, dataset shuffling, dropout, etc. Without fixed seeds,
# you get a different result every run, which makes debugging miserable
# ("did my change help, or did I just get lucky this time?").
#
# We seed all four sources of randomness our code touches:
#   - Python's built-in `random`      (used by some libraries internally)
#   - NumPy                            (used by our dataset code)
#   - PyTorch (CPU)                    (weight init, DataLoader shuffling)
#   - PyTorch (CUDA)                   (GPU operations)
#
# Note: even with seeds, some CUDA ops are non-deterministic for speed reasons.
# Setting `torch.backends.cudnn.deterministic = True` fixes this at a small
# performance cost. We do this because reproducibility matters more than
# squeezing out the last 5% of training speed for a class project.
# ---------------------------------------------------------------------------
import random
import numpy as np
import torch

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Also seed PyTorch DataLoader workers so shuffle order is reproducible.
def seed_worker(worker_id):
    """Called by DataLoader to seed each worker process."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(SEED)

print(f"All random seeds set to {SEED}")
