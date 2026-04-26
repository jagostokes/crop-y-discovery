"""Auto-extracted from notebooks/Crop_y_Discovery.ipynb (code cell 1).
This file mirrors notebook logic for script-style orchestration.
"""

# Install dependencies. libeccodes is required by pygrib (HRRR weather files
# are in GRIB format). cropnet is the official package wrapping the dataset.
# NOTE: notebook magic/shell command (run in notebook): !apt-get install -q -y libeccodes-dev
# NOTE: notebook magic/shell command (run in notebook): %pip install -q pygrib cropnet

import torch
print("torch:", torch.__version__, "| cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
# NOTE: notebook magic/shell command (run in notebook):     !nvidia-smi -L
