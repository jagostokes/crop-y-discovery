"""Run model definition and training stages from the notebook."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline_runner import run_cells


if __name__ == "__main__":
    # Includes setup/data cells so this script is self-contained.
    run_cells(range(1, 13))
