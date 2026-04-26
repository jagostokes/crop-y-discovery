"""Run data setup and dataset-building stages from the notebook."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline_runner import run_cells


if __name__ == "__main__":
    # Cells 1-8 cover setup, dataset construction, split, and baseline prep.
    run_cells(range(1, 9))
