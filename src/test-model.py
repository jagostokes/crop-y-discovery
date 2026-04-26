"""Run through training and evaluate on the test split."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline_runner import run_cells


if __name__ == "__main__":
    # Includes evaluation and diagnostics cells.
    run_cells(range(1, 16))
