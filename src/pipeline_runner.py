"""Utilities for running extracted notebook cell modules in order."""

from pathlib import Path


def run_cells(cell_numbers, context=None):
    """Execute selected cell modules inside one shared global context."""
    root = Path(__file__).resolve().parent / "cells"
    if context is None:
        context = {"__name__": "__main__"}

    for n in cell_numbers:
        path = root / f"cell_{n:02d}.py"
        code = path.read_text()
        compiled = compile(code, str(path), "exec")
        exec(compiled, context, context)

    return context
