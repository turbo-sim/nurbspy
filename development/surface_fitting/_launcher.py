"""Small command-line and thread setup shared by the five development demos."""

import argparse
import os
from pathlib import Path


def run_demo(method, **settings):
    """Set unset native thread budgets before importing numerical libraries."""
    for name in (
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "RAYON_NUM_THREADS",
    ):
        os.environ.setdefault(name, "4")
    from _study import run

    return run(method=method, **settings)


def cli(main, description, *, alternating=False):
    """Forward only explicit CLI overrides, preserving each demo's Python defaults."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--grids", dest="grid_sizes", nargs="+", type=int)
    parser.add_argument("--samples", type=int)
    parser.add_argument("--sigma", type=float)
    parser.add_argument("--max-iterations", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--show", action="store_true", default=None)
    if alternating:
        parser.add_argument("--max-cycles", type=int)
    return main(
        **{
            key: value
            for key, value in vars(parser.parse_args()).items()
            if value is not None
        }
    )
