"""Fit the analytic terrain at known u/v labels with unit-weight cubic B-splines.

Control coordinates are found by direct tensor-product linear least squares.
There is no point projection or iterative nonlinear optimization."""

from pathlib import Path
from _launcher import run_demo, cli

GRID_SIZES = (4, 8, 16, 32)
SAMPLES_PER_AXIS = 45
NOISE_SIGMA = 0.0
MAX_ITERATIONS = 1000
TOLERANCE = 1e-8
SHOW_REPORT = False
OUTPUT = Path(__file__).resolve().parent / "output" / "01_linear"


def main(
    *,
    show=SHOW_REPORT,
    grid_sizes=GRID_SIZES,
    samples=SAMPLES_PER_AXIS,
    sigma=NOISE_SIGMA,
    max_iterations=MAX_ITERATIONS,
    tolerance=TOLERANCE,
    output=OUTPUT
):
    """Run the configured refinements and save an offline interactive report."""
    return run_demo(
        "linear",
        show=show,
        grid_sizes=grid_sizes,
        samples=samples,
        sigma=sigma,
        max_iterations=max_iterations,
        tolerance=tolerance,
        output=output,
    )


if __name__ == "__main__":
    cli(main, __doc__, alternating=False)
