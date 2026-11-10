"""Fit an unlabeled cloud by jointly optimizing control coordinates, weights and u/v.

Pounce interior point uses the full exact sparse Hessian, including mixed surface/
parameter derivatives. Local parameter windows keep each derivative graph sparse;
windows are recentered when needed, with native multiplier checks against the
global [0,1]^2 domain. Every solve updates ALL unknowns jointly. No projection/refit
alternation is performed. True generating u/v labels are not supplied to the fit."""

from pathlib import Path
from _launcher import run_demo, cli

GRID_SIZES = (4, 8, 16, 32)
SAMPLES_PER_AXIS = 45
NOISE_SIGMA = 0.0
MAX_ITERATIONS = 1000
TOLERANCE = 1e-8
SHOW_REPORT = False
OUTPUT = Path(__file__).resolve().parent / "output" / "03_joint_parameters"


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
        "joint",
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
