"""Fit an unlabeled cloud by alternating point projection and surface optimization.

The surface fit holds u/v fixed and uses Pounce interior point with an exact
sparse Hessian. Projection holds the surface fixed and reuses NurbsPy's existing
batched JAX bounded-Newton API, including its exact 2x2 Hessians. The objective,
geometric bounds and regularization match case 03. No projection algorithm is
reimplemented in the demo; compiled library calls are reused between cycles.
Overall convergence requires both blocks to be stationary at the same state.
"""

from pathlib import Path
from _launcher import run_demo, cli

GRID_SIZES = (4, 8, 16, 32)
SAMPLES_PER_AXIS = 45
NOISE_SIGMA = 0.0
MAX_ITERATIONS = 1000
TOLERANCE = 1e-8
SHOW_REPORT = False
MAX_CYCLES = 100
OUTPUT = Path(__file__).resolve().parent / "output" / "04_alternating_projection"


def main(
    *,
    show=SHOW_REPORT,
    grid_sizes=GRID_SIZES,
    samples=SAMPLES_PER_AXIS,
    sigma=NOISE_SIGMA,
    max_iterations=MAX_ITERATIONS,
    tolerance=TOLERANCE,
    output=OUTPUT,
    max_cycles=MAX_CYCLES
):
    """Run the configured refinements and save an offline interactive report."""
    return run_demo(
        "alternating",
        show=show,
        grid_sizes=grid_sizes,
        samples=samples,
        sigma=sigma,
        max_iterations=max_iterations,
        tolerance=tolerance,
        output=output,
        max_cycles=max_cycles,
    )


if __name__ == "__main__":
    cli(main, __doc__, alternating=True)
