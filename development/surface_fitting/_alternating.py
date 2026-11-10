"""Alternating terrain study, using NurbsPy's shared fit/projection orchestration."""

import numpy as np
import nurbspy as nrb
from _nonlinear import OBJECTIVE_SCALE, geometric_options, legacy_result


def alternating(
    initial, observations, *, max_iterations=1000, tolerance=1e-8, max_cycles=100
):
    """Keep terrain choices here; fitting/projection algorithms live in src."""
    guesses = np.clip(observations["points"][:, :2], 0, 1)
    result = nrb.fit_surface(
        observations["points"],
        initial,
        correspondence="alternating",
        initial_parameters=guesses,
        optimize_weights=True,
        weight_bounds=(0.1, 10.0),
        objective_scale=OBJECTIVE_SCALE,
        tolerance=tolerance,
        max_iterations=max_iterations,
        max_cycles=max_cycles,
        record_history=True,
        print_convergence=True,
        **geometric_options(initial)
    )
    fitted, info, history = legacy_result(
        result, initial, tolerance=tolerance, geometric=True, snapshot_stride=5
    )
    info["variables"] = 4 * initial.W.size - 1 + 2 * len(observations["points"])
    return fitted, info, history, result.parameters
