"""Joint-fitting terrain study, using the reusable NurbsPy implementation."""

import numpy as np
import nurbspy as nrb
from nurbspy.fitting._joint import JointSurfaceFitProblem, parameter_windows
from _nonlinear import OBJECTIVE_SCALE, geometric_options, study_layout, legacy_result


class JointSurfaceProblem(JointSurfaceFitProblem):
    """Compatibility constructor; all joint derivative logic is in src."""

    def __init__(self, initial, observations, *, start=None):
        guesses = np.clip(observations["points"][:, :2], 1e-10, 1 - 1e-10)
        options = geometric_options(initial)
        super().__init__(
            study_layout(initial, geometric=True),
            observations["points"],
            guesses,
            objective_scale=OBJECTIVE_SCALE,
            control_regularization=options["control_regularization"],
            weight_regularization=options["weight_regularization"],
            start=start,
        )


def joint(initial, observations, *, max_iterations=1000, tolerance=1e-8, max_passes=8):
    """Fit jointly; x/y-based initial estimates are specific to this terrain."""
    guesses = np.clip(observations["points"][:, :2], 1e-10, 1 - 1e-10)
    result = nrb.fit_surface(
        observations["points"],
        initial,
        correspondence="joint",
        initial_parameters=guesses,
        optimize_weights=True,
        weight_bounds=(0.1, 10.0),
        objective_scale=OBJECTIVE_SCALE,
        tolerance=tolerance,
        max_iterations=max_iterations,
        max_passes=max_passes,
        record_history=True,
        print_convergence=True,
        **geometric_options(initial)
    )
    fitted, info, history = legacy_result(
        result, initial, tolerance=tolerance, geometric=True, snapshot_stride=5
    )
    info["variables"] = 4 * initial.W.size - 1 + 2 * len(observations["points"])
    return fitted, info, history, result.parameters
