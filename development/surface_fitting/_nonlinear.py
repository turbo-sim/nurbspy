"""Terrain-study settings and compatibility adapters for the public fitting API.

The fitting loss, sparse derivatives and Pounce driver live in nurbspy.fitting.
This module retains only the study's normalized bounds, penalties and archive
schema, plus small constructor aliases used by its numerical regression tests.
"""

import numpy as np
import nurbspy as nrb
from nurbspy.fitting._common import FitLayout
from nurbspy.fitting._nonlinear import SurfaceFitProblem
from _experiment import observation_parameters

OBJECTIVE_SCALE = 1000.0
TOLERANCE = 1e-8
FRAME_ANCHOR = 1e-4
WEIGHT_PENALTY = 1e-6


def geometric_options(initial):
    """Explicit modelling choices for this unit-domain analytic terrain."""
    lower = np.full_like(initial.P, -0.25)
    upper = np.full_like(initial.P, 1.25)
    lower[2], upper[2] = -0.6, 0.6
    penalty = np.zeros_like(initial.P)
    penalty[:2] = FRAME_ANCHOR / 2
    return dict(
        control_bounds=(lower, upper),
        control_regularization=penalty,
        weight_regularization=WEIGHT_PENALTY,
    )


def study_layout(initial, *, geometric=False):
    """Build the same variable ordering used by the archived terrain study."""
    options = geometric_options(initial) if geometric else {}
    return FitLayout(
        initial,
        optimize_weights=True,
        weight_bounds=(0.1, 10.0),
        control_bounds=options.get("control_bounds"),
    )


def hessian_pattern(shape, degrees):
    """Adapt the shared sparse pattern to the study's 3D, W00=1 convention."""
    from nurbspy.fitting._nonlinear import hessian_pattern as pattern

    initial = nrb.NurbsSurface(
        np.zeros((3, *shape)), u_degree=degrees[0], v_degree=degrees[1]
    )
    return pattern(study_layout(initial))


class RationalSurfaceProblem(SurfaceFitProblem):
    """Compatibility constructor around the shared fixed-parameter problem."""

    def __init__(self, initial, observations, *, geometric=False):
        options = geometric_options(initial) if geometric else {}
        super().__init__(
            study_layout(initial, geometric=geometric),
            observations["points"],
            observation_parameters(observations),
            objective_scale=OBJECTIVE_SCALE,
            control_regularization=options.get("control_regularization", 0.0),
            weight_regularization=options.get("weight_regularization", 0.0),
        )


def _plain(value):
    """Convert numerical metadata to JSON-compatible Python values."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def legacy_result(result, initial, *, tolerance, geometric=False, snapshot_stride=1):
    """Keep the existing reports/archives while consuming a public FitResult."""
    info = dict(result.diagnostics)
    for key in ("bound_multipliers", "x", "surface", "history"):
        info.pop(key, None)
    info.update(result.timings)
    info.update(
        success=bool(result.success),
        message=result.message,
        iterations=int(result.iterations),
        tolerance=tolerance,
        objective_scale=OBJECTIVE_SCALE,
        history_kind="native_observations",
        fitting_api="nurbspy.fit_surface",
    )
    info.setdefault("variables", 4 * initial.W.size - 1)
    info.setdefault("native_history", [])
    layout = study_layout(initial, geometric=geometric)
    final = np.r_[
        np.asarray(result.surface.P)
        .reshape(layout.ndim, -1)
        .T.ravel()[layout.control_indices],
        np.log(np.asarray(result.surface.W).ravel()[layout.weight_indices]),
    ]
    info.setdefault(
        "bound_violation",
        float(max(0.0, np.max(layout.lower - final), np.max(final - layout.upper))),
    )
    history = []
    for saved in result.history:
        if (
            snapshot_stride > 1
            and saved["event_kind"] == "native_observation"
            and saved["iteration"] % snapshot_stride
        ):
            continue
        frame = dict(saved)
        frame.setdefault("training_rms", frame.get("rms"))
        history.append(frame)
    return result.surface, _plain(info), history


def nonlinear(initial, observations, *, max_iterations=1000, tolerance=TOLERANCE):
    """Run known-label rational fitting through the installed public API."""
    result = nrb.fit_surface(
        observations["points"],
        initial,
        parameters=observation_parameters(observations),
        optimize_weights=True,
        weight_bounds=(0.1, 10.0),
        objective_scale=OBJECTIVE_SCALE,
        tolerance=tolerance,
        max_iterations=max_iterations,
        record_history=True,
        print_convergence=True,
    )
    return legacy_result(result, initial, tolerance=tolerance)
