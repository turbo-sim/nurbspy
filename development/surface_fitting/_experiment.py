"""Shared analytic terrain and scalable, known-parameter fitting for all five studies."""

import numpy as np
from scipy.interpolate import make_interp_spline
import nurbspy as nrb
from nurbspy.fitting._common import basis_matrix, make_surface

GRIDS = (4, 8, 16, 32)
SAMPLES = 45


def target(u, v):
    u, v = np.broadcast_arrays(u, v)
    z = np.zeros_like(u, dtype=float)
    for height, x, y, sx, sy in (
        (0.48, 0.23, 0.25, 0.12, 0.14),
        (0.42, 0.75, 0.67, 0.14, 0.12),
        (0.34, 0.25, 0.82, 0.12, 0.11),
        (-0.38, 0.55, 0.30, 0.13, 0.13),
        (-0.32, 0.50, 0.78, 0.11, 0.12),
        (-0.28, 0.85, 0.27, 0.10, 0.13),
    ):
        z += height * np.exp(-0.5 * (((u - x) / sx) ** 2 + ((v - y) / sy) ** 2))
    return np.stack((u, v, z), axis=-1)


def data(samples=SAMPLES, sigma=0.0, seed=2407):
    if samples < 4:
        raise ValueError("At least four observations per direction are required")
    axis = np.linspace(0.0, 1.0, samples)
    u, v = np.meshgrid(axis, axis, indexing="ij")
    clean = target(u, v).reshape(-1, 3)
    points = clean + np.random.default_rng(seed).normal(0.0, sigma, clean.shape)
    return dict(axis=axis, points=points, clean=clean, sigma=sigma, seed=seed)


def flat(count):
    if count < 4:
        raise ValueError("Cubic surfaces require at least four controls per direction")
    knots = np.r_[np.zeros(3), np.linspace(0.0, 1.0, count - 2), np.ones(3)]
    greville = np.array([knots[i + 1 : i + 4].mean() for i in range(count)])
    x, y = np.meshgrid(greville, greville, indexing="ij")
    return nrb.NurbsSurface(
        np.stack((x, y, 0 * x)), u_degree=3, v_degree=3, u_knots=knots, v_knots=knots
    )


def basis(surface, u, v=None):
    """Cartesian study samples using the installed paired-basis implementation."""
    u, v = np.meshgrid(u, u if v is None else v, indexing="ij")
    return basis_matrix(surface, np.column_stack((u.ravel(), v.ravel())))


def evaluate(surface, u, v=None):
    b = basis(surface, u, v)
    w = surface.W.ravel()
    return (b @ (surface.P.reshape(3, -1).T * w[:, None])) / (b @ w)[:, None]


def observation_parameters(observations):
    """Translate the study's Cartesian data format to public paired labels."""
    if "parameters" in observations:
        return np.asarray(observations["parameters"])
    u, v = np.meshgrid(observations["axis"], observations["axis"], indexing="ij")
    return np.column_stack((u.ravel(), v.ravel()))


def metrics(surface, observations):
    prediction = evaluate(surface, observations["axis"])
    axis = np.linspace(0.0, 1.0, 193)
    u, v = np.meshgrid(axis, axis, indexing="ij")
    error = evaluate(surface, axis) - target(u, v).reshape(-1, 3)
    return dict(
        training_rms=float(
            np.sqrt(np.mean(np.sum((prediction - observations["points"]) ** 2, axis=1)))
        ),
        clean_rms=float(np.sqrt(np.mean(np.sum(error**2, axis=1)))),
        clean_max=float(np.linalg.norm(error, axis=1).max()),
    )


def linear(initial, observations):
    """Run the shared linear fit; retain the study's surface/info return format."""
    result = nrb.fit_surface(
        observations["points"],
        initial,
        parameters=observation_parameters(observations),
        optimize_weights=False,
    )
    info = dict(result.diagnostics)
    info.pop("bound_multipliers", None)
    info.update(result.timings)
    info.update(
        success=bool(result.success),
        message=result.message,
        variables=initial.P.size,
        iterations=result.iterations,
        fitting_api="nurbspy.fit_surface",
    )
    return result.surface, info


def reference_display():
    """Smooth interpolation for display only; all errors use the analytic target."""
    axis = np.linspace(0.0, 1.0, 33)
    u, v = np.meshgrid(axis, axis, indexing="ij")
    spline = make_interp_spline(axis, target(u, v), axis=0)
    second = make_interp_spline(axis, spline.c, axis=1)
    return nrb.NurbsSurface(
        second.c.transpose(2, 1, 0),
        u_degree=3,
        v_degree=3,
        u_knots=spline.t,
        v_knots=second.t,
    )
