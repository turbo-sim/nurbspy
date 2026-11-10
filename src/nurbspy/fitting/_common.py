"""Backend-independent geometry, variable layout and sparse fitting bases."""

import numpy as np
from scipy.interpolate import BSpline
from scipy.sparse import csr_matrix


def _real_array(value, name):
    array = np.asarray(value)
    if array.dtype.kind not in "iuf" or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite real numbers")
    return array.astype(float, copy=True)


def surface_arrays(initial):
    """Validate one fixed-topology NumPy or JAX NURBS patch without importing JAX."""
    if not all(hasattr(initial, key) for key in ("P", "W", "p", "q", "U", "V")):
        raise TypeError("initial_surface must be a NurbsSurface")
    points = _real_array(initial.P, "Control points")
    weights = _real_array(initial.W, "Weights")
    if points.ndim != 3 or min(points.shape) < 1 or weights.shape != points.shape[1:]:
        raise ValueError(
            "Expected control points (dimension, Nu, Nv) and weights (Nu, Nv)"
        )
    if np.any(weights <= 0):
        raise ValueError("Surface weights must be strictly positive")
    for knots, degree, count in zip(
        (initial.U, initial.V), (initial.p, initial.q), weights.shape
    ):
        if (
            isinstance(degree, (bool, np.bool_))
            or int(degree) != degree
            or not 0 <= degree < count
        ):
            raise ValueError(
                "Surface degrees must be integers below their control counts"
            )
        degree = int(degree)
        knots = _real_array(knots, "Knots")
        if (
            knots.ndim != 1
            or len(knots) != count + degree + 1
            or np.any(np.diff(knots) < 0)
        ):
            raise ValueError("Invalid nondecreasing knot vector")
        if knots[degree] >= knots[-degree - 1]:
            raise ValueError("Surface knot domains must have positive width")
    return points, weights


def parameter_domain(initial):
    p, q = int(initial.p), int(initial.q)
    return (
        np.array([initial.U[p], initial.V[q]], dtype=float),
        np.array([initial.U[-p - 1], initial.V[-q - 1]], dtype=float),
    )


def validate_points(points, ndim):
    points = _real_array(points, "Target points")
    if points.ndim != 2 or points.shape[1] != ndim or not len(points):
        raise ValueError(f"Target points must have shape (N, {ndim}) with N > 0")
    return points


def validate_parameters(parameters, initial, count, name="parameters"):
    parameters = _real_array(parameters, name)
    if parameters.shape != (count, 2):
        raise ValueError(f"{name} must have shape (N, 2)")
    lower, upper = parameter_domain(initial)
    if np.any(parameters < lower) or np.any(parameters > upper):
        raise ValueError(f"{name} lie outside the surface knot domain")
    return parameters


def _mask(value, shape, name, default=True):
    value = np.asarray(default if value is None else value)
    if value.dtype.kind != "b":
        raise ValueError(f"{name} must contain booleans; True means free")
    try:
        return np.broadcast_to(value, shape).copy()
    except ValueError as error:
        raise ValueError(f"{name} must broadcast to {shape}") from error


def _bounds(bounds, shape, name, default):
    if bounds is None:
        bounds = default
    if not isinstance(bounds, (tuple, list)) or len(bounds) != 2:
        raise ValueError(f"{name} must be a (lower, upper) pair")
    try:
        lower, upper = [
            np.broadcast_to(np.asarray(value, dtype=float), shape).copy()
            for value in bounds
        ]
    except (ValueError, TypeError) as error:
        raise ValueError(f"{name} must broadcast to {shape}") from error
    if (
        np.isnan(lower).any()
        or np.isnan(upper).any()
        or np.any(lower > upper)
        or np.isposinf(lower).any()
        or np.isneginf(upper).any()
    ):
        raise ValueError(f"{name} require lower <= upper and no NaNs")
    return lower, upper


class FitLayout:
    """Eliminate fixed coordinates/weights and the common rational-weight gauge."""

    def __init__(
        self,
        initial,
        *,
        optimize_weights=False,
        control_mask=None,
        weight_mask=None,
        control_bounds=None,
        weight_bounds=None,
    ):
        self.initial = initial
        points, weights = surface_arrays(initial)
        self.ndim, *shape = points.shape
        self.shape, self.count = tuple(shape), weights.size
        cmask = _mask(control_mask, points.shape, "control_mask")
        if not isinstance(optimize_weights, (bool, np.bool_)):
            raise ValueError("optimize_weights must be a boolean")
        if weight_mask is not None and not optimize_weights:
            raise ValueError("weight_mask requires optimize_weights=True")
        wmask = _mask(
            weight_mask, weights.shape, "weight_mask", default=optimize_weights
        )
        self.weight_scale = 1.0
        if wmask.all():
            self.weight_scale = float(weights.flat[0])
            with np.errstate(over="ignore", under="ignore"):
                weights /= self.weight_scale
            if not np.isfinite(weights).all() or np.any(weights <= 0):
                raise ValueError(
                    "Weight normalization exceeds the floating-point range"
                )
            wmask.flat[0] = False
        clower, cupper = _bounds(
            control_bounds, points.shape, "control_bounds", (-np.inf, np.inf)
        )
        wlower, wupper = _bounds(
            weight_bounds, weights.shape, "weight_bounds", (0.0, np.inf)
        )
        if np.any(wlower < 0) or np.any(wupper <= 0):
            raise ValueError("weight_bounds must permit strictly positive weights")
        for values, free, lower, upper, name in (
            (points, cmask, clower, cupper, "control"),
            (weights, wmask, wlower, wupper, "weight"),
        ):
            if np.any((~free) & ((values < lower) | (values > upper))):
                raise ValueError(f"Fixed {name} values violate their bounds")
            values[free] = np.clip(values[free], lower[free], upper[free])
            free[lower == upper] = False
        if np.any(weights <= 0):
            raise ValueError(
                "Initial free weights must remain positive within weight_bounds"
            )
        self.controls0 = points.reshape(self.ndim, -1).T.copy()
        self.weights0 = weights.ravel().copy()
        self.control_indices = np.flatnonzero(cmask.reshape(self.ndim, -1).T.ravel())
        self.weight_indices = np.flatnonzero(wmask.ravel())
        self.ncontrol_variables = len(self.control_indices)
        self.nweight_variables = len(self.weight_indices)
        self.size = self.ncontrol_variables + self.nweight_variables
        self.x0 = np.r_[
            self.controls0.ravel()[self.control_indices],
            np.log(self.weights0[self.weight_indices]),
        ]
        with np.errstate(divide="ignore"):
            self.lower = np.r_[
                clower.reshape(self.ndim, -1).T.ravel()[self.control_indices],
                np.log(wlower.ravel()[self.weight_indices]),
            ]
            self.upper = np.r_[
                cupper.reshape(self.ndim, -1).T.ravel()[self.control_indices],
                np.log(wupper.ravel()[self.weight_indices]),
            ]

    def unpack(self, x):
        controls, weights = self.controls0.copy(), self.weights0.copy()
        controls.ravel()[self.control_indices] = np.asarray(x)[
            : self.ncontrol_variables
        ]
        weights[self.weight_indices] = np.exp(
            np.asarray(x)[self.ncontrol_variables : self.size]
        )
        return controls, weights

    def surface(self, x):
        return make_surface(self.initial, *self.unpack(x))


def make_surface(initial, controls, weights):
    """Return independent geometry in the input backend, with unchanged topology."""
    if any(
        base.__module__.startswith("nurbspy.jax.") for base in type(initial).__mro__
    ):
        from ..jax.nurbs_surface import NurbsSurface
    else:
        from ..numpy.nurbs_surface import NurbsSurface
    return NurbsSurface(
        np.asarray(controls).T.reshape(np.shape(initial.P)),
        weights=np.asarray(weights).reshape(np.shape(initial.W)),
        u_degree=int(initial.p),
        v_degree=int(initial.q),
        u_knots=np.asarray(initial.U).copy(),
        v_knots=np.asarray(initial.V).copy(),
    )


def basis_table(initial, parameters):
    """Local polynomial support at arbitrary paired samples, including zero entries."""
    count_u, count_v = np.shape(initial.W)
    p, q = int(initial.p), int(initial.q)
    a = BSpline.design_matrix(parameters[:, 0], np.asarray(initial.U), p).tocsr()
    b = BSpline.design_matrix(parameters[:, 1], np.asarray(initial.V), q).tocsr()
    ai, bi = a.indices.reshape(-1, p + 1), b.indices.reshape(-1, q + 1)
    indices = (ai[:, :, None] * count_v + bi[:, None, :]).reshape(len(parameters), -1)
    values = (
        a.data.reshape(-1, p + 1)[:, :, None] * b.data.reshape(-1, q + 1)[:, None, :]
    ).reshape(len(parameters), -1)
    return indices, values


def basis_matrix(initial, parameters, weights=None):
    indices, values = basis_table(initial, parameters)
    if weights is not None:
        values = values * np.asarray(weights).ravel()[indices]
        values = values / values.sum(axis=1, keepdims=True)
    width = indices.shape[1]
    return csr_matrix(
        (
            values.ravel(),
            indices.ravel(),
            np.arange(0, width * (len(parameters) + 1), width),
        ),
        shape=(len(parameters), np.size(initial.W)),
    )


def evaluate(initial, controls, weights, parameters):
    return basis_matrix(initial, parameters, weights) @ controls


def control_penalty(value, layout):
    try:
        value = np.broadcast_to(
            np.asarray(value, dtype=float), np.shape(layout.initial.P)
        )
    except ValueError as error:
        raise ValueError(
            "control_regularization must broadcast to control-point shape"
        ) from error
    if not np.isfinite(value).all() or np.any(value < 0):
        raise ValueError("control_regularization must be finite and nonnegative")
    return value.reshape(layout.ndim, -1).T.copy()


def loss(
    layout,
    x,
    points,
    parameters,
    *,
    objective_scale=1.0,
    control_regularization=0.0,
    weight_regularization=0.0,
):
    controls, weights = layout.unpack(x)
    error = evaluate(layout.initial, controls, weights, parameters) - points
    value = 0.5 * np.mean(np.sum(error**2, axis=1))
    value += (
        0.5
        / layout.count
        * np.sum(
            control_penalty(control_regularization, layout)
            * (controls - layout.controls0) ** 2
        )
    )
    alpha = np.log(
        weights[layout.weight_indices] / layout.weights0[layout.weight_indices]
    )
    value += (
        0.5
        * weight_regularization
        / max(1, layout.nweight_variables)
        * np.sum(alpha**2)
    )
    return float(objective_scale * value), float(
        np.sqrt(np.mean(np.sum(error**2, axis=1)))
    )
