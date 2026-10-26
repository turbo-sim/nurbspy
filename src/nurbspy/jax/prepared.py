"""Explicit reusable basis tables for fixed knots and evaluation parameters.

Preparation stores polynomial basis functions, not rational geometry. Control
points and weights are supplied on every evaluation and remain differentiable.
There is no hidden cache: rebuild a prepared object when knots, degree, sample
locations or the required derivative orders change. If knots or parameters are
optimization variables, keep preparation inside the differentiated function.
"""

from math import comb as _comb
from numbers import Integral as _Integral

import equinox as eqx
import jax
import jax.numpy as jnp

from .nurbs_basis_functions import _compute_basis_derivatives

__all__ = [
    "PreparedCurveBasis",
    "PreparedSurfaceBasis",
    "prepare_curve",
    "prepare_surface",
]


def _order(value, name):
    """Validate shape-determining metadata without converting traced arrays."""
    if isinstance(value, bool) or not isinstance(value, _Integral):
        raise TypeError(f"{name} must be a static nonnegative integer")
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return int(value)


def _samples(values, name):
    values = jnp.atleast_1d(jnp.asarray(values))
    if values.ndim != 1 or values.size == 0:
        raise ValueError(f"{name} must be a scalar or a nonempty one-dimensional array")
    return values


def _knots(values, count, degree, name):
    degree = _order(degree, f"{name} degree")
    if not 0 <= degree < count:
        raise ValueError("Degree must be smaller than the positive control-point count")
    values = jnp.asarray(values)
    if values.shape != (count + degree + 1,):
        raise ValueError(f"{name} length must equal control-point count plus degree plus one")
    return values, degree


def _curve_inputs(points, weights, count):
    points = jnp.asarray(points)
    if points.ndim != 2 or points.shape[0] == 0 or points.shape[1] != count:
        raise ValueError(f"P must have shape (ndim, {count}) with positive ndim")
    weights = jnp.ones(count, dtype=points.dtype) if weights is None else jnp.asarray(weights)
    if weights.shape != (count,):
        raise ValueError(f"W must have shape ({count},)")
    return points, weights


def _surface_inputs(points, weights, count_u, count_v):
    points = jnp.asarray(points)
    if points.ndim != 3 or points.shape[0] == 0 or points.shape[1:] != (count_u, count_v):
        raise ValueError(f"P must have shape (ndim, {count_u}, {count_v}) with positive ndim")
    weights = jnp.ones((count_u, count_v)) if weights is None else jnp.asarray(weights)
    if weights.shape != (count_u, count_v):
        raise ValueError(f"W must have shape ({count_u}, {count_v})")
    # Match NurbsSurface construction, including integer input promotion.
    dtype = jnp.result_type(points, weights, 1.0)
    return points.astype(dtype), weights.astype(dtype)


def _curve_partials(points, weights, basis, order):
    """Contract a polynomial table, then apply the rational quotient recurrence."""
    homogeneous_points = jnp.concatenate((points * weights, weights[None]), axis=0)
    available = min(order + 1, basis.shape[0])
    homogeneous = jnp.einsum(
        "di,riu->rdu", homogeneous_points, basis[:available]
    ).astype(homogeneous_points.dtype)
    homogeneous = jnp.pad(homogeneous, ((0, order + 1 - available), (0, 0), (0, 0)))
    numerator, weight_partials = homogeneous[:, :-1], homogeneous[:, -1:]
    result = []
    for derivative in range(order + 1):
        value = numerator[derivative]
        for index in range(1, derivative + 1):
            value = value - (
                _comb(derivative, index)
                * weight_partials[index]
                * result[derivative - index]
            )
        # Preserve the existing curve derivative API's control-point dtype.
        result.append((value / weight_partials[0]).astype(points.dtype))
    return jnp.stack(result)


def _surface_contraction(homogeneous_points, basis_u, basis_v, grid):
    # Contract the independent 1D tables directly. In particular, grid mode
    # never stores a (control_u, control_v, samples_u, samples_v) product table.
    if grid:
        return jnp.einsum("dij,aik,bjl->abdkl", homogeneous_points, basis_u, basis_v)
    return jnp.einsum("dij,aik,bjk->abdk", homogeneous_points, basis_u, basis_v)


def _surface_partials(points, weights, basis_u, basis_v, order_u, order_v, grid):
    homogeneous_points = jnp.concatenate((points * weights, weights[None]), axis=0)
    available_u = min(order_u + 1, basis_u.shape[0])
    available_v = min(order_v + 1, basis_v.shape[0])
    homogeneous = _surface_contraction(
        homogeneous_points, basis_u[:available_u], basis_v[:available_v], grid
    )
    padding = [(0, order_u + 1 - available_u), (0, order_v + 1 - available_v)]
    homogeneous = jnp.pad(homogeneous, padding + [(0, 0)] * (homogeneous.ndim - 2))
    numerator, weight_partials = homogeneous[:, :, :-1], homogeneous[:, :, -1:]
    result = jnp.zeros_like(numerator)
    # w*S=A, differentiated in both parameter directions. Polynomial orders
    # above either degree are zero; rational derivatives generally are not.
    for order_i in range(order_u + 1):
        for order_j in range(order_v + 1):
            value = numerator[order_i, order_j]
            for index_i in range(order_i + 1):
                for index_j in range(order_j + 1):
                    if index_i or index_j:
                        value = value - (
                            _comb(order_i, index_i)
                            * _comb(order_j, index_j)
                            * weight_partials[index_i, index_j]
                            * result[order_i - index_i, order_j - index_j]
                        )
            result = result.at[order_i, order_j].set(value / weight_partials[0, 0])
    return result


class PreparedCurveBasis(eqx.Module):
    """Reusable polynomial curve basis tables, constructed by ``prepare_curve``.

    ``basis`` has shape ``(min(degree,max_order)+1, n_control, n_samples)``.
    No control points or weights are retained. Omitting W means unit weights,
    including when the curve used for preparation originally had other weights.
    Evaluation can use any positive spatial dimension at the same control count.
    """

    basis: jax.Array
    degree: int = eqx.field(static=True)
    n_control: int = eqx.field(static=True)
    max_order: int = eqx.field(static=True)

    def _require_order(self, order):
        order = _order(order, "order")
        if order > self.max_order:
            raise ValueError(
                "Requested derivative order exceeds the prepared maximum; prepare again"
            )
        return order

    @eqx.filter_jit
    def get_value(self, P, W=None):
        """Evaluate supplied geometry, returning ``(ndim,n_samples)``."""
        P, W = _curve_inputs(P, W, self.n_control)
        homogeneous = jnp.concatenate((P * W, W[None]), axis=0) @ self.basis[0]
        return homogeneous[:-1] / homogeneous[-1:]

    @eqx.filter_jit
    def get_derivatives(self, P, W=None):
        """Return all partials through max_order, shape ``(max_order+1,ndim,N)``."""
        P, W = _curve_inputs(P, W, self.n_control)
        return _curve_partials(P, W, self.basis, self.max_order)

    @eqx.filter_jit
    def get_derivative(self, P, W=None, order=1):
        """Evaluate one static derivative order within the prepared maximum."""
        order = self._require_order(order)
        P, W = _curve_inputs(P, W, self.n_control)
        return _curve_partials(P, W, self.basis, order)[order]

    @eqx.filter_jit
    def get_tangent(self, P, W=None):
        """Return unit tangents; undefined at zero speed."""
        tangent = self.get_derivative(P, W, order=1)
        return tangent / jnp.linalg.norm(tangent, axis=0)

    @eqx.filter_jit
    def get_curvature(self, P, W=None):
        """Return unsigned curvature of a regular planar or spatial curve."""
        self._require_order(2)
        P, W = _curve_inputs(P, W, self.n_control)
        if P.shape[0] not in (2, 3):
            raise ValueError("Curve curvature requires two- or three-dimensional points")
        derivatives = _curve_partials(P, W, self.basis, 2)
        first = jnp.pad(derivatives[1], ((0, 3 - P.shape[0]), (0, 0)))
        second = jnp.pad(derivatives[2], ((0, 3 - P.shape[0]), (0, 0)))
        numerator = jnp.linalg.norm(jnp.cross(first, second, axis=0), axis=0)
        denominator = jnp.linalg.norm(first, axis=0) ** 3
        # Retain the existing curve API's zero-speed value convention. This
        # does not make curvature or its derivative defined at singular points.
        denominator = jnp.where(denominator == 0, 1.0, denominator)
        return numerator / denominator


class PreparedSurfaceBasis(eqx.Module):
    """Reusable factorized surface tables, constructed by ``prepare_surface``.

    Paired output has shape ``(ndim,N)``. Grid output has shape
    ``(ndim,Nu_samples,Nv_samples)`` in ``ij`` ordering. Derivative tables prepend
    ``(max_order_u+1,max_order_v+1)``. P/W remain dynamic; no geometry is stored.
    """

    basis_u: jax.Array
    basis_v: jax.Array
    degree_u: int = eqx.field(static=True)
    degree_v: int = eqx.field(static=True)
    n_control_u: int = eqx.field(static=True)
    n_control_v: int = eqx.field(static=True)
    max_order_u: int = eqx.field(static=True)
    max_order_v: int = eqx.field(static=True)
    grid: bool = eqx.field(static=True)

    def _require_orders(self, order_u, order_v):
        order_u, order_v = _order(order_u, "order_u"), _order(order_v, "order_v")
        if order_u > self.max_order_u or order_v > self.max_order_v:
            raise ValueError(
                "Requested derivative order exceeds a prepared maximum; prepare again"
            )
        return order_u, order_v

    @eqx.filter_jit
    def get_value(self, P, W=None):
        """Evaluate supplied geometry at fixed paired points or a tensor grid."""
        P, W = _surface_inputs(P, W, self.n_control_u, self.n_control_v)
        homogeneous = jnp.concatenate((P * W, W[None]), axis=0)
        values = _surface_contraction(
            homogeneous, self.basis_u[:1], self.basis_v[:1], self.grid
        )[0, 0]
        return values[:-1] / values[-1:]

    @eqx.filter_jit
    def get_derivatives(self, P, W=None):
        """Return the complete rectangle of prepared rational mixed partials."""
        P, W = _surface_inputs(P, W, self.n_control_u, self.n_control_v)
        return _surface_partials(
            P,
            W,
            self.basis_u,
            self.basis_v,
            self.max_order_u,
            self.max_order_v,
            self.grid,
        )

    @eqx.filter_jit
    def get_derivative(self, P, W=None, order_u=1, order_v=0):
        """Evaluate one mixed partial with static orders within prepared limits."""
        order_u, order_v = self._require_orders(order_u, order_v)
        P, W = _surface_inputs(P, W, self.n_control_u, self.n_control_v)
        return _surface_partials(
            P, W, self.basis_u, self.basis_v, order_u, order_v, self.grid
        )[order_u, order_v]

    @eqx.filter_jit
    def get_normals(self, P, W=None):
        """Return unit oriented normals; requires a regular 3D surface."""
        self._require_orders(1, 1)
        P, W = _surface_inputs(P, W, self.n_control_u, self.n_control_v)
        if P.shape[0] != 3:
            raise ValueError("Surface normals require three-dimensional points")
        derivatives = _surface_partials(P, W, self.basis_u, self.basis_v, 1, 1, self.grid)
        normal = jnp.cross(derivatives[1, 0], derivatives[0, 1], axis=0)
        return normal / jnp.linalg.norm(normal, axis=0)

    @eqx.filter_jit
    def get_curvature(self, P, W=None):
        """Return (signed mean, Gaussian) curvature of a regular 3D surface."""
        self._require_orders(2, 2)
        P, W = _surface_inputs(P, W, self.n_control_u, self.n_control_v)
        if P.shape[0] != 3:
            raise ValueError("Surface curvature requires three-dimensional points")
        derivatives = _surface_partials(P, W, self.basis_u, self.basis_v, 2, 2, self.grid)
        su, sv = derivatives[1, 0], derivatives[0, 1]
        normal = jnp.cross(su, sv, axis=0)
        normal = normal / jnp.linalg.norm(normal, axis=0)
        E, F, G = jnp.sum(su * su, axis=0), jnp.sum(su * sv, axis=0), jnp.sum(sv * sv, axis=0)
        L = jnp.sum(derivatives[2, 0] * normal, axis=0)
        M = jnp.sum(derivatives[1, 1] * normal, axis=0)
        N = jnp.sum(derivatives[0, 2] * normal, axis=0)
        determinant = E * G - F * F
        return (E * N - 2 * F * M + G * L) / (2 * determinant), (L * N - M * M) / determinant


def prepare_curve(curve, u, max_order=2):
    """Prepare polynomial basis rows for a curve's fixed knots and samples.

    Only degree, knots and control count are read from ``curve``. Its points and
    weights are not retained. Supply both explicitly at evaluation for rational
    geometry. ``max_order`` is a static nonnegative integer and may exceed degree.
    Include this call inside AD when differentiating knots or sample parameters;
    preparing once outside that function intentionally fixes those inputs.
    """
    max_order = _order(max_order, "max_order")
    count = curve.P.shape[1]
    knots, degree = _knots(curve.U, count, curve.p, "U")
    u = _samples(u, "u")
    basis = _compute_basis_derivatives(count - 1, degree, knots, u, max_order=max_order)
    return PreparedCurveBasis(basis, degree, count, max_order)


def prepare_surface(surface, u, v, max_order_u=2, max_order_v=2, *, grid=False):
    """Prepare factorized polynomial basis rows for fixed surface parameters.

    ``grid=False`` uses scalar or equal-length paired arrays, like NurbsSurface.
    ``grid=True`` uses independent 1D u/v axes and returns an ``ij`` tensor grid.
    Only 1D basis tables are stored in either mode. Control points and rational
    weights are always explicit evaluation inputs. Rebuild for new knots/samples;
    include preparation inside AD if their sensitivities are required.
    """
    max_order_u = _order(max_order_u, "max_order_u")
    max_order_v = _order(max_order_v, "max_order_v")
    if not isinstance(grid, bool):
        raise TypeError("grid must be a static Python bool")
    count_u, count_v = surface.P.shape[1:]
    knots_u, degree_u = _knots(surface.U, count_u, surface.p, "U")
    knots_v, degree_v = _knots(surface.V, count_v, surface.q, "V")
    u, v = _samples(u, "u"), _samples(v, "v")
    if not grid and u.shape != v.shape:
        raise ValueError(
            "Paired u and v must have equal shapes; use grid=True for independent axes"
        )
    basis_u = _compute_basis_derivatives(count_u - 1, degree_u, knots_u, u, max_order=max_order_u)
    basis_v = _compute_basis_derivatives(count_v - 1, degree_v, knots_v, v, max_order=max_order_v)
    return PreparedSurfaceBasis(
        basis_u,
        basis_v,
        degree_u,
        degree_v,
        count_u,
        count_v,
        max_order_u,
        max_order_v,
        grid,
    )
