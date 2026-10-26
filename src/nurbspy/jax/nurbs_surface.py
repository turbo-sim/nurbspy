"""Differentiable tensor-product NURBS surfaces with the NumPy surface API."""

from math import comb

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from .nurbs_basis_functions import (
    compute_basis_polynomials,
    compute_all_basis_polynomials_derivatives,
    _compute_basis_derivatives,
)
from .nurbs_curve import NurbsCurve

__all__ = ["NurbsSurface", "make_offset_surface"]


def _parameters(u, v):
    """Normalize scalar or paired one-dimensional parameters without broadcasting."""
    u, v = jnp.atleast_1d(u), jnp.atleast_1d(v)
    if u.ndim != 1 or v.ndim != 1 or u.shape != v.shape:
        raise ValueError(
            "u and v must be scalars or one-dimensional arrays of equal size"
        )
    return u, v


def compute_bspline_coordinates(P, p, q, U, V, u, v):
    """Evaluate a polynomial patch at paired parameters, returning (ndim, N)."""
    u, v = _parameters(u, v)
    P = jnp.asarray(P)
    basis_u = compute_basis_polynomials(P.shape[1] - 1, p, U, u)
    basis_v = compute_basis_polynomials(P.shape[2] - 1, q, V, v)
    return jnp.einsum("dij,ik,jk->dk", P, basis_u, basis_v)


def compute_nurbs_coordinates(P, W, p, q, U, V, u, v):
    """Evaluate a rational patch with control points (ndim, Nu, Nv) and weights (Nu, Nv)."""
    P, W = jnp.asarray(P), jnp.asarray(W)
    homogeneous = jnp.concatenate((P * W, W[None]), axis=0)
    values = compute_bspline_coordinates(homogeneous, p, q, U, V, u, v)
    return values[:-1] / values[-1:]


def compute_bspline_derivatives(P, p, q, U, V, u, v, up_to_order_u, up_to_order_v):
    """Return partials with shape (ku+1, kv+1, ndim, N), zero above each degree.

    Degrees and requested orders are nonnegative Python integers, fixed while
    tracing. Parameter derivatives at knots follow the basis span convention.
    """
    u, v = _parameters(u, v)
    if up_to_order_u < 0 or up_to_order_v < 0:
        raise ValueError("Derivative orders must be nonnegative")
    P = jnp.asarray(P)
    ku, kv = min(p, up_to_order_u), min(q, up_to_order_v)
    # Reuse the shared recurrence, forming only rows needed by this call.
    basis_u = _compute_basis_derivatives(P.shape[1] - 1, p, U, u, max_order=ku)
    basis_v = _compute_basis_derivatives(P.shape[2] - 1, q, V, v, max_order=kv)
    values = jnp.einsum("dij,aik,bjk->abdk", P, basis_u[: ku + 1], basis_v[: kv + 1])
    # Polynomial zero partials are needed by the rational recurrence as well.
    return jnp.pad(
        values, ((0, up_to_order_u - ku), (0, up_to_order_v - kv), (0, 0), (0, 0))
    )


def compute_nurbs_derivatives(P, W, p, q, U, V, u, v, up_to_order_u, up_to_order_v):
    """Return rational mixed partials through the requested orders.

    The output has shape (ku+1, kv+1, ndim, N). Unlike polynomial partials,
    rational partials may be nonzero above either degree. Differentiating the
    identity w*S=A gives the two-dimensional quotient-rule recurrence.
    """
    P, W = jnp.asarray(P), jnp.asarray(W)
    homogeneous = jnp.concatenate((P * W, W[None]), axis=0)
    derivatives = compute_bspline_derivatives(
        homogeneous, p, q, U, V, u, v, up_to_order_u, up_to_order_v
    )
    numerator, weights = derivatives[:, :, :-1], derivatives[:, :, -1:]
    result = jnp.zeros_like(numerator)
    for k in range(up_to_order_u + 1):
        for ell in range(up_to_order_v + 1):
            value = numerator[k, ell]
            for i in range(k + 1):
                for j in range(ell + 1):
                    if i or j:
                        value = (
                            value
                            - comb(k, i)
                            * comb(ell, j)
                            * weights[i, j]
                            * result[k - i, ell - j]
                        )
            result = result.at[k, ell].set(value / weights[0, 0])
    return result


def _clamped_knots(count, degree):
    """Build an open uniform knot vector for a statically sized control net."""
    return jnp.concatenate(
        (jnp.zeros(degree), jnp.linspace(0, 1, count - degree + 1), jnp.ones(degree))
    )


class NurbsSurface(eqx.Module):
    """A differentiable surface with paired parameters and NumPy-compatible shapes.

    ``control_points`` has shape (ndim, Nu, Nv), and ``weights`` has shape
    (Nu, Nv). Omitting degrees makes a Bezier patch; omitted knots are open
    uniform. Values and derivatives have shape (ndim, N), including (ndim, 1)
    for scalar parameters. To evaluate a grid, flatten an ``ij`` meshgrid.

    Degrees and control-net dimensions are static. Points, weights and knots
    remain differentiable array leaves. Normals and curvature require a
    regular three-dimensional patch; they are undefined at singular points.
    Pass changing parameters as arrays to reuse compiled evaluations; Python
    scalar arguments are static under ``eqx.filter_jit``, as in JAX curves.
    """

    P: jax.Array
    W: jax.Array
    p: int
    q: int
    U: jax.Array
    V: jax.Array
    ndim: int
    surface_type: str

    def __init__(
        self,
        control_points=None,
        weights=None,
        u_degree=None,
        v_degree=None,
        u_knots=None,
        v_knots=None,
    ):
        """Construct a polynomial/rational Bezier or B-spline surface."""
        if control_points is None:
            if any(
                x is not None for x in (weights, u_degree, v_degree, u_knots, v_knots)
            ):
                raise ValueError("control_points are required")
            self.P, self.W = jnp.zeros((0, 0, 0)), jnp.zeros((0, 0))
            self.U, self.V = jnp.zeros(0), jnp.zeros(0)
            self.p = self.q = self.ndim = 0
            self.surface_type = "Void"
            return
        P = jnp.asarray(control_points)
        if P.ndim != 3 or min(P.shape) < 1:
            raise ValueError("control_points must have shape (ndim, Nu, Nv)")
        W = jnp.ones(P.shape[1:]) if weights is None else jnp.asarray(weights)
        dtype = jnp.result_type(P, W, 1.0)
        self.P, self.W = P.astype(dtype), W.astype(dtype)
        if W.shape != P.shape[1:]:
            raise ValueError("weights must match the control-net shape (Nu, Nv)")
        self.p = P.shape[1] - 1 if u_degree is None else u_degree
        self.q = P.shape[2] - 1 if v_degree is None else v_degree
        for degree, count in ((self.p, P.shape[1]), (self.q, P.shape[2])):
            if not isinstance(degree, (int, np.integer)) or not 0 <= degree < count:
                raise ValueError(
                    "Each degree must be a static integer smaller than its control-point count"
                )
        self.U = (
            _clamped_knots(P.shape[1], self.p)
            if u_knots is None
            else jnp.asarray(u_knots)
        )
        self.V = (
            _clamped_knots(P.shape[2], self.q)
            if v_knots is None
            else jnp.asarray(v_knots)
        )
        if self.U.shape != (P.shape[1] + self.p + 1,) or self.V.shape != (
            P.shape[2] + self.q + 1,
        ):
            raise ValueError(
                "Knot-vector lengths must equal control-point count plus degree plus one"
            )
        self.ndim = P.shape[0]
        bezier = all(x is None for x in (u_degree, v_degree, u_knots, v_knots))
        self.surface_type = (
            ("Bezier" if weights is None else "R-Bezier")
            if bezier
            else ("B-Spline" if weights is None else "NURBS")
        )

    @property
    def U_values(self):
        """Distinct u knots for host-side CAD export; unavailable inside JIT."""
        return np.unique(np.asarray(self.U))

    @property
    def U_mults(self):
        """Multiplicities of distinct u knots for host-side CAD export."""
        return np.unique(np.asarray(self.U), return_counts=True)[1]

    @property
    def V_values(self):
        """Distinct v knots for host-side CAD export; unavailable inside JIT."""
        return np.unique(np.asarray(self.V))

    @property
    def V_mults(self):
        """Multiplicities of distinct v knots for host-side CAD export."""
        return np.unique(np.asarray(self.V), return_counts=True)[1]

    compute_nurbs_coordinates = staticmethod(compute_nurbs_coordinates)
    compute_bspline_coordinates = staticmethod(compute_bspline_coordinates)
    compute_nurbs_derivatives = staticmethod(compute_nurbs_derivatives)
    compute_bspline_derivatives = staticmethod(compute_bspline_derivatives)

    @eqx.filter_jit
    def get_value(self, u, v):
        """Evaluate paired parameters as an array of shape (ndim, N)."""
        return compute_nurbs_coordinates(
            self.P, self.W, self.p, self.q, self.U, self.V, u, v
        )

    @eqx.filter_jit
    def get_derivative(self, u, v, order_u, order_v):
        """Evaluate the requested mixed partial; derivative orders are static."""
        return compute_nurbs_derivatives(
            self.P, self.W, self.p, self.q, self.U, self.V, u, v, order_u, order_v
        )[order_u, order_v]

    @eqx.filter_jit
    def get_normals(self, u, v):
        """Return oriented unit normals S_u cross S_v, with shape (3, N)."""
        if self.ndim != 3:
            raise ValueError("Surface normals require three-dimensional control points")
        derivatives = compute_nurbs_derivatives(
            self.P, self.W, self.p, self.q, self.U, self.V, u, v, 1, 1
        )
        normal = jnp.cross(derivatives[1, 0], derivatives[0, 1], axis=0)
        return normal / jnp.linalg.norm(normal, axis=0)

    @eqx.filter_jit
    def get_curvature(self, u, v):
        """Return signed mean and Gaussian curvature, each with shape (N,)."""
        if self.ndim != 3:
            raise ValueError(
                "Surface curvature requires three-dimensional control points"
            )
        derivatives = compute_nurbs_derivatives(
            self.P, self.W, self.p, self.q, self.U, self.V, u, v, 2, 2
        )
        su, sv = derivatives[1, 0], derivatives[0, 1]
        normal = jnp.cross(su, sv, axis=0)
        normal = normal / jnp.linalg.norm(normal, axis=0)
        E, F, G = (
            jnp.sum(su * su, axis=0),
            jnp.sum(su * sv, axis=0),
            jnp.sum(sv * sv, axis=0),
        )
        L = jnp.sum(derivatives[2, 0] * normal, axis=0)
        M = jnp.sum(derivatives[1, 1] * normal, axis=0)
        N = jnp.sum(derivatives[0, 2] * normal, axis=0)
        determinant = E * G - F * F
        return (E * N - 2 * F * M + G * L) / (2 * determinant), (
            L * N - M * M
        ) / determinant

    @eqx.filter_jit
    def get_isocurve_u(self, u0):
        """Return the rational curve S(u0,v), differentiable in scalar u0."""
        basis = compute_basis_polynomials(
            self.P.shape[1] - 1, self.p, self.U, jnp.asarray(u0)
        ).reshape(-1)
        homogeneous = jnp.concatenate((self.P * self.W, self.W[None]))
        curve = jnp.einsum("dij,i->dj", homogeneous, basis)
        return NurbsCurve(curve[:-1] / curve[-1:], curve[-1], self.q, self.V)

    @eqx.filter_jit
    def get_isocurve_v(self, v0):
        """Return the rational curve S(u,v0), differentiable in scalar v0."""
        basis = compute_basis_polynomials(
            self.P.shape[2] - 1, self.q, self.V, jnp.asarray(v0)
        ).reshape(-1)
        homogeneous = jnp.concatenate((self.P * self.W, self.W[None]))
        curve = jnp.einsum("dij,j->di", homogeneous, basis)
        return NurbsCurve(curve[:-1] / curve[-1:], curve[-1], self.p, self.U)

    def _attach(self, other, direction):
        """Map two compatible patches to half of a normalized parameter domain."""
        if self.p != other.p or self.q != other.q or self.ndim != other.ndim:
            raise ValueError(
                "Attached patches must have matching degrees and dimensions"
            )
        transverse = self.V if direction == 0 else self.U
        other_transverse = other.V if direction == 0 else other.U
        if transverse.shape != other_transverse.shape:
            raise ValueError("Attached patches must have matching transverse knots")
        points = eqx.error_if(
            other.P,
            jnp.any(jnp.abs(transverse - other_transverse) > 1e-12),
            "Attached patches must have matching transverse knots",
        )
        first_knots, second_knots = (
            (self.U, other.U) if direction == 0 else (self.V, other.V)
        )
        degree = self.p if direction == 0 else self.q
        # Full seam multiplicity retains each patch exactly, including a gap
        # if the caller supplied incompatible geometric boundary curves.
        knots = jnp.concatenate(
            (
                jnp.zeros(degree + 1),
                first_knots[degree + 1 : -degree - 1] / 2,
                jnp.full(degree + 1, 0.5),
                0.5 + second_knots[degree + 1 : -degree - 1] / 2,
                jnp.ones(degree + 1),
            )
        )
        return NurbsSurface(
            jnp.concatenate((self.P, points), axis=direction + 1),
            jnp.concatenate((self.W, other.W), axis=direction),
            self.p,
            self.q,
            knots if direction == 0 else self.U,
            self.V if direction == 0 else knots,
        )

    @eqx.filter_jit
    def attach_nurbs_udir(self, new_nurbs):
        """Attach a patch along u, mapping the originals to [0,.5] and [.5,1]."""
        return self._attach(new_nurbs, 0)

    @eqx.filter_jit
    def attach_nurbs_vdir(self, new_nurbs):
        """Attach a patch along v, mapping the originals to [0,.5] and [.5,1]."""
        return self._attach(new_nurbs, 1)

    @eqx.filter_jit
    def project_point_to_surface(self, P, maxiter=200, ftol=1e-6, gtol=1e-6):
        """Return the bounded closest-point parameters (u,v) in [0,1]^2.

        A multistart projected Newton solve minimizes half the squared
        distance. ``maxiter`` and tolerances are static. A custom JVP uses
        implicit differentiation of the converged stationarity conditions,
        including fixed active bounds, in both forward and reverse mode.

        Derivatives require a locally unique regular minimum with a nonsingular
        reduced Hessian and a stable active set. They are not defined at ties,
        transitions onto/off a boundary, or singular surface points. The search
        is local with multiple starts; it does not guarantee a global minimum
        on arbitrary folded patches. Plotting and solver diagnostics are not
        part of this differentiable calculation.
        """
        from ._surface_projection import project_surface_point

        point = jnp.asarray(P).reshape(-1)
        if point.shape != (self.ndim,):
            raise ValueError("The query point must match the surface dimension")
        if maxiter < 0 or ftol < 0 or gtol < 0:
            raise ValueError("Iteration limit and tolerances must be nonnegative")
        parameters = project_surface_point(
            self.P, self.W, self.U, self.V, point, self.p, self.q, maxiter, ftol, gtol
        )
        return parameters[0], parameters[1]

    @eqx.filter_jit
    def project_points(self, points, maxiter=200, ftol=1e-6, gtol=1e-6):
        """Project points shaped (ndim,N), returning parameters shaped (2,N)."""
        points = jnp.asarray(points)
        if points.ndim != 2 or points.shape[0] != self.ndim:
            raise ValueError("points must have shape (ndim,N)")
        return jnp.stack(
            jax.vmap(
                lambda point: self.project_point_to_surface(point, maxiter, ftol, gtol),
                in_axes=1,
            )(points)
        )

    class PointToSurfaceProjectionProblem:
        """Compatibility wrapper for the NumPy distance-objective helper."""

        def __init__(self, S, dS, P):
            """Store evaluation callables and a point as an (ndim,1) array."""
            self.S_func, self.dS_func = S, dS
            self.P = jnp.asarray(P).reshape(-1, 1)

        @staticmethod
        def get_bounds():
            """Return the normalized parameter bounds of the projection problem."""
            return [0.0, 0.0], [1.0, 1.0]

        def fitness(self, x):
            """Return Euclidean distance in a length-one array."""
            residual = self.S_func(x[0], x[1]) - self.P
            return jnp.atleast_1d(jnp.linalg.norm(residual))

        def gradient(self, x):
            """Return the distance gradient, using zero at an exact intersection."""
            residual = self.S_func(x[0], x[1]) - self.P
            distance = jnp.linalg.norm(residual)
            denominator = jnp.where(distance > 0, distance, 1.0)
            return (
                jnp.stack(
                    (
                        jnp.sum(residual * self.dS_func(x[0], x[1], 1, 0)),
                        jnp.sum(residual * self.dS_func(x[0], x[1], 0, 1)),
                    )
                )
                / denominator
            )

    def _as_numpy_surface(self):
        """Transfer geometry to the host for Matplotlib/CAD-style utilities only."""
        from ..numpy.nurbs_surface import NurbsSurface as NumpySurface

        return NumpySurface(
            np.asarray(self.P),
            np.asarray(self.W),
            self.p,
            self.q,
            np.asarray(self.U),
            np.asarray(self.V),
        )

    def plot(
        self,
        fig=None,
        ax=None,
        surface=True,
        surface_color="blue",
        colorbar=False,
        boundary=True,
        control_points=False,
        normals=False,
        axis_off=False,
        ticks_off=False,
        Nu=50,
        Nv=50,
        isocurves_u=None,
        isocurves_v=None,
    ):
        """Plot on the host with the NumPy API; plotting is outside JIT and autodiff."""
        return self._as_numpy_surface().plot(
            fig,
            ax,
            surface,
            surface_color,
            colorbar,
            boundary,
            control_points,
            normals,
            axis_off,
            ticks_off,
            Nu,
            Nv,
            isocurves_u,
            isocurves_v,
        )

    def plot_surface(
        self, fig, ax, color="blue", alpha=0.30, colorbar=False, Nu=50, Nv=50
    ):
        """Draw sampled surface values or curvature colors on host axes."""
        return self._as_numpy_surface().plot_surface(
            fig, ax, color, alpha, colorbar, Nu, Nv
        )

    def plot_boundary(
        self,
        fig,
        ax,
        color="black",
        linewidth=1.0,
        linestyle="-",
        south=True,
        north=True,
        east=True,
        west=True,
    ):
        """Draw the four boundary curves on host axes."""
        return self._as_numpy_surface().plot_boundary(
            fig, ax, color, linewidth, linestyle, south, north, east, west
        )

    def plot_isocurve_u(
        self, fig, ax, u_values, color="black", linewidth=1.0, linestyle="-"
    ):
        """Draw sections with fixed u on host axes."""
        return self._as_numpy_surface().plot_isocurve_u(
            fig, ax, u_values, color, linewidth, linestyle
        )

    def plot_isocurve_v(
        self, fig, ax, v_values, color="black", linewidth=1.0, linestyle="-"
    ):
        """Draw sections with fixed v on host axes."""
        return self._as_numpy_surface().plot_isocurve_v(
            fig, ax, v_values, color, linewidth, linestyle
        )

    def plot_control_points(
        self,
        fig,
        ax,
        color="red",
        linewidth=1.0,
        linestyle="-",
        markersize=5,
        markerstyle="o",
    ):
        """Draw the control net on host axes."""
        return self._as_numpy_surface().plot_control_points(
            fig, ax, color, linewidth, linestyle, markersize, markerstyle
        )

    def plot_normals(self, fig, ax, number_u=10, number_v=10, scale=0.075):
        """Draw sampled unit normals on host axes."""
        return self._as_numpy_surface().plot_normals(fig, ax, number_u, number_v, scale)

    def rescale_plot(self, fig, ax):
        """Set equal three-dimensional plot scales on the host."""
        return self._as_numpy_surface().rescale_plot(fig, ax)

    def plot_curvature(self, fig=None, ax=None, curvature_type="mean"):
        """Plot mean or Gaussian curvature over parameter space on the host."""
        return self._as_numpy_surface().plot_curvature(fig, ax, curvature_type)


def make_offset_surface(surface, offset):
    """Approximate S-offset*N with a sampled polynomial control net.

    Match the NumPy algorithm: use ten times as many control points in each
    direction, retain the degrees, and use uniform clamped knots. This is
    exact for planes but is not an exact general NURBS offset construction.
    """
    nu, nv = 10 * surface.P.shape[1], 10 * surface.P.shape[2]
    u, v = jnp.meshgrid(jnp.linspace(0, 1, nu), jnp.linspace(0, 1, nv), indexing="ij")
    u, v = u.ravel(), v.ravel()
    points = surface.get_value(u, v) - offset * surface.get_normals(u, v)
    return NurbsSurface(
        points.reshape(3, nu, nv), u_degree=surface.p, v_degree=surface.q
    )
