"""Sampled curve initialization and one safeguarded local solve per query."""

from functools import partial

import jax
import jax.numpy as jnp

from .nurbs_curve import compute_all_nurbs_derivatives, compute_nurbs_coordinates
from ._point_projection import (
    Objective, check_geometry, checked_parameters, implicit_tangent,
    nearest_samples, refine, sample_axis,
)


def _objective(P, W, U, point, p, u):
    derivatives = compute_all_nurbs_derivatives(P, W, p, U, u, 2)[..., 0]
    residual = derivatives[0] - point
    tangent = derivatives[1]
    return Objective(
        0.5 * jnp.sum(residual**2),
        jnp.atleast_1d(jnp.dot(residual, tangent)),
        jnp.atleast_2d(jnp.dot(tangent, tangent) + jnp.dot(residual, derivatives[2])),
        tangent[:, None], derivatives[0],
    )


def sample_curve(P, W, U, p, samples_per_span):
    u = sample_axis(U, p, W, samples_per_span)
    return u[:, None], compute_nurbs_coordinates(P, W, p, U, u)


def project_curve_points(P, W, U, points, p, maxiter, ftol, gtol, xtol, samples_per_span):
    P = check_geometry(P, W, (U,), (p,))
    parameters, positions = sample_curve(P, W, U, p, samples_per_span)
    return _project_from_samples(P, W, U, points, parameters, positions,
                                 p, maxiter, ftol, gtol, xtol)


@partial(jax.custom_jvp, nondiff_argnums=(6, 7, 8, 9, 10))
def _project_from_samples(P, W, U, points, parameters, positions, p, maxiter, ftol, gtol, xtol):
    seeds = nearest_samples(parameters, positions, points)
    lower, upper = U[p:p + 1], U[-p - 1:-p] if p else U[-1:]

    def solve(point, seed):
        data = lambda u: _objective(P, W, U, point, p, u)
        def costs(u):
            residual = compute_nurbs_coordinates(P, W, p, U, u[:, 0]) - point[:, None]
            return 0.5 * jnp.sum(residual**2, axis=0)
        result = refine(data, costs, seed, lower, upper, point, (U,), (p,),
                        maxiter, ftol, gtol, xtol)
        return checked_parameters(result)[0]

    return jax.vmap(solve, in_axes=(1, 0))(points, seeds)


def _sampled_tie(parameters, positions, point, ftol):
    """A conservative AD guard for equal sampled valleys, not a global proof."""
    costs = 0.5 * jnp.sum((positions - point[:, None])**2, axis=0)
    index = jnp.argmin(costs)
    ids = jnp.arange(costs.size)
    left = jax.lax.associative_scan(
        jnp.maximum, jnp.where(ids <= index, costs, -jnp.inf), reverse=True
    )
    right = jax.lax.associative_scan(
        jnp.maximum, jnp.where(ids >= index, costs, -jnp.inf)
    )
    barrier = jnp.where(ids <= index, left, right)
    tolerance = ftol * jnp.maximum(1.0, jnp.abs(costs[index]))
    separated = jnp.abs(parameters[:, 0] - parameters[index, 0]) > 1e-5 * (
        parameters[-1, 0] - parameters[0, 0]
    )
    return jnp.any(separated & (jnp.abs(costs - costs[index]) <= tolerance)
                   & (barrier > costs[index] + tolerance))


@_project_from_samples.defjvp
def _projection_jvp(p, maxiter, ftol, gtol, xtol, primals, tangents):
    P, W, U, points, parameters, positions = primals
    u = _project_from_samples(*primals, p, maxiter, ftol, gtol, xtol)

    def stationarity(P, W, U, points):
        return jax.vmap(lambda t, point: _objective(P, W, U, point, p, t[None]).gradient)(u, points.T)

    _, rhs = jax.jvp(stationarity, primals[:4], tangents[:4])
    lower, upper = U[p:p + 1], U[-p - 1:-p] if p else U[-1:]

    def differentiate(t, point, rhs):
        x = t[None]
        objective = _objective(P, W, U, point, p, x)
        bound_tangent = jnp.where(x == lower, tangents[2][p], tangents[2][-p - 1])
        du = implicit_tangent(x, objective, rhs, lower, upper, bound_tangent,
                              point, (U,), (p,), gtol)[0]
        return jnp.where(_sampled_tie(parameters, positions, point, ftol), jnp.nan, 1.0) * du

    return u, jax.vmap(differentiate)(u, points.T, rhs)
