"""Sampled surface initialization with one local solve and implicit AD."""

from functools import partial

import jax
import jax.numpy as jnp

from .nurbs_surface import compute_nurbs_coordinates, compute_nurbs_derivatives
from ._point_projection import (
    Objective, check_geometry, checked_parameters, implicit_tangent,
    nearest_samples, refine, sample_axis,
)


def _objective_data(uv, P, W, U, V, point, p, q):
    derivatives = compute_nurbs_derivatives(P, W, p, q, U, V, uv[0], uv[1], 2, 2)[..., 0]
    residual = derivatives[0, 0] - point
    jacobian = jnp.stack((derivatives[1, 0], derivatives[0, 1]), axis=1)
    second = jnp.stack((jnp.stack((derivatives[2, 0], derivatives[1, 1])),
                        jnp.stack((derivatives[1, 1], derivatives[0, 2]))))
    return Objective(0.5 * jnp.sum(residual**2), jacobian.T @ residual,
                     jacobian.T @ jacobian + jnp.einsum("ijd,d->ij", second, residual),
                     jacobian, derivatives[0, 0])


def sample_surface(P, W, U, V, p, q, samples_per_span):
    u = sample_axis(U, p, W, samples_per_span)
    v = sample_axis(V, q, W, samples_per_span)
    u, v = jnp.meshgrid(u, v, indexing="ij")
    parameters = jnp.stack((u.ravel(), v.ravel()), axis=1)
    positions = compute_nurbs_coordinates(P, W, p, q, U, V, u.ravel(), v.ravel())
    return parameters, positions


def project_surface_points(P, W, U, V, points, p, q, maxiter, ftol, gtol, xtol,
                           samples_per_span):
    P = check_geometry(P, W, (U, V), (p, q))
    parameters, positions = sample_surface(P, W, U, V, p, q, samples_per_span)
    return project_surface_from_samples(P, W, U, V, points, parameters, positions,
                                        p, q, maxiter, ftol, gtol, xtol)


@partial(jax.custom_jvp, nondiff_argnums=(7, 8, 9, 10, 11, 12))
def project_surface_from_samples(P, W, U, V, points, parameters, positions,
                                 p, q, maxiter, ftol, gtol, xtol):
    """The same single-seed refinement for general and prepared evaluation."""
    seeds = nearest_samples(parameters, positions, points)
    lower, upper = jnp.array([U[p], V[q]]), jnp.array([U[-p - 1], V[-q - 1]])

    def solve(point, seed):
        data = lambda uv: _objective_data(uv, P, W, U, V, point, p, q)
        def costs(uv):
            values = compute_nurbs_coordinates(P, W, p, q, U, V, uv[:, 0], uv[:, 1])
            return 0.5 * jnp.sum((values - point[:, None])**2, axis=0)
        result = refine(data, costs, seed, lower, upper, point, (U, V), (p, q),
                        maxiter, ftol, gtol, xtol)
        return checked_parameters(result)

    return jax.vmap(solve, in_axes=(1, 0), out_axes=1)(points, seeds)


@project_surface_from_samples.defjvp
def _projection_jvp(p, q, maxiter, ftol, gtol, xtol, primals, tangents):
    P, W, U, V, points, _, _ = primals
    uv = project_surface_from_samples(*primals, p, q, maxiter, ftol, gtol, xtol)

    def stationarity(P, W, U, V, points):
        return jax.vmap(lambda x, point: _objective_data(x, P, W, U, V, point, p, q).gradient)(uv.T, points.T)

    _, rhs = jax.jvp(stationarity, primals[:5], tangents[:5])
    lower, upper = jnp.array([U[p], V[q]]), jnp.array([U[-p - 1], V[-q - 1]])
    lower_tangent = jnp.array([tangents[2][p], tangents[3][q]])
    upper_tangent = jnp.array([tangents[2][-p - 1], tangents[3][-q - 1]])

    def differentiate(x, point, rhs):
        objective = _objective_data(x, P, W, U, V, point, p, q)
        bound_tangent = jnp.where(x == lower, lower_tangent, upper_tangent)
        return implicit_tangent(x, objective, rhs, lower, upper, bound_tangent,
                                 point, (U, V), (p, q), gtol)

    return uv, jax.vmap(differentiate)(uv.T, points.T, rhs).T
