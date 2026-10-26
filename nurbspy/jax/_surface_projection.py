"""Bounded surface projection with implicit derivatives of a regular minimum."""

from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp

from .nurbs_surface import compute_nurbs_coordinates, compute_nurbs_derivatives


def _objective_data(uv, P, W, U, V, point, p, q):
    """Return half squared distance, its gradient and exact parameter Hessian."""
    derivatives = compute_nurbs_derivatives(P, W, p, q, U, V, uv[0], uv[1], 2, 2)[
        ..., 0
    ]
    residual = derivatives[0, 0] - point
    jacobian = jnp.stack((derivatives[1, 0], derivatives[0, 1]), axis=1)
    second = jnp.stack(
        (
            jnp.stack((derivatives[2, 0], derivatives[1, 1])),
            jnp.stack((derivatives[1, 1], derivatives[0, 2])),
        )
    )
    hessian = jacobian.T @ jacobian + jnp.einsum("ijd,d->ij", second, residual)
    return 0.5 * jnp.sum(residual**2), jacobian.T @ residual, hessian


def _active_bounds(uv, gradient):
    """Identify strictly outward gradients at lower or upper active bounds."""
    return ((uv <= 1e-10) & (gradient > 0)) | ((uv >= 1 - 1e-10) & (gradient < 0))


def _solve(P, W, U, V, point, p, q, maxiter, ftol, gtol):
    """Refine four sampled starting points with projected, damped Newton steps."""
    samples_u = jnp.concatenate(((U[:-1] + U[1:]) / 2, jnp.linspace(0, 1, 5)))
    samples_v = jnp.concatenate(((V[:-1] + V[1:]) / 2, jnp.linspace(0, 1, 5)))
    u, v = jnp.meshgrid(samples_u, samples_v, indexing="ij")
    samples = jnp.stack((u.ravel(), v.ravel()), axis=1)

    def costs(parameters):
        """Evaluate a batch of line-search candidates without derivative work."""
        values = compute_nurbs_coordinates(
            P, W, p, q, U, V, parameters[:, 0], parameters[:, 1]
        )
        return 0.5 * jnp.sum((values - point[:, None]) ** 2, axis=0)

    indices = jax.lax.top_k(-costs(samples), 4)[1]
    seeds = samples[indices]

    def refine(seed):
        """Minimize from one seed while retaining a bounded, nonincreasing objective."""

        def condition(state):
            """Continue until convergence or the static iteration limit."""
            iteration, _, done = state
            return (iteration < maxiter) & ~done

        def step(state):
            """Use an active-set Newton direction and a short backtracking search."""
            iteration, uv, _ = state
            value, gradient, hessian = _objective_data(uv, P, W, U, V, point, p, q)
            active = _active_bounds(uv, gradient)
            free = (~active).astype(uv.dtype)
            projected_gradient = gradient * free
            hessian = hessian * free[:, None] * free[None, :] + jnp.diag(
                active.astype(uv.dtype)
            )
            # Shift indefinite Hessians to obtain a descent direction, while
            # leaving the exact local Hessian unchanged near a regular minimum.
            smallest = jnp.linalg.eigvalsh(hessian)[0]
            shift = jnp.maximum(0.0, 1e-10 - smallest)
            direction = -jnp.linalg.solve(
                hessian + shift * jnp.eye(2), projected_gradient
            )
            direction = direction / jnp.maximum(1.0, jnp.max(jnp.abs(direction)))
            scales = jnp.concatenate((0.5 ** jnp.arange(14), jnp.zeros(1)))
            candidates = jnp.clip(uv[None] + scales[:, None] * direction, 0.0, 1.0)
            candidate_costs = costs(candidates)
            best = jnp.argmin(candidate_costs)
            updated = candidates[best]
            small_step = jnp.max(jnp.abs(updated - uv)) <= jnp.sqrt(ftol)
            small_change = jnp.abs(value - candidate_costs[best]) <= ftol * jnp.maximum(
                1.0, value
            )
            done = (jnp.max(jnp.abs(projected_gradient)) <= gtol) | (
                small_step & small_change
            )
            return iteration + 1, updated, done

        _, uv, _ = jax.lax.while_loop(condition, step, (0, seed, jnp.array(False)))
        _, gradient, _ = _objective_data(uv, P, W, U, V, point, p, q)
        residual = jnp.max(
            jnp.abs(jnp.where(_active_bounds(uv, gradient), 0.0, gradient))
        )
        return uv, residual

    parameters, residuals = jax.vmap(refine)(seeds)
    best = jnp.argmin(costs(parameters))
    result = parameters[best]
    failed = ~jnp.all(jnp.isfinite(result)) | (
        residuals[best] > jnp.maximum(gtol, jnp.sqrt(ftol))
    )
    return eqx.error_if(
        result,
        failed,
        "Surface projection did not converge; increase maxiter or inspect the surface",
    )


@partial(jax.custom_jvp, nondiff_argnums=(5, 6, 7, 8, 9))
def project_surface_point(P, W, U, V, point, p, q, maxiter, ftol, gtol):
    """Return a closest-point parameter pair with a regular-minimum custom JVP."""
    return _solve(P, W, U, V, point, p, q, maxiter, ftol, gtol)


@project_surface_point.defjvp
def _projection_jvp(p, q, maxiter, ftol, gtol, primals, tangents):
    """Differentiate stationarity on free coordinates and fixed active bounds."""
    P, W, U, V, point = primals
    uv = project_surface_point(*primals, p, q, maxiter, ftol, gtol)
    _, gradient, hessian = _objective_data(uv, P, W, U, V, point, p, q)
    active = _active_bounds(uv, gradient)

    def residual_for_geometry(P, W, U, V, point):
        """Hold the parameter solution fixed for the geometric partial derivative."""
        _, gradient, _ = _objective_data(uv, P, W, U, V, point, p, q)
        return jnp.where(active, 0.0, gradient)

    _, rhs = jax.jvp(residual_for_geometry, primals, tangents)
    # Active parameter derivatives vanish; free rows retain Hessian coupling.
    system = jnp.where(active[:, None], jnp.eye(2), hessian)
    tangent = jnp.linalg.solve(system, -rhs)
    return uv, tangent
