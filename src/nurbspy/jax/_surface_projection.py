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


def _sample_axes(U, V):
    """Return the original coarse-grid axes, including repeated seed locations."""
    return (
        jnp.concatenate(((U[:-1] + U[1:]) / 2, jnp.linspace(0, 1, 5))),
        jnp.concatenate(((V[:-1] + V[1:]) / 2, jnp.linspace(0, 1, 5))),
    )


def _sample_parameters(U, V):
    """Flatten the coarse seed grid in the same ij order as the general solver."""
    samples_u, samples_v = _sample_axes(U, V)
    u, v = jnp.meshgrid(samples_u, samples_v, indexing="ij")
    return jnp.stack((u.ravel(), v.ravel()), axis=1)


def _select_seeds(samples, sampled_points, point):
    """Select four original grid parameters from their sampled spatial points."""
    costs = 0.5 * jnp.sum((sampled_points - point[:, None]) ** 2, axis=0)
    indices = jax.lax.top_k(-costs, 4)[1]
    return samples[indices]


def _refine_seeds(P, W, U, V, point, seeds, p, q, maxiter, ftol, gtol):
    """Run the shared dynamic-parameter Newton solver from four supplied seeds.

    Only initial sampling can be prepared. Every Newton and line-search step
    evaluates the general surface at its changing parameters. Keeping this
    implementation shared avoids a second solver with different tolerances or
    convergence behavior for prepared projection.
    """

    def costs(parameters):
        """Evaluate a batch of line-search candidates without derivative work."""
        values = compute_nurbs_coordinates(
            P, W, p, q, U, V, parameters[:, 0], parameters[:, 1]
        )
        return 0.5 * jnp.sum((values - point[:, None]) ** 2, axis=0)

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


def _solve(P, W, U, V, point, p, q, maxiter, ftol, gtol):
    """Sample the original grid, then refine four projected Newton starts."""
    samples = _sample_parameters(U, V)
    sampled_points = compute_nurbs_coordinates(
        P, W, p, q, U, V, samples[:, 0], samples[:, 1]
    )
    seeds = _select_seeds(samples, sampled_points, point)
    return _refine_seeds(P, W, U, V, point, seeds, p, q, maxiter, ftol, gtol)


@partial(jax.custom_jvp, nondiff_argnums=(5, 6, 7, 8, 9))
def project_surface_point(P, W, U, V, point, p, q, maxiter, ftol, gtol):
    """Return a closest-point parameter pair with a regular-minimum custom JVP."""
    return _solve(P, W, U, V, point, p, q, maxiter, ftol, gtol)


@project_surface_point.defjvp
def _projection_jvp(p, q, maxiter, ftol, gtol, primals, tangents):
    """Differentiate stationarity on free coordinates and fixed active bounds."""
    uv = project_surface_point(*primals, p, q, maxiter, ftol, gtol)
    return uv, _implicit_tangent(uv, primals, tangents, p, q)


def _implicit_tangent(uv, primals, tangents, p, q):
    """Differentiate the converged minimum, independently of seed selection."""
    P, W, U, V, point = primals
    _, gradient, hessian = _objective_data(uv, P, W, U, V, point, p, q)
    active = _active_bounds(uv, gradient)

    def residual_for_geometry(P, W, U, V, point):
        """Hold the parameter solution fixed for the geometric partial derivative."""
        _, gradient, _ = _objective_data(uv, P, W, U, V, point, p, q)
        return jnp.where(active, 0.0, gradient)

    _, rhs = jax.jvp(residual_for_geometry, primals, tangents)
    # Active parameter derivatives vanish; free rows retain Hessian coupling.
    system = jnp.where(active[:, None], jnp.eye(2), hessian)
    return jnp.linalg.solve(system, -rhs)


@partial(jax.custom_jvp, nondiff_argnums=(6, 7, 8, 9, 10))
def project_surface_point_from_seeds(P, W, U, V, point, seeds, p, q, maxiter, ftol, gtol):
    """Use prepared seed locations with the same refinement and implicit AD.

    Seeds select a basin; they are not part of the converged stationarity
    equations. Their tangents therefore do not enter the implicit derivative.
    The same regular-minimum and stable-active-set assumptions as the general
    projection apply.
    """
    return _refine_seeds(P, W, U, V, point, seeds, p, q, maxiter, ftol, gtol)


@project_surface_point_from_seeds.defjvp
def _seeded_projection_jvp(p, q, maxiter, ftol, gtol, primals, tangents):
    """Reuse exact stationarity derivatives, excluding initialization tangents."""
    uv = project_surface_point_from_seeds(*primals, p, q, maxiter, ftol, gtol)
    return uv, _implicit_tangent(uv, primals[:5], tangents[:5], p, q)
