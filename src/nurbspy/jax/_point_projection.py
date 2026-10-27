"""Shared sampling and one bounded Newton solve for point projection.

Initialization evaluates positions, not derivatives, once per geometry/batch.
The closest sample starts exactly one solve. Finite sampling is a heuristic,
so a converged local minimum is not a certificate of global optimality.
"""

import math
import operator
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp


class Objective(NamedTuple):
    value: jax.Array
    gradient: jax.Array
    hessian: jax.Array
    jacobian: jax.Array
    position: jax.Array


class Minimum(NamedTuple):
    parameters: jax.Array
    converged: jax.Array
    regular: jax.Array


def validate_options(maxiter, samples_per_span, ftol, gtol, xtol):
    """Options are static, as are the number of samples and iteration budget."""
    if operator.index(maxiter) < 0:
        raise ValueError("maxiter must be nonnegative")
    if operator.index(samples_per_span) < 1:
        raise ValueError("samples_per_span must be positive")
    if any(not math.isfinite(x) or x < 0 for x in (ftol, gtol, xtol)):
        raise ValueError("Projection tolerances must be finite and nonnegative")


def check_geometry(P, W, knot_vectors, degrees):
    """Require finite continuous NURBS with positive rational weights.

    Value-dependent checks remain effective inside JIT. Interior multiplicity
    degree+1 is conservatively unsupported, even if two pieces happen to meet.
    """
    # The general curve derivative helper preserves P.dtype. Projection must
    # use real arithmetic even when users supplied integer control points.
    dtype = jnp.result_type(P, W, *knot_vectors, jnp.zeros(()))
    if jnp.issubdtype(dtype, jnp.complexfloating):
        raise ValueError("Projection requires real geometry")
    P = jnp.asarray(P, dtype=dtype)
    invalid = ~jnp.all(jnp.isfinite(P)) | ~jnp.all(jnp.isfinite(W)) | jnp.any(W <= 0)
    for knots, degree in zip(knot_vectors, degrees):
        multiplicity = jnp.sum(knots[:, None] == knots[None, :], axis=1)
        interior = (knots > knots[degree]) & (knots < knots[-degree - 1])
        invalid |= (~jnp.all(jnp.isfinite(knots)) | jnp.any(jnp.diff(knots) < 0)
                    | (knots[degree] >= knots[-degree - 1])
                    | jnp.any(interior & (multiplicity > degree)))
    return eqx.error_if(P, invalid,
                       "Projection requires finite geometry, positive weights and continuous nondecreasing knots")


def sample_axis(knots, degree, weights, count):
    """Sample each active knot span, keeping endpoints and a static shape.

    Very large weight ratios can compress most geometry between uniform
    samples. Reallocate half the interior budget to logarithmic endpoint
    clusters in that case. This practical safeguard is not an error bound.
    Repeated zero-width spans intentionally retain duplicate sample entries.
    """
    stop = knots.shape[0] - degree - 1
    boundaries = knots[degree:stop + 1]
    uniform = (jnp.arange(count) + 0.5) / count
    if count >= 8:
        clustered = count // 4
        middle = count - 2 * clustered
        ratio = jnp.maximum(1.0, jnp.max(weights) / jnp.min(weights))
        layer = 0.5 * ratio ** (-jnp.linspace(0.0, 1.0, clustered))
        mixed = jnp.concatenate(((jnp.arange(middle) + 0.5) / middle, layer, 1.0 - layer))
        fractions = jnp.where(ratio > 1e4, mixed, uniform)
    else:
        fractions = uniform
    interior = boundaries[:-1, None] + jnp.diff(boundaries)[:, None] * fractions
    return jnp.sort(jnp.concatenate((boundaries, interior.ravel())))


def nearest_samples(parameters, positions, points):
    """Select one sample per query, sharing the spatial grid across queries."""
    distance = jnp.sum((positions[:, :, None] - points[:, None, :]) ** 2, axis=0)
    return parameters[jnp.argmin(distance, axis=0)]


def active_bounds(x, gradient, lower, upper):
    """Use exact feasible bounds: a parameter epsilon can swallow thin layers."""
    return ((x <= lower) & (gradient > 0)) | ((x >= upper) & (gradient < 0))


def _curvature_conditions(objective, active):
    norms = jnp.linalg.norm(objective.jacobian, axis=0)
    norms = jnp.where(norms > 0, norms, 1.0)
    normalized = objective.hessian / norms[:, None] / norms[None, :]
    scale = jnp.maximum(jnp.max(jnp.abs(normalized)), 1.0)
    free = (~active).astype(normalized.dtype)
    reduced = normalized * free[:, None] * free[None, :] + jnp.diag(active * scale)
    smallest = jnp.linalg.eigvalsh(reduced)[0]
    eps = jnp.finfo(normalized.dtype).eps
    return smallest >= -64 * eps * scale, smallest > jnp.sqrt(eps) * scale


def knot_constraints(x, objective, data, point, knot_vectors, degrees, gtol):
    """Treat continuous C0 corners as one-sided local constraints.

    A minimum at a corner can have nonzero smooth gradient. The common
    residual is evaluated at the corner and paired with each side's tangent.
    C2 knots are ordinary smooth points and require no special handling.
    """
    constraints = []
    valid_sides = []
    tolerance = gtol * (1.0 + jnp.linalg.norm(objective.position - point))
    lower = jnp.stack([knots[p] for knots, p in zip(knot_vectors, degrees)])
    upper = jnp.stack([knots[-p - 1] for knots, p in zip(knot_vectors, degrees)])
    bounds = active_bounds(x, objective.gradient, lower, upper)
    for axis, (knots, degree) in enumerate(zip(knot_vectors, degrees)):
        count = jnp.sum(knots == x[axis])
        corner = ((x[axis] > knots[degree]) & (x[axis] < knots[-degree - 1])
                  & (count >= degree) & (count > 0))

        def check(_):
            left_x = x.at[axis].set(jnp.nextafter(x[axis], -jnp.inf))
            left = data(left_x)
            tangent = left.jacobian[:, axis]
            left_g = jnp.dot(objective.position - point, tangent)
            left_norm = jnp.linalg.norm(tangent)
            right_norm = jnp.linalg.norm(objective.jacobian[:, axis])
            signs = ((left_g <= tolerance * left_norm)
                     & (objective.gradient[axis] >= -tolerance * right_norm)
                     & jnp.isfinite(left_g))
            # A zero one-sided slope does not block that direction. Retain a
            # curvature check there, rather than masking a stationary maximum.
            left_strict = left_g < -tolerance * left_norm
            right_strict = objective.gradient[axis] > tolerance * right_norm
            left_minimum, _ = _curvature_conditions(left, bounds)
            right_minimum, _ = _curvature_conditions(objective, bounds)
            valid = signs & (left_strict | left_minimum) & (right_strict | right_minimum)
            return valid, valid

        constraint, valid = jax.lax.cond(
            corner, check, lambda _: (jnp.array(False), jnp.array(True)), None
        )
        constraints.append(constraint)
        valid_sides.append(valid)
    return jnp.stack(constraints), jnp.all(jnp.stack(valid_sides))


def minimum_conditions(x, objective, active, point, gtol):
    """Check spatially scaled KKT residuals and the unshifted free Hessian."""
    eps = jnp.finfo(x.dtype).eps
    tangent_norms = jnp.linalg.norm(objective.jacobian, axis=0)
    safe_norms = jnp.where(tangent_norms > 0, tangent_norms, 1.0)
    residual = jnp.where(active, 0.0, jnp.abs(objective.gradient) / safe_norms)
    tolerance = gtol * (1.0 + jnp.linalg.norm(objective.position - point))
    # Congruent tangent normalization preserves inertia, while making the
    # conditioning test independent of arbitrary parameter-axis units.
    minimum, regular_curvature = _curvature_conditions(objective, active)
    finite = (jnp.isfinite(objective.value) & jnp.all(jnp.isfinite(objective.gradient))
              & jnp.all(jnp.isfinite(objective.hessian)) & jnp.all(jnp.isfinite(x)))
    converged = finite & jnp.all(residual <= tolerance) & minimum
    regular = converged & regular_curvature
    return converged, regular


def refine(data, costs, seed, lower, upper, point, knot_vectors, degrees,
           maxiter, ftol, gtol, xtol):
    """Refine exactly one start; stagnation is separate from convergence.

    Spectral clamping avoids cancellation in H + (floor-lambda_min) I.
    Scale displacement to the feasible box BEFORE backtracking, so tiny
    endpoint layers do not produce identical clipped candidates. One common
    scale preserves descent for coupled surface directions.
    """
    width = upper - lower
    eps = jnp.finfo(seed.dtype).eps
    scales = 0.5 ** jnp.arange(20)

    def constraints(x, objective):
        corners, valid = knot_constraints(x, objective, data, point, knot_vectors, degrees, gtol)
        return active_bounds(x, objective.gradient, lower, upper) | corners, valid

    def feasible_direction(x, direction):
        safe = jnp.where(direction != 0, direction, 1.0)
        fraction = jnp.where(direction > 0, (upper - x) / safe,
                             jnp.where(direction < 0, (lower - x) / safe, jnp.inf))
        return direction * jnp.clip(jnp.min(fraction), 0.0, 1.0)

    def condition(state):
        iteration, _, done = state
        return (iteration < maxiter) & ~done

    def step(state):
        iteration, x, _ = state
        objective = data(x)
        active, valid_sides = constraints(x, objective)
        converged, _ = minimum_conditions(x, objective, active, point, gtol)
        converged &= valid_sides
        free = (~active).astype(x.dtype)
        hessian = objective.hessian * width[:, None] * width[None, :]
        gradient = objective.gradient * width * free
        scale = jnp.maximum(jnp.max(jnp.abs(hessian)),
                            jnp.sum((objective.jacobian * width)**2))
        scale = jnp.maximum(scale, jnp.finfo(x.dtype).tiny)
        system = hessian * free[:, None] * free[None, :] + jnp.diag(active * scale)
        finite = jnp.all(jnp.isfinite(system)) & jnp.all(jnp.isfinite(gradient))
        system = jnp.where(finite, system, jnp.eye(x.size))
        eigenvalues, eigenvectors = jnp.linalg.eigh(system)
        eigenvalues = jnp.maximum(eigenvalues, jnp.sqrt(eps) * scale)
        direction = -eigenvectors @ ((eigenvectors.T @ gradient) / eigenvalues)
        direction /= jnp.maximum(1.0, jnp.max(jnp.abs(direction)) / 0.25)
        direction = feasible_direction(x, direction * width)
        # A blocked Newton direction can still have feasible steepest descent.
        # This changes the direction within this solve, never its initial seed.
        magnitude = jnp.max(jnp.abs(gradient))
        descent = -0.25 * gradient / jnp.where(magnitude > 0, magnitude, 1.0)
        descent = feasible_direction(x, descent * width)
        uphill = (~jnp.all(jnp.isfinite(direction))
                  | (jnp.dot(objective.gradient, direction) >= 0))
        direction = jnp.where(uphill, descent, direction)
        candidates = jnp.clip(x[None] + scales[:, None] * direction, lower, upper)
        values = costs(candidates)
        roundoff = 8 * eps * jnp.maximum(1.0, jnp.abs(objective.value))
        adequate = jnp.isfinite(values) & (
            values <= objective.value + 1e-4 * ((candidates - x) @ objective.gradient) + roundoff
        )
        accepted = jnp.any(adequate) & finite
        index = jnp.argmax(adequate)
        updated = jnp.where(accepted & ~converged, candidates[index], x)
        spatial_step = jnp.linalg.norm(objective.jacobian @ (updated - x))
        small_step = spatial_step <= xtol * (1.0 + jnp.linalg.norm(objective.position - point))
        small_change = jnp.abs(values[index] - objective.value) <= ftol * jnp.maximum(1.0, objective.value)
        stagnant = jnp.all(updated == x) | (small_step & small_change)
        return iteration + 1, updated, converged | ~accepted | stagnant

    _, x, _ = jax.lax.while_loop(condition, step, (0, seed, jnp.array(False)))
    # Retain the best sampled point if numerical line-search roundoff worsens it.
    x = jnp.where(data(x).value > data(seed).value, seed, x)
    objective = data(x)
    active, valid_sides = constraints(x, objective)
    converged, regular = minimum_conditions(x, objective, active, point, gtol)
    return Minimum(x, converged & valid_sides, regular & valid_sides)


def checked_parameters(result):
    return eqx.error_if(result.parameters, ~result.converged,
                       "Point projection did not converge; increase maxiter, adjust tolerances or inspect the geometry")


def smooth_at_knots(x, knot_vectors, degrees):
    """The implicit equation needs C2 geometry; simple cubic knots are fine."""
    smooth = jnp.array(True)
    for axis, (knots, degree) in enumerate(zip(knot_vectors, degrees)):
        count = jnp.sum(knots == x[axis])
        interior = (x[axis] > knots[degree]) & (x[axis] < knots[-degree - 1])
        smooth &= ~(interior & (count > 0) & (degree - count < 2))
    return smooth


def implicit_tangent(x, objective, rhs, lower, upper, bound_tangent, point,
                     knot_vectors, degrees, gtol):
    """Differentiate exact stationarity, with moving active-bound rows."""
    active = active_bounds(x, objective.gradient, lower, upper)
    _, regular = minimum_conditions(x, objective, active, point, gtol)
    on_bound = (x == lower) | (x == upper)
    tangent_norms = jnp.linalg.norm(objective.jacobian, axis=0)
    tolerance = gtol * (1.0 + jnp.linalg.norm(objective.position - point))
    weak = jnp.any(on_bound & (jnp.abs(objective.gradient) <= tolerance * tangent_norms))
    regular &= ~weak & smooth_at_knots(x, knot_vectors, degrees)
    # Solve the exact equations in unit-tangent coordinates, without altering
    # the Hessian. This also avoids an arbitrary knot-domain scale dominating
    # the numerical conditioning of the implicit solve.
    norms = jnp.where(tangent_norms > 0, tangent_norms, 1.0)
    normalized = objective.hessian / norms[:, None] / norms[None, :]
    system = jnp.where(active[:, None], jnp.eye(x.size), normalized)
    # Safe linear algebra for invalid derivatives; the final mask exposes NaNs.
    system = jnp.where(regular, system, jnp.eye(x.size))
    right_hand_side = jnp.where(active, bound_tangent * norms, -rhs / norms)
    tangent = jnp.linalg.solve(system, right_hand_side) / norms
    return jnp.where(regular, 1.0, jnp.nan) * tangent
