"""Shared trace-safe validation for surface construction."""

import equinox as eqx
import jax.numpy as jnp


def matching_curve_points(first, second):
    """Check matching spline spaces and return the second curve's control points."""
    if (
        first.P.shape != second.P.shape
        or first.p != second.p
        or first.U.shape != second.U.shape
    ):
        raise ValueError(
            "Boundary curves must have matching control-point shapes, degrees and knots"
        )
    return eqx.error_if(
        second.P,
        jnp.any(jnp.abs(first.U - second.U) > 1e-12),
        "Boundary curves must have the same knot values",
    )


def unit_direction(direction):
    """Normalize a nonzero 3-vector while retaining geometric input gradients."""
    direction = jnp.asarray(direction)
    if direction.shape != (3,):
        raise ValueError("The direction must be a three-dimensional vector")
    direction = eqx.error_if(
        direction, jnp.sum(direction**2) == 0, "The direction must be nonzero"
    )
    return direction / jnp.linalg.norm(direction)
