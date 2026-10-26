"""Explicit fixed-geometry snapshots for repeated surface point projection.

Only coarse initialization is prepared. Newton parameters change, so refinement
continues to use the general evaluator and its exact implicit derivatives.
"""

import equinox as eqx
import jax
import jax.numpy as jnp

from ._surface_projection import (
    _sample_axes,
    _sample_parameters,
    _select_seeds,
    project_surface_point_from_seeds,
)
from .nurbs_surface import NurbsSurface
from .prepared import prepare_surface

__all__ = ["PreparedSurfaceProjection", "prepare_surface_projection"]


class PreparedSurfaceProjection(eqx.Module):
    """An immutable surface snapshot with its original coarse points prepared.

    Rebuild with :func:`prepare_surface_projection` after changing control
    points, weights, knots, or degrees. Replacing geometry leaves alone would
    make the cached coarse points stale. Ordinary JAX arrays and Equinox
    modules are immutable; this object has no hidden cache or mutation.

    Query derivatives use the same implicit minimum equations as
    ``NurbsSurface.project_point_to_surface``. Geometry derivatives remain
    available when the factory is inside the differentiated function. They
    require a unique regular minimum and a stable active set; seed ties and
    basin changes are not differentiable.
    """

    surface: NurbsSurface
    sample_parameters: jax.Array
    sampled_points: jax.Array

    @eqx.filter_jit
    def project_point(self, point, maxiter=200, ftol=1e-6, gtol=1e-6):
        """Return parameters shaped (2,), using the general solver's settings."""
        point = jnp.asarray(point).reshape(-1)
        if point.shape != (self.surface.ndim,):
            raise ValueError("The query point must match the surface dimension")
        if maxiter < 0 or ftol < 0 or gtol < 0:
            raise ValueError("Iteration limit and tolerances must be nonnegative")
        seeds = _select_seeds(self.sample_parameters, self.sampled_points, point)
        surface = self.surface
        return project_surface_point_from_seeds(
            surface.P, surface.W, surface.U, surface.V, point, seeds,
            surface.p, surface.q, maxiter, ftol, gtol,
        )

    @eqx.filter_jit
    def project_points(self, points, maxiter=200, ftol=1e-6, gtol=1e-6):
        """Project points shaped (ndim,N), returning parameters shaped (2,N)."""
        points = jnp.asarray(points)
        if points.ndim != 2 or points.shape[0] != self.surface.ndim:
            raise ValueError("points must have shape (ndim,N)")
        return jax.vmap(
            lambda point: self.project_point(point, maxiter, ftol, gtol),
            in_axes=1, out_axes=1,
        )(points)


def prepare_surface_projection(surface):
    """Prepare coarse points for repeated projection onto one fixed surface.

    The seed grid and four-start selection are the same as the general solver.
    Factorized one-dimensional polynomial basis tables evaluate that grid once;
    the snapshot retains only its parameter/spatial points, plus the surface.
    Include preparation in first-use timing. Across query batches this avoids
    rebuilding the initial spatial grid; no promise is made that it accelerates
    an individual Newton iteration or that XLA has not hoisted common sampling
    work already in a general batched call.
    """
    if not isinstance(surface, NurbsSurface) or surface.P.size == 0:
        raise ValueError("A nonempty NurbsSurface is required")
    u, v = _sample_axes(surface.U, surface.V)
    basis = prepare_surface(surface, u, v, max_order_u=0, max_order_v=0, grid=True)
    sampled_points = basis.get_value(surface.P, surface.W).reshape(surface.ndim, -1)
    return PreparedSurfaceProjection(
        surface=surface,
        sample_parameters=_sample_parameters(surface.U, surface.V),
        sampled_points=sampled_points,
    )
