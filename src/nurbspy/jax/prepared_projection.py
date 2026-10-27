"""Explicit fixed-geometry spatial sampling for repeated surface projection."""

import equinox as eqx
import jax
import jax.numpy as jnp

from ._point_projection import check_geometry, validate_options
from ._surface_projection import sample_surface, project_surface_from_samples
from .nurbs_surface import NurbsSurface

__all__ = ["PreparedSurfaceProjection", "prepare_surface_projection"]


class PreparedSurfaceProjection(eqx.Module):
    """Immutable geometry/grid snapshot, with one local solve per query.

    Rebuild after changing P/W, knots, degrees or sampling density. Replacing
    geometry leaves alone makes cached positions stale. Query derivatives use
    the same exact implicit minimum equations as general surface projection.
    Build inside AD for geometry sensitivities; uniqueness, regularity and a
    stable active set are required. There is no global cache or hidden retry.
    """

    surface: NurbsSurface
    sample_parameters: jax.Array
    sampled_points: jax.Array
    samples_per_span: int = eqx.field(static=True, default=8)

    @eqx.filter_jit
    def project_point(self, point, maxiter=200, ftol=1e-6, gtol=1e-6, *, xtol=1e-12):
        """Return a parameter pair shaped (2,) for one query."""
        point = jnp.asarray(point).reshape(-1)
        if point.shape != (self.surface.ndim,):
            raise ValueError("The query point must match the surface dimension")
        return self.project_points(point[:, None], maxiter, ftol, gtol, xtol=xtol)[:, 0]

    @eqx.filter_jit
    def project_points(self, points, maxiter=200, ftol=1e-6, gtol=1e-6, *, xtol=1e-12):
        """Project (ndim,N) queries with the general single-start solver."""
        points = jnp.asarray(points)
        if points.ndim != 2 or points.shape[0] != self.surface.ndim:
            raise ValueError("points must have shape (ndim,N)")
        validate_options(maxiter, self.samples_per_span, ftol, gtol, xtol)
        surface = self.surface
        return project_surface_from_samples(
            surface.P, surface.W, surface.U, surface.V, points,
            self.sample_parameters, self.sampled_points, surface.p, surface.q,
            maxiter, ftol, gtol, xtol,
        )


def prepare_surface_projection(surface, *, samples_per_span=8):
    """Sample one fixed surface once for repeated query batches.

    The default interior budget is eight samples per knot span/axis, with all
    knot boundaries included. Strong positive weight ratios inform endpoint
    coverage. Preparation and general projection use the identical grid and
    one-start solver. This does not certify global closest points on folded or
    poorly resolved geometry. Include preparation when assessing first-use cost.
    """
    if not isinstance(surface, NurbsSurface) or surface.P.size == 0:
        raise ValueError("A nonempty NurbsSurface is required")
    validate_options(200, samples_per_span, 1e-6, 1e-6, 1e-12)
    P = check_geometry(surface.P, surface.W, (surface.U, surface.V), (surface.p, surface.q))
    parameters, positions = sample_surface(
        P, surface.W, surface.U, surface.V, surface.p, surface.q, samples_per_span,
    )
    return PreparedSurfaceProjection(surface=surface, sample_parameters=parameters,
                                     sampled_points=positions, samples_per_span=samples_per_span)
