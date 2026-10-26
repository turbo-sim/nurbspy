"""Ruled JAX surfaces between compatible rational curves."""

import jax.numpy as jnp
from .nurbs_surface import NurbsSurface
from ._surface_generators import matching_curve_points

__all__ = ["NurbsSurfaceRuled"]


class NurbsSurfaceRuled:
    """Join C1 at v=0 to C2 at v=1 with linear homogeneous interpolation.

    Curves must have matching degrees, knots and control-point counts. With
    unequal weight functions, v is a rational parameter along each ruling.
    """

    def __init__(self, C1, C2):
        """Store the compatible boundary curves and construct their ruled patch."""
        self.C1, self.C2 = C1, C2
        self.make_nurbs_surface()

    def make_nurbs_surface(self):
        """Rebuild the patch, retaining dependence on both curves' points and weights."""
        second_points = matching_curve_points(self.C1, self.C2)
        points = jnp.stack((self.C1.P, second_points), axis=2)
        weights = jnp.stack((self.C1.W, self.C2.W), axis=1)
        self.NurbsSurface = NurbsSurface(
            points, weights, self.C1.p, 1, self.C1.U, jnp.array([0.0, 0.0, 1.0, 1.0])
        )
