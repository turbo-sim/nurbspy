"""JAX extrusion of a spatial NURBS curve."""

import jax.numpy as jnp
from .nurbs_surface import NurbsSurface
from ._surface_generators import unit_direction

__all__ = ["NurbsSurfaceExtruded"]


class NurbsSurfaceExtruded:
    """Construct S(u,v)=C(u)+v*d*D/|D| with differentiable geometry inputs."""

    def __init__(self, C, D, d):
        """Store a 3D curve, nonzero extrusion direction and signed distance."""
        if C.P.shape[0] != 3:
            raise ValueError("The input curve must be three-dimensional")
        self.C, self.D, self.d = C, unit_direction(D), jnp.asarray(d)
        self.make_Nurbs_surface()

    def make_Nurbs_surface(self):
        """Rebuild the extrusion; retain the original public method's capitalization."""
        points = jnp.stack((self.C.P, self.C.P + self.d * self.D[:, None]), axis=2)
        weights = jnp.stack((self.C.W, self.C.W), axis=1)
        self.NurbsSurface = NurbsSurface(
            points, weights, self.C.p, 1, self.C.U, jnp.array([0.0, 0.0, 1.0, 1.0])
        )
