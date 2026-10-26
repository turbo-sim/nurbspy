"""Bilinear JAX surface construction."""

import jax.numpy as jnp
from .nurbs_surface import NurbsSurface

__all__ = ["NurbsSurfaceBilinear"]


class NurbsSurfaceBilinear:
    """Build a patch from four 3D corners, exposed through ``NurbsSurface``.

    Preserve the original ordering: P00=S(0,0), P01=S(1,0), P10=S(0,1),
    P11=S(1,1). All corner coordinates remain differentiable.
    """

    def __init__(self, P00, P01, P10, P11):
        """Store the corners and construct the degree-one patch."""
        self.P00, self.P01, self.P10, self.P11 = map(jnp.asarray, (P00, P01, P10, P11))
        self.ndim = 3
        if any(
            point.shape != (3,) for point in (self.P00, self.P01, self.P10, self.P11)
        ):
            raise ValueError("The input points must be three-dimensional")
        self.make_nurbs_surface()

    def make_nurbs_surface(self):
        """Rebuild the bilinear patch from the stored corner coordinates."""
        points = jnp.stack(
            (
                jnp.stack((self.P00, self.P01), axis=1),
                jnp.stack((self.P10, self.P11), axis=1),
            ),
            axis=2,
        )
        self.NurbsSurface = NurbsSurface(points)
