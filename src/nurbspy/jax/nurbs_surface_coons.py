"""Homogeneous Coons control-net blending with JAX geometry inputs."""

import equinox as eqx
import jax.numpy as jnp
from .nurbs_surface import NurbsSurface
from ._surface_generators import matching_curve_points

__all__ = ["NurbsSurfaceCoons"]


class NurbsSurfaceCoons:
    """Blend four compatible boundaries using the NumPy homogeneous construction.

    South/north curves run west to east; west/east curves run south to north.
    Opposite curves share a spline space, and meeting points and weights must
    agree. The construction interpolates all four clamped boundary curves.
    """

    def __init__(self, C_south, C_north, C_west, C_east):
        """Store the four curves and form their homogeneous blended control net."""
        self.C_south, self.C_north = C_south, C_north
        self.C_west, self.C_east = C_west, C_east
        self.make_nurbs_surface()

    def make_nurbs_surface(self):
        """Validate complete corner agreement and rebuild the rational patch."""
        south, north, west, east = self.C_south, self.C_north, self.C_west, self.C_east
        self.P_south, self.P_west = south.P, west.P
        self.P_north = matching_curve_points(south, north)
        self.P_east = matching_curve_points(west, east)
        self.W_south, self.W_north = south.W, north.W
        self.W_west, self.W_east = west.W, east.W
        if any(curve.P.shape[0] != 3 for curve in (south, north, west, east)):
            raise ValueError("Coons boundaries must be three-dimensional")
        incompatible = jnp.array(False)
        for first, i, second, j in (
            (south, 0, west, 0),
            (south, -1, east, 0),
            (north, 0, west, -1),
            (north, -1, east, -1),
        ):
            incompatible = (
                incompatible
                | jnp.any(jnp.abs(first.P[:, i] - second.P[:, j]) > 1e-12)
                | (jnp.abs(first.W[i] - second.W[j]) > 1e-12)
            )
        self.P_south = eqx.error_if(
            self.P_south, incompatible, "Coons corner points and weights must agree"
        )
        self.P_sw, self.P_se = self.P_south[:, 0], self.P_south[:, -1]
        self.P_nw, self.P_ne = self.P_north[:, 0], self.P_north[:, -1]
        self.W_sw, self.W_se = south.W[0], south.W[-1]
        self.W_nw, self.W_ne = north.W[0], north.W[-1]

        # Blend in homogeneous coordinates so rational boundaries are retained.
        hs = jnp.concatenate((self.P_south * south.W, south.W[None]))[:, :, None]
        hn = jnp.concatenate((self.P_north * north.W, north.W[None]))[:, :, None]
        hw = jnp.concatenate((self.P_west * west.W, west.W[None]))[:, None, :]
        he = jnp.concatenate((self.P_east * east.W, east.W[None]))[:, None, :]
        u = jnp.linspace(0, 1, south.P.shape[1])[None, :, None]
        v = jnp.linspace(0, 1, west.P.shape[1])[None, None, :]
        corners = (
            (1 - u) * (1 - v) * hs[:, :1]
            + u * (1 - v) * hs[:, -1:]
            + (1 - u) * v * hn[:, :1]
            + u * v * hn[:, -1:]
        )
        homogeneous = (1 - v) * hs + v * hn + (1 - u) * hw + u * he - corners
        self.NurbsSurface = NurbsSurface(
            homogeneous[:-1] / homogeneous[-1:],
            homogeneous[-1],
            south.p,
            west.p,
            south.U,
            west.U,
        )
