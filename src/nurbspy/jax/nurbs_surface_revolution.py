"""Exact rational revolution patches, with explicit topology for traced angles."""

import math

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from .nurbs_surface import NurbsSurface
from ._surface_generators import unit_direction

__all__ = ["NurbsSurfaceRevolution"]


class NurbsSurfaceRevolution:
    """Revolve a 3D curve about an axis; u follows the arc, v the generatrix.

    Angles are radians. As in NumPy, an end angle below the start wraps once
    through 2*pi. Eager calls infer one to four quadratic arcs automatically.
    When angles are traced, pass a static keyword ``n_arcs`` (1 to 4) so the
    control-net shape is fixed. Each arc must span at most pi/2. Within that
    topology, angles, curve geometry and axis geometry are differentiable.
    """

    def __init__(
        self,
        generatrix,
        axis_point,
        axis_direction,
        angle_start,
        angle_end,
        *,
        n_arcs=None
    ):
        """Store the generatrix and axis, then construct rational circular rows."""
        if generatrix.P.shape[0] != 3 or jnp.shape(axis_point) != (3,):
            raise ValueError("The curve and axis point must be three-dimensional")
        if n_arcs is None:
            if isinstance(angle_start, jax.core.Tracer) or isinstance(
                angle_end, jax.core.Tracer
            ):
                raise ValueError(
                    "Pass a static n_arcs when revolution angles are traced"
                )
            span = float(np.asarray(angle_end)) - float(np.asarray(angle_start))
            span = span + 2 * math.pi if span < 0 else span
            if not 0 <= span <= 2 * math.pi:
                raise ValueError("Revolution span must be between zero and 2*pi")
            n_arcs = max(1, math.ceil(span / (math.pi / 2)))
        if not isinstance(n_arcs, (int, np.integer)) or not 1 <= n_arcs <= 4:
            raise ValueError("n_arcs must be a static integer from one to four")
        self.C, self.S, self.T = (
            generatrix,
            jnp.asarray(axis_point),
            unit_direction(axis_direction),
        )
        self.theta_start, self.theta_end = jnp.asarray(angle_start), jnp.asarray(
            angle_end
        )
        self.n_arcs = n_arcs
        self.make_nurbs_surface()

    def make_nurbs_surface(self):
        """Build exact arc rows without normalizing zero-radius vectors on the axis."""
        span = self.theta_end - self.theta_start
        span = jnp.where(span < 0, span + 2 * jnp.pi, span)
        delta = span / self.n_arcs
        points = eqx.error_if(
            self.C.P,
            (span < 0) | (span > 2 * jnp.pi) | (delta > jnp.pi / 2 + 1e-14),
            "Revolution arcs must span at most pi/2 each",
        )
        centers = self.S[:, None] + self.T[:, None] * jnp.sum(
            self.T[:, None] * (points - self.S[:, None]), axis=0
        )
        radial = points - centers
        transverse = jnp.cross(self.T[:, None], radial, axis=0)
        angles = self.theta_start + jnp.arange(2 * self.n_arcs + 1) * (delta / 2)
        middle = jnp.arange(2 * self.n_arcs + 1) % 2 == 1
        arc_weights = jnp.where(middle, jnp.cos(delta / 2), 1.0)
        # Midpoint control vertices lie at radius/cos(delta/2). Keeping the
        # radius vector unnormalized also handles axis points and zero sweep.
        rotated = (
            jnp.cos(angles)[None, :, None] * radial[:, None, :]
            + jnp.sin(angles)[None, :, None] * transverse[:, None, :]
        )
        control_points = centers[:, None, :] + rotated / arc_weights[None, :, None]
        weights = arc_weights[:, None] * self.C.W[None, :]
        knots = jnp.concatenate(
            (
                jnp.zeros(3),
                jnp.repeat(jnp.arange(1, self.n_arcs) / self.n_arcs, 2),
                jnp.ones(3),
            )
        )
        self.NurbsSurface = NurbsSurface(
            control_points, weights, 2, self.C.p, knots, self.C.U
        )

    @staticmethod
    def project_point_to_line(S, T, P):
        """Return the orthogonal projection onto the line S+t*T."""
        S, T, P = jnp.asarray(S), jnp.asarray(T), jnp.asarray(P)
        return S + jnp.sum(T * (P - S)) / jnp.sum(T * T) * T

    @staticmethod
    def intersect_lines(P0, T0, P2, T2):
        """Intersect two nonparallel coplanar tangent lines; retain the NumPy helper API."""
        P0, T0, P2, T2 = map(jnp.asarray, (P0, T0, P2, T2))
        matrix = jnp.stack(
            (
                jnp.stack((jnp.dot(T0, T0), -jnp.dot(T2, T0))),
                jnp.stack((jnp.dot(T0, T2), -jnp.dot(T2, T2))),
            )
        )
        rhs = jnp.stack((jnp.dot(P2 - P0, T0), jnp.dot(P2 - P0, T2)))
        parameters = jnp.linalg.solve(matrix, rhs)
        return P0 + parameters[0] * T0
