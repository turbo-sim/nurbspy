"""Small numerical examples shared by the JAX backend-specific checks."""

import numpy as np
import pytest
from scipy.interpolate import BSpline


@pytest.fixture(scope="module")
def jax_rational_curve(jax_backend):
    """Provide a weighted quadratic curve with one interior knot and finite curvature."""
    return jax_backend.NurbsCurve(
        control_points=np.array([[0.0, 0.4, 1.1, 2.0], [0.0, 0.8, -0.2, 0.3]]),
        weights=np.array([1.0, 0.7, 1.3, 1.0]),
        degree=2,
        knots=np.array([0.0, 0.0, 0.0, 0.4, 1.0, 1.0, 1.0]),
    )


@pytest.fixture(scope="module")
def jax_basis_case(jax_backend):
    """Provide repeated knots and the independent SciPy basis used as an oracle."""
    # Repeated end and interior knots exercise the zero-denominator branches.
    import jax.numpy as jnp

    knots = jnp.array([0.0, 0.0, 0.0, 0.4, 0.4, 1.0, 1.0, 1.0])
    samples = jnp.array([0.0, 0.13, 0.4, 0.68, 1.0])
    reference = BSpline(np.asarray(knots), np.eye(5), 2)
    return knots, samples, reference
