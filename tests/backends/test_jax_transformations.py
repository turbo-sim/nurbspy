"""JAX compilation, batching, and traced arguments across the public API."""

from importlib.util import find_spec

import numpy as np
from numpy.testing import assert_allclose
import pytest
from scipy.interpolate import BSpline

if find_spec("jax") is None:
    pytest.skip(
        "Install nurbspy[jax] to run JAX transformation tests.", allow_module_level=True
    )

import jax
import jax.numpy as jnp
import nurbspy.jax as nrb


def test_jit_vmap_over_curve_geometries(jax_rational_curve):
    """Compile a batch of distinct geometries and compare every curve with SciPy."""
    curve = jax_rational_curve
    samples = jnp.array([0.0, 0.2, 0.7, 1.0])
    points = jnp.stack([curve.P, 1.5 * curve.P + 0.2])
    weights = jnp.stack([curve.W, curve.W.at[1].set(1.2)])

    def evaluate(control_points, curve_weights):
        """Evaluate one geometry so vmap can batch control points and weights."""
        return nrb.NurbsCurve(
            control_points, curve_weights, curve.p, curve.U
        ).get_value(samples)

    actual = jax.jit(jax.vmap(evaluate))(points, weights)
    basis = BSpline(np.asarray(curve.U), np.eye(4), 2)(np.asarray(samples)).T
    expected = np.stack(
        [
            (np.asarray(p) * np.asarray(w)) @ basis / (np.asarray(w) @ basis)
            for p, w in zip(points, weights)
        ]
    )
    assert actual.shape == (2, 2, samples.size)
    assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_batch_projection_recovers_normal_offsets():
    """Compile the public batched projection API on a parabola with known closest points."""
    # C(u) = (u, u**2); short normal offsets have these unique projections.
    curve = nrb.NurbsCurve(jnp.array([[0.0, 0.5, 1.0], [0.0, 0.0, 1.0]]))
    expected = jnp.array([0.15, 0.45, 0.8])
    normals = jnp.stack([-2 * expected, jnp.ones_like(expected)])
    normals = normals / jnp.linalg.norm(normals, axis=0)
    queries = curve.get_value(expected) + 0.03 * normals
    actual = jax.jit(lambda q: curve.project_points(q))(queries)
    assert actual.shape == expected.shape
    assert_allclose(actual, expected, atol=1e-8)


def test_merged_curves_preserve_segment_geometry():
    """Compile merged-curve evaluation without losing either segment geometry."""
    first = nrb.NurbsCurve(jnp.array([[0.0, 0.5, 1.0], [0.0, 0.0, 0.0]]))
    second = nrb.NurbsCurve(jnp.array([[1.0, 1.0, 1.0], [0.0, 0.5, 1.0]]))
    merged = nrb.merge_nurbs_curves([first, second])
    samples = jnp.array([0.0, 0.25, 0.75, 1.0])
    expected = np.array([[0.0, 0.5, 1.0, 1.0], [0.0, 0.0, 0.5, 1.0]])
    assert_allclose(
        jax.jit(lambda u: merged.get_value(u))(samples), expected, atol=1e-7
    )


@pytest.mark.parametrize("order", [0, 1, 2, 3])
def test_basis_derivative_apis_match_scipy(jax_basis_case, order):
    """Compile derivative helpers with a traced order, including orders above degree."""
    knots, samples, reference = jax_basis_case
    expected = reference(np.asarray(samples), nu=order).T
    for derivative in (
        nrb.compute_basis_polynomials_derivatives,
        nrb.compute_basis_polynomials_derivatives_jax,
    ):
        # The order is traced, rather than a static Python argument.
        evaluate = jax.jit(lambda k: derivative(4, 2, knots, samples, k))
        assert_allclose(evaluate(jnp.asarray(order)), expected, atol=1e-12)
    if order <= 2:
        all_derivatives = nrb.compute_all_basis_polynomials_derivatives(
            4, 2, knots, samples
        )
        assert all_derivatives.shape == (3, 5, samples.size)
        assert_allclose(all_derivatives[order], expected, atol=1e-12)
