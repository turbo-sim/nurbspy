"""Numerical and AD contracts for vectorized analytical derivative tables."""

from importlib.util import find_spec
from math import comb, factorial

import numpy as np
from numpy.testing import assert_allclose
import pytest
from scipy.interpolate import BSpline

if find_spec("jax") is None:
    pytest.skip("Install nurbspy[jax] to run these tests.", allow_module_level=True)

import jax
import jax.numpy as jnp
import nurbspy.jax as nrb


def test_float32_inputs_preserve_basis_and_curve_output_dtypes():
    """Keep historical default-float basis tables and control-point curve dtype."""
    knots = jnp.array([0., 0., 0., 1., 1., 1.], dtype=jnp.float32)
    samples = jnp.array([0.2, 0.6], dtype=jnp.float32)
    points = jnp.array([[0., 1., 2.], [0., 1., 0.]], dtype=jnp.float32)
    table = nrb.compute_all_basis_polynomials_derivatives(2, 2, knots, samples)
    values = nrb.compute_basis_polynomials(2, 2, knots, samples)
    assert values.dtype == table.dtype == jnp.zeros(()).dtype
    reference = BSpline(np.asarray(knots), np.eye(3), 2)
    expected = np.stack([reference(np.asarray(samples), nu=r).T for r in range(3)])
    assert_allclose(table, expected, rtol=3e-6, atol=1e-6)
    curve = nrb.compute_all_bspline_derivatives(points, 2, knots, samples, 3)
    assert curve.dtype == points.dtype
    assert_allclose(curve[:3], np.einsum("di,riu->rdu", points, expected), rtol=3e-6, atol=1e-6)
    assert_allclose(curve[3], 0., atol=0.)


@pytest.mark.parametrize(
    "degree,interior",
    [(0, [0.3, 0.8]), (1, [0.3, 0.3, 0.8]), (2, [0.3, 0.3, 0.8]),
     (3, [0.25, 0.55, 0.55]), (5, [0.18, 0.52, 0.52, 0.8])],
)
def test_full_analytic_basis_table_matches_scipy(degree, interior):
    """Retain every derivative, including repeated-knot and endpoint branches."""
    knots = np.r_[np.zeros(degree + 1), interior, np.ones(degree + 1)]
    count = knots.size - degree - 1
    samples = np.unique(np.r_[0.0, 0.13, interior, 0.69, 1.0])
    reference = BSpline(knots, np.eye(count), degree)
    evaluate = jax.jit(
        lambda U, u: nrb.compute_all_basis_polynomials_derivatives(
            count - 1, degree, U, u
        )
    )
    actual = evaluate(knots, samples)
    expected = np.stack([reference(samples, nu=r).T for r in range(degree + 1)])
    assert actual.shape == (degree + 1, count, samples.size)
    assert_allclose(actual, expected, rtol=3e-12, atol=2e-10)
    assert_allclose(
        actual[0], nrb.compute_basis_polynomials(count - 1, degree, knots, samples),
        rtol=3e-12, atol=1e-13,
    )
    # Keep the public derivative-order argument traceable rather than static.
    select = jax.jit(
        lambda order: nrb.compute_basis_polynomials_derivatives(
            count - 1, degree, knots, samples, order
        )
    )
    assert_allclose(select(jnp.asarray(degree)), expected[-1], rtol=3e-12, atol=2e-10)
    assert_allclose(select(jnp.asarray(degree + 1)), 0.0, atol=0.0)


@pytest.mark.parametrize("degree", [1, 2, 3])
def test_unclamped_basis_support_uses_complete_lower_degree_tables(degree):
    """Compare individual compact-support bases without SciPy extrapolation.

    Lower-degree bases have more entries than the final degree. Their right
    tail is needed when evaluating the support outside the partition-of-unity
    interval of an unclamped knot vector.
    """
    knots = np.linspace(-0.3, 1.3, degree + 7)
    count = knots.size - degree - 1
    samples = knots[:-1] + 0.37 * np.diff(knots)
    expected = []
    for order in range(degree + 1):
        rows = []
        for index in range(count):
            basis = BSpline.basis_element(
                knots[index:index + degree + 2], extrapolate=False
            )
            rows.append(np.nan_to_num(basis(samples, nu=order), nan=0.0))
        expected.append(rows)
    actual = nrb.compute_all_basis_polynomials_derivatives(
        count - 1, degree, knots, samples
    )
    assert_allclose(actual, expected, rtol=2e-12, atol=2e-11)


def _scipy_rational_derivatives(points, weights, knots, samples, degree, order):
    """Use independent SciPy homogeneous splines and the rational quotient rule."""
    coefficients = np.vstack([points * weights, weights]).T
    spline = BSpline(knots, coefficients, degree)
    homogeneous = [
        spline(samples, nu=r).T if r <= degree else np.zeros((points.shape[0] + 1, samples.size))
        for r in range(order + 1)
    ]
    result = []
    for r in range(order + 1):
        numerator = homogeneous[r][:-1].copy()
        for k in range(1, r + 1):
            numerator -= comb(r, k) * homogeneous[k][-1:] * result[r - k]
        result.append(numerator / homogeneous[0][-1:])
    return np.stack(result)


@pytest.mark.parametrize("rational", [False, True])
def test_jitted_batched_curve_derivative_tables_match_scipy(rational):
    """Batch complete derivative tables, with rational orders above degree."""
    degree = 2
    knots = np.array([0., 0., 0., 0.4, 0.4, 1., 1., 1.])
    points = np.array([[0., 0.2, 0.6, 0.9, 1.2], [0., 0.6, -0.2, 0.4, 0.1]])
    weights = np.array([1., 0.7, 1.3, 0.8, 1.1]) if rational else np.ones(5)
    samples = np.array([0., 0.17, 0.4, 0.78, 1.])
    point_batch = np.stack([points, 0.8 * points + 0.15])
    weight_batch = np.stack([weights, weights[::-1]])
    evaluate = jax.jit(jax.vmap(
        lambda P, W: nrb.compute_all_nurbs_derivatives(P, W, degree, knots, samples, 4)
    ))
    actual = evaluate(point_batch, weight_batch)
    expected = np.stack([
        _scipy_rational_derivatives(P, W, knots, samples, degree, 4)
        for P, W in zip(point_batch, weight_batch)
    ])
    assert actual.shape == (2, 5, 2, samples.size)
    assert_allclose(actual, expected, rtol=3e-11, atol=2e-10)
    if not rational:
        assert_allclose(actual[:, degree + 1:], 0.0, atol=2e-11)


def test_jitted_batch_accepts_changing_knots_and_parameter_arrays():
    """Keep basis inputs dynamic across geometries and successive JIT calls."""
    points = np.array([[0., 0.2, 0.6, 0.9, 1.2], [0., 0.6, -0.2, 0.4, 0.1]])
    weights = np.array([1., 0.7, 1.3, 0.8, 1.1])
    knot_batch = np.array([
        [0., 0., 0., 0.35, 0.35, 1., 1., 1.],
        [0., 0., 0., 0.55, 0.55, 1., 1., 1.],
    ])
    sample_batch = np.array([[0.11, 0.24, 0.63, 0.86], [0.15, 0.41, 0.72, 0.91]])
    evaluate = jax.jit(jax.vmap(
        lambda U, u: nrb.compute_all_nurbs_derivatives(points, weights, 2, U, u, 3)
    ))
    for offset in (0., 0.02):
        # Move repeated knots together and keep samples away from their jumps.
        # Reuse the compiled callable at the same shapes with different values.
        knots = knot_batch.copy()
        knots[:, 3:5] += offset
        samples = sample_batch + 0.2 * offset
        actual = evaluate(knots, samples)
        expected = np.stack([
            _scipy_rational_derivatives(points, weights, U, u, 2, 3)
            for U, u in zip(knots, samples)
        ])
        assert actual.shape == (2, 4, 2, samples.shape[1])
        assert_allclose(actual, expected, rtol=3e-11, atol=2e-10)


@pytest.mark.parametrize("mode", [jax.jacfwd, jax.jacrev], ids=["forward", "reverse"])
def test_analytic_derivative_table_shape_sensitivities_match_scipy(mode):
    """Differentiate derivative-based losses in P, W, u, and a smooth knot.

    The repeated interior knots move together, preserving their multiplicity.
    This differentiates through inactive zero-denominator branches while all
    samples remain away from knots, where the sensitivities are well defined.
    """
    points = np.array([[0., 0.4, 0.8, 1.1, 2.], [0., 0.8, -0.1, -0.2, 0.3]])
    weights = np.array([1., 0.7, 1.1, 1.3, 1.])
    samples = np.array([0.13, 0.27, 0.62, 0.83])
    inputs = (points, weights, samples, np.array(0.4))
    scale = np.array([0.7, 0.1, 0.01])[:, None, None]

    def loss(P, W, u, interior):
        knots = jnp.array([0., 0., 0., interior, interior, 1., 1., 1.])
        derivatives = nrb.compute_all_nurbs_derivatives(P, W, 2, knots, u, 2)
        return jnp.sum(jnp.sin(derivatives * scale))

    def reference(P, W, u, interior):
        knots = np.array([0., 0., 0., float(interior), float(interior), 1., 1., 1.])
        derivatives = _scipy_rational_derivatives(P, W, knots, u, 2, 2)
        return np.sum(np.sin(derivatives * scale))

    gradients = jax.jit(mode(loss, argnums=(0, 1, 2, 3)))(*inputs)
    assert_allclose(jax.jit(loss)(*inputs), reference(*inputs), rtol=2e-12, atol=1e-12)
    for argument_index, (value, gradient) in enumerate(zip(inputs, gradients)):
        assert np.isfinite(gradient).all()
        expected = np.empty_like(value)
        for element in np.ndindex(value.shape):
            delta = np.zeros_like(value)
            delta[element] = 1e-6
            plus, minus = list(inputs), list(inputs)
            plus[argument_index] = value + delta
            minus[argument_index] = value - delta
            expected[element] = (reference(*plus) - reference(*minus)) / 2e-6
        assert_allclose(gradient, expected, rtol=3e-6, atol=2e-8)


def test_batched_exact_rational_high_derivatives():
    """An exact rational line validates high orders without a shared recurrence oracle."""
    samples = jnp.array([0., 0.2, 0.7, 1.])
    points = jnp.array([[0., 1.]])
    weights = jnp.array([1., 2.])
    knots = jnp.array([0., 0., 1., 1.])
    actual = jax.jit(jax.vmap(
        lambda u: nrb.compute_all_nurbs_derivatives(points, weights, 1, knots, u, 5)
    ))(samples)
    expected = [2 * samples / (1 + samples)]
    expected.extend(
        2 * (-1) ** (r + 1) * factorial(r) / (1 + samples) ** (r + 1)
        for r in range(1, 6)
    )
    assert_allclose(actual[:, :, 0, 0], np.stack(expected, axis=1), rtol=2e-12, atol=2e-12)
