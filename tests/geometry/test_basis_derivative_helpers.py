"""Numerical contracts of additional basis-derivative helpers exposed by JAX."""

import numpy as np
from numpy.testing import assert_allclose
import pytest
from scipy.interpolate import BSpline


@pytest.mark.parametrize("order", [1, 2, 3])
def test_basis_derivatives_with_nonclamped_knots(basis_backend, order):
    """Retain lower-degree right-hand neighbors at non-clamped boundaries."""
    knots = np.arange(9, dtype=float)
    degree, count = 3, 5
    # The highest derivative jumps at the active upper boundary. Compare its
    # left limit, avoiding different right/left endpoint conventions.
    samples = np.array([3., 3.2, 3.9, 4., 4.8, np.nextafter(5., 3.)])
    expected = BSpline(knots, np.eye(count), degree)(samples, nu=order).T
    actual = basis_backend.compute_basis_polynomials_derivatives(
        count-1, degree, knots, samples, order
    )
    assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("degree", [0, 2, 3])
@pytest.mark.parametrize("excess_order", [1, 4])
@pytest.mark.parametrize("sample_kind", ["scalar", "array", "complex"])
def test_basis_derivatives_above_degree_are_zero(
    basis_backend, degree, excess_order, sample_kind
):
    """Above-degree requests must not recurse into negative-degree bases.

    This exercises NumPy with its optional Numba decorator, as well as JAX.
    Repeated knots and endpoints use the piecewise derivative convention;
    distributional derivatives at jumps are not part of this API.
    """
    if sample_kind == "complex" and basis_backend.__name__.endswith(".jax"):
        pytest.skip("Complex-step sample support is specific to NumPy.")
    internal = [.4] if degree == 0 else [.4, .4]
    knots = np.r_[np.zeros(degree+1), internal, np.ones(degree+1)]
    count = len(knots)-degree-1
    samples = .3 if sample_kind == "scalar" else np.array([0., .13, .4, .68, 1.])
    if sample_kind == "complex":
        samples = samples.astype(complex)+1e-30j
    actual = basis_backend.compute_basis_polynomials_derivatives(
        count-1, degree, knots, samples, degree+excess_order
    )
    assert actual.shape == (count, np.size(samples))
    assert_allclose(actual, 0, rtol=0, atol=0)
    if sample_kind == "complex":
        assert np.iscomplexobj(actual)


@pytest.mark.parametrize("order", [0, 1, 2, 3])
def test_jax_basis_derivative_helpers_match_scipy(jax_backend, order):
    """Check every public derivative helper, including derivatives above degree."""
    knots = np.array([0.0, 0.0, 0.0, 0.4, 0.4, 1.0, 1.0, 1.0])
    samples = np.array([0.0, 0.13, 0.4, 0.68, 1.0])
    reference = BSpline(knots, np.eye(5), 2)
    expected = reference(samples, nu=order).T

    # The JAX AD helper and stacked-derivative API supplement the shared API.
    actual = jax_backend.compute_basis_polynomials_derivatives_jax(
        4, 2, knots, samples, order
    )
    assert_allclose(actual, expected, rtol=0, atol=1e-12)
    if order <= 2:
        all_derivatives = jax_backend.compute_all_basis_polynomials_derivatives(
            4, 2, knots, samples
        )
        assert all_derivatives.shape == (3, 5, samples.size)
        assert_allclose(all_derivatives[order], expected, rtol=0, atol=1e-12)


@pytest.mark.parametrize("parameter_kind", ["python-scalar", "jax-scalar", "array"])
def test_jax_basis_derivative_helpers_accept_scalar_and_array(
    jax_backend, parameter_kind
):
    """Retain a singleton sample axis for Python scalars and JAX scalar arrays."""
    import jax.numpy as jnp

    parameters = {
        "python-scalar": 0.3,
        "jax-scalar": jnp.array(0.3),
        "array": jnp.array([0.3]),
    }
    parameter = parameters[parameter_kind]
    knots = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    expected = np.array([[-1.4], [0.8], [0.6]])

    # A scalar must have the same output shape and values as a one-item batch.
    for derivative in (
        jax_backend.compute_basis_polynomials_derivatives,
        jax_backend.compute_basis_polynomials_derivatives_jax,
    ):
        result = derivative(2, 2, knots, parameter, 1)
        assert result.shape == (3, 1)
        assert_allclose(result, expected, rtol=0, atol=1e-12)
    result = jax_backend.compute_all_basis_polynomials_derivatives(
        2, 2, knots, parameter
    )
    assert result.shape == (3, 3, 1)
    assert_allclose(result[1], expected, rtol=0, atol=1e-12)
