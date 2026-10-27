"""JAX differentiation guarantees for basis functions and curve operations."""

from importlib.util import find_spec
from math import factorial

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


def _finite_difference_gradient(function, value, step=1e-6):
    """Approximate each input sensitivity with a symmetric perturbation."""
    value = np.asarray(value, dtype=float)
    gradient = np.empty_like(value)
    for index in np.ndindex(value.shape):
        delta = np.zeros_like(value)
        delta[index] = step
        gradient[index] = (function(value + delta) - function(value - delta)) / (
            2 * step
        )
    return gradient


@pytest.mark.parametrize("mode", [jax.jacfwd, jax.jacrev], ids=["forward", "reverse"])
@pytest.mark.parametrize("order", [1, 2])
def test_curve_parameter_autodiff_matches_analytic_derivatives(
    jax_rational_curve, mode, order
):
    """Differentiate curve values in both AD modes, including the clamped endpoints."""
    curve = jax_rational_curve
    samples = jnp.array([0.0, 0.17, 0.59, 1.0])
    derivative = lambda u: curve.get_value(u)[:, 0]
    for _ in range(order):
        derivative = mode(derivative)
    actual = jax.jit(jax.vmap(derivative))(samples).T
    assert np.isfinite(actual).all()
    assert_allclose(
        actual, curve.get_derivative(samples, order), rtol=1e-11, atol=1e-11
    )


@pytest.mark.parametrize("mode", [jax.jacfwd, jax.jacrev], ids=["forward", "reverse"])
@pytest.mark.parametrize("order", [2, 3, 4])
def test_rational_autodiff_above_polynomial_degree(mode, order):
    """Recover nonzero high-order rational derivatives in either autodiff mode."""
    curve = nrb.NurbsCurve(jnp.array([[0.0, 1.0]]), jnp.array([1.0, 2.0]))
    samples = jnp.array([0.0, 0.2, 0.7, 1.0])

    # C(u) = 2u / (1 + u) remains differentiable beyond its degree-one basis.
    derivative = lambda u: curve.get_value(u)[0, 0]
    for _ in range(order):
        derivative = mode(derivative)
    actual = jax.jit(jax.vmap(derivative))(samples)
    expected = 2 * (-1) ** (order + 1) * factorial(order) / (1 + samples) ** (order + 1)
    assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_curve_shape_gradients_match_independent_finite_differences(jax_rational_curve):
    """Differentiate control points, weights, and an interior knot against SciPy."""
    curve = jax_rational_curve
    samples = jnp.array([0.13, 0.27, 0.62, 0.83])

    def loss(points, weights, interior_knot):
        """Measure squared positions with all three geometric inputs differentiable."""
        knots = curve.U.at[3].set(interior_knot)
        candidate = nrb.NurbsCurve(points, weights, degree=2, knots=knots)
        return jnp.sum(candidate.get_value(samples) ** 2)

    def reference_loss(points, weights, interior_knot):
        """Evaluate the same rational curve using SciPy's independent basis."""
        knots = np.array([0.0, 0.0, 0.0, float(interior_knot), 1.0, 1.0, 1.0])
        basis = BSpline(knots, np.eye(4), 2)(np.asarray(samples)).T
        values = (points * weights) @ basis / (weights @ basis)
        return np.sum(values**2)

    inputs = (np.asarray(curve.P), np.asarray(curve.W), np.array(0.4))
    value, gradients = jax.jit(jax.value_and_grad(loss, argnums=(0, 1, 2)))(*inputs)
    assert_allclose(value, reference_loss(*inputs), rtol=1e-12)
    for index, gradient in enumerate(gradients):

        def vary(argument):
            """Perturb only the selected geometric input for the numerical reference."""
            arguments = list(inputs)
            arguments[index] = argument
            return reference_loss(*arguments)

        expected = _finite_difference_gradient(vary, inputs[index])
        assert_allclose(gradient, expected, rtol=1e-6, atol=1e-8)


def test_curvature_gradient_matches_finite_differences(jax_rational_curve):
    """Differentiate rational curvature with an independent finite-difference reference."""
    curve = jax_rational_curve
    samples = jnp.array([0.2, 0.7])

    def loss(weights):
        """Sum curvature while exposing the rational weights to autodiff."""
        candidate = nrb.NurbsCurve(curve.P, weights, curve.p, curve.U)
        return jnp.sum(candidate.get_curvature(samples))

    def reference_loss(weights):
        """Compute rational curvature from SciPy homogeneous derivatives."""
        # Independent homogeneous B-spline reference for rational curvature.
        coefficients = np.vstack([np.asarray(curve.P) * weights, weights]).T
        spline = BSpline(np.asarray(curve.U), coefficients, curve.p)
        value, first, second = [spline(np.asarray(samples), nu=k).T for k in range(3)]
        position = value[:-1] / value[-1]
        tangent = (first[:-1] - first[-1] * position) / value[-1]
        acceleration = (
            second[:-1] - 2 * first[-1] * tangent - second[-1] * position
        ) / value[-1]
        cross = tangent[0] * acceleration[1] - tangent[1] * acceleration[0]
        return np.sum(np.abs(cross) / np.linalg.norm(tangent, axis=0) ** 3)

    gradient = jax.jit(jax.grad(loss))(curve.W)
    expected = _finite_difference_gradient(reference_loss, curve.W)
    assert_allclose(gradient, expected, rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize("bounds", [(0.0, 1.0), (0.2, 0.8), (0.4, 0.4)])
def test_arclength_quadrature_gradient(bounds):
    """Differentiate exact line length, including a zero-length integration interval."""
    origin = jnp.array([1.0, 2.0])
    endpoint = jnp.array([4.0, 6.0])

    def length(end):
        """Measure the specified parameter interval as the line endpoint varies."""
        return nrb.NurbsCurve(jnp.stack([origin, end], axis=1)).get_arclength_quad_fixed(
            *bounds
        )

    expected = np.array([0.6, 0.8]) * (bounds[1] - bounds[0])
    assert_allclose(jax.jit(jax.grad(length))(endpoint), expected, atol=1e-12)


def test_coordinate_reparametrization_and_inverse_gradient():
    """Differentiate the inverse coordinate map against the inverse-function theorem."""
    # This quadratic Bezier has x(u) = u + u**2, y(u) = u.
    curve = nrb.NurbsCurve(jnp.array([[0.0, 0.5, 2.0], [0.0, 0.5, 1.0]]))
    inverse = curve.reparametrize_by_coordinate(dim=0, tol=1e-11)
    coordinates = jnp.array([0.1, 0.8, 1.7])
    expected = (np.sqrt(1 + 4 * np.asarray(coordinates)) - 1) / 2
    assert_allclose(inverse(coordinates), expected, atol=1e-10)
    derivative = jax.jit(jax.jacrev(inverse))(coordinates)
    assert_allclose(derivative, np.diag(1 / (1 + 2 * expected)), atol=1e-9)


def test_projected_distance_gradient_with_frozen_parameters():
    """Differentiate closest-point distance with the documented frozen-parameter pattern."""
    points = jnp.array([[0.0, 0.5, 1.0], [0.0, 0.0, 1.0]])
    queries = jnp.array([[0.2, 0.6], [0.08, 0.4]])

    def loss(control_points):
        """Measure closest-point squared distance with frozen projection parameters."""
        curve = nrb.NurbsCurve(control_points)
        # Supported gradient pattern documented by project_points.
        u = jax.lax.stop_gradient(curve.project_points(queries))
        return jnp.sum((curve.get_value(u) - queries) ** 2)

    evaluate = jax.jit(loss)
    expected = _finite_difference_gradient(evaluate, points)
    assert_allclose(jax.jit(jax.grad(loss))(points), expected, rtol=1e-5, atol=1e-8)


@pytest.mark.parametrize("mode", [jax.jacfwd, jax.jacrev], ids=["forward", "reverse"])
@pytest.mark.parametrize("order", [1, 2])
def test_basis_autodiff_with_repeated_knots(jax_basis_case, mode, order):
    """Differentiate repeated-knot bases in forward and reverse mode without NaNs."""
    knots, samples, reference = jax_basis_case
    # At knots, compare the selected polynomial span's one-sided derivative.
    derivative = lambda u: nrb.compute_basis_polynomials(4, 2, knots, u)[:, 0]
    for _ in range(order):
        derivative = mode(derivative)
    result = jax.jit(jax.vmap(derivative))(samples).T
    assert_allclose(result, reference(np.asarray(samples), nu=order).T, atol=1e-12)
