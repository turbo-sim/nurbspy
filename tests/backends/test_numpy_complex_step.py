"""Complex-step guarantees of the NumPy implementation, with or without Numba.

These tests perturb parameters, control points, and weights by a tiny imaginary
increment. JAX uses automatic differentiation and does not run these checks.
"""

import numpy as np
from numpy.testing import assert_allclose
import pytest
from scipy.interpolate import BSpline

from tests.cases import rational_curve_data, rational_surface_data


def test_basis_complex_step_matches_analytic_derivative(numpy_backend):
    """Preserve imaginary parameter perturbations through basis evaluation.

    Include both clamped endpoints and points on either side of an interior
    knot. The real part of the parameter selects the relevant polynomial span.
    """
    data = rational_curve_data()
    knots, degree = data["knots"], data["degree"]
    highest_index = data["weights"].size - 1
    parameters = np.array([0.0, 0.13, 0.41, 0.82, 1.0])
    step = 1e-30

    # The imaginary part divided by the step is the first derivative.
    perturbed = numpy_backend.compute_basis_polynomials(
        highest_index, degree, knots, parameters + 1j * step
    )
    expected = numpy_backend.compute_basis_polynomials_derivatives(
        highest_index, degree, knots, parameters, 1
    )
    assert np.iscomplexobj(perturbed)
    assert_allclose(perturbed.imag / step, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("order", [0, 1, 2])
def test_curve_parameter_complex_step(numpy_backend, order):
    """Differentiate curve values and derivatives through a complex parameter.

    Applying complex step to derivative order k must recover order k+1. This
    checks the rational quotient recurrence as well as the basis polynomials.
    """
    curve = numpy_backend.NurbsCurve(**rational_curve_data())
    parameters = np.array([0.0, 0.13, 0.41, 0.82, 1.0])
    step = 1e-30

    # Test the public value method separately from the derivative method.
    if order == 0:
        perturbed = curve.get_value(parameters + 1j * step)
    else:
        perturbed = curve.get_derivative(parameters + 1j * step, order)
    expected = curve.get_derivative(parameters, order + 1)
    assert np.iscomplexobj(perturbed)
    assert_allclose(perturbed.imag / step, expected, rtol=1e-11, atol=1e-10)


@pytest.mark.parametrize("argument", ["control_points", "weights"])
def test_curve_shape_complex_step(numpy_backend, argument):
    """Check shape sensitivities against an independent rational-spline formula.

    A control-point perturbation changes only the weighted numerator. A weight
    perturbation changes both numerator and denominator, so it tests the full
    quotient rule rather than simply preserving a complex array's dtype.
    """
    data = rational_curve_data()
    points, weights = data["control_points"], data["weights"]
    parameters = np.array([0.0, 0.17, 0.52, 0.86, 1.0])
    basis = BSpline(data["knots"], np.eye(weights.size), data["degree"])(parameters).T
    denominator = weights @ basis
    coordinates = (points * weights) @ basis / denominator

    # Use a deterministic direction that perturbs every design variable.
    direction = np.linspace(-0.4, 0.6, data[argument].size).reshape(
        data[argument].shape
    )
    if argument == "control_points":
        expected = (direction * weights) @ basis / denominator
    else:
        expected = (
            (points * direction) @ basis - coordinates * (direction @ basis)
        ) / denominator

    step = 1e-30
    data[argument] = data[argument].astype(complex) + 1j * step * direction
    perturbed = numpy_backend.NurbsCurve(**data).get_value(parameters)
    assert np.iscomplexobj(perturbed)
    assert_allclose(perturbed.imag / step, expected, rtol=1e-11, atol=1e-12)


@pytest.mark.parametrize(
    "parameter, initial_order, expected_order",
    [("u", (0, 0), (1, 0)), ("v", (0, 0), (0, 1)), ("u", (0, 1), (1, 1))],
    ids=["first-u", "first-v", "mixed-uv"],
)
def test_surface_parameter_complex_step(
    numpy_backend, parameter, initial_order, expected_order
):
    """Recover pure and mixed surface partials from a complex parameter step."""
    surface = numpy_backend.NurbsSurface(**rational_surface_data())
    u = np.array([0.0, 0.17, 0.41, 0.83, 1.0])
    v = np.array([1.0, 0.72, 0.33, 0.16, 0.0])
    step = 1e-30

    # Perturb one coordinate at a time, including boundary parameter pairs.
    perturbed_u = u + 1j * step if parameter == "u" else u
    perturbed_v = v + 1j * step if parameter == "v" else v
    if initial_order == (0, 0):
        perturbed = surface.get_value(perturbed_u, perturbed_v)
    else:
        perturbed = surface.get_derivative(perturbed_u, perturbed_v, *initial_order)
    expected = surface.get_derivative(u, v, *expected_order)
    assert np.iscomplexobj(perturbed)
    assert_allclose(perturbed.imag / step, expected, rtol=1e-11, atol=1e-10)


@pytest.mark.parametrize("argument", ["control_points", "weights"])
def test_surface_shape_complex_step(numpy_backend, argument):
    """Preserve control-net and weight sensitivities through rational surfaces.

    SciPy supplies the two independent univariate bases. Their tensor product
    gives a reference for the surface's directional shape derivative.
    """
    data = rational_surface_data()
    points, weights = data["control_points"], data["weights"]
    u = np.array([0.0, 0.17, 0.41, 0.83, 1.0])
    v = np.array([1.0, 0.72, 0.33, 0.16, 0.0])
    basis_u = BSpline(data["u_knots"], np.eye(weights.shape[0]), data["u_degree"])(u).T
    basis_v = BSpline(data["v_knots"], np.eye(weights.shape[1]), data["v_degree"])(v).T
    basis = np.einsum("ik,jk->ijk", basis_u, basis_v)
    denominator = np.einsum("ij,ijk->k", weights, basis)
    coordinates = np.einsum("dij,ijk->dk", points * weights, basis) / denominator

    # Differentiate the same rational expression with respect to its inputs.
    direction = np.linspace(-0.4, 0.6, data[argument].size).reshape(
        data[argument].shape
    )
    if argument == "control_points":
        expected = np.einsum("dij,ijk->dk", direction * weights, basis) / denominator
    else:
        numerator_derivative = np.einsum("dij,ijk->dk", points * direction, basis)
        denominator_derivative = np.einsum("ij,ijk->k", direction, basis)
        expected = (
            numerator_derivative - coordinates * denominator_derivative
        ) / denominator

    step = 1e-30
    data[argument] = data[argument].astype(complex) + 1j * step * direction
    perturbed = numpy_backend.NurbsSurface(**data).get_value(u, v)
    assert np.iscomplexobj(perturbed)
    assert_allclose(perturbed.imag / step, expected, rtol=1e-11, atol=1e-12)
