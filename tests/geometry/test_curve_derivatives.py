"""Analytic curve derivatives, frames, and exact differential geometry."""

from math import factorial

import numpy as np
from numpy.testing import assert_allclose
import pytest


@pytest.mark.parametrize("order", [0, 1, 2, 3, 4])
def test_rational_derivatives_above_polynomial_degree(curve_backend, order):
    """A degree-one rational curve has nonzero derivatives of every order."""
    curve = curve_backend.NurbsCurve(
        control_points=np.array([[0.0, 1.0]]),
        weights=np.array([1.0, 2.0]),
    )
    samples = np.array([0.0, 0.13, 0.5, 0.87, 1.0])

    # C(u) = 2u / (1 + u); the denominator keeps higher derivatives nonzero.
    if order == 0:
        expected = 2 * samples / (1 + samples)
    else:
        expected = (
            2 * (-1) ** (order + 1) * factorial(order) / (1 + samples) ** (order + 1)
        )
    actual = curve.get_derivative(samples, order)

    assert actual.shape == (1, samples.size)
    assert_allclose(actual[0], expected, rtol=1e-12, atol=1e-12)


def test_nonplanar_curve_torsion_matches_exact_reference(curve_backend):
    """Recover the nonzero torsion of the polynomial curve (u, u**2, u**3)."""
    # These cubic Bernstein coefficients represent the three monomials exactly.
    points = np.array(
        [[0.0, 1 / 3, 2 / 3, 1.0], [0.0, 0.0, 1 / 3, 1.0], [0.0, 0.0, 0.0, 1.0]]
    )
    curve = curve_backend.NurbsCurve(points)
    samples = np.array([0.0, 0.17, 0.5, 0.83, 1.0])

    # The scalar triple product is 12 and |C' x C''|**2 is the denominator.
    expected = 12 / (36 * samples**4 + 36 * samples**2 + 4)
    assert_allclose(curve.get_torsion(samples), expected, rtol=1e-12, atol=1e-12)


def test_nurbs_curve_endpoint_curvature(curve_backend):
    """Test the NURBS curve end-point curvature property"""

    # Define the array of control points
    P = np.zeros((3, 11))
    P[:, 0] = [0.00, 0.00, 0.00]
    P[:, 1] = [0.10, 0.50, 0.00]
    P[:, 2] = [0.20, 0.00, 0.50]
    P[:, 3] = [0.30, 0.50, 1.00]
    P[:, 4] = [0.40, 0.00, 0.50]
    P[:, 5] = [0.50, 0.50, 0.00]
    P[:, 6] = [0.60, 0.00, 0.50]
    P[:, 7] = [0.70, 0.50, 1.00]
    P[:, 8] = [0.80, 0.00, 0.50]
    P[:, 9] = [0.90, 0.00, 0.00]
    P[:, 10] = [1.00, 0.00, 0.00]

    # Maximum index of the control points (counting from zero)
    n = np.shape(P)[1] - 1

    # Define the array of control point weights
    W = np.asarray([1, 2, 3, 2, 1, 2, 3, 2, 1, 2, 3])

    # Define the order of the basis polynomials
    p = 4

    # Define the knot vector (clamped spline)
    # p+1 zeros, n-p equispaced points between 0 and 1, and p+1 ones. In total r+1 points where r=n+p+1
    U = np.concatenate((np.zeros(p), np.linspace(0, 1, n - p + 2), np.ones(p)))

    # Create the NURBS curve
    myCurve = curve_backend.NurbsCurve(control_points=P, weights=W, degree=p, knots=U)

    # Get NURBS curve parameters
    p = myCurve.p
    U = myCurve.U
    P = myCurve.P
    W = myCurve.W
    n = np.shape(P)[1] - 1

    # Get the endpoint curvature numerically
    curvature_1a = myCurve.get_curvature(u=0.00)[0]
    curvature_1b = myCurve.get_curvature(u=1.00)[0]

    # Get the endpoint curvature analytically
    curvature_2a = (
        (p - 1)
        / p
        * (U[p + 1] / U[p + 2])
        * (W[2] * W[0] / W[1] ** 2)
        * np.sum(np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]) ** 2) ** (1 / 2)
        * np.sum((P[:, 1] - P[:, 0]) ** 2) ** (-3 / 2)
    )

    curvature_2b = (
        (p - 1)
        / p
        * (1 - U[n])
        / (1 - U[n - 1])
        * (W[n] * W[n - 2] / W[n - 1] ** 2)
        * np.sum(np.cross(P[:, n - 1] - P[:, n], P[:, n - 2] - P[:, n]) ** 2) ** (1 / 2)
        * np.sum((P[:, n - 1] - P[:, n]) ** 2) ** (-3 / 2)
    )

    # Check the error
    error_curvature_start = np.sqrt((curvature_1a - curvature_2a) ** 2)
    error_curvature_end = np.sqrt((curvature_1b - curvature_2b) ** 2)
    print("Start point curvature error                     :  ", error_curvature_start)
    print("End point curvature error                       :  ", error_curvature_end)
    assert error_curvature_start < 1e-10
    assert error_curvature_end < 1e-6


def test_nurbs_curve_zeroth_derivative(curve_backend):
    """Test that the zero-th derivative agrees with the function evaluation"""

    # Define the array of control points
    P = np.zeros((3, 5))
    P[:, 0] = [0.00, 0.00, 0.00]
    P[:, 1] = [0.00, 0.30, 0.05]
    P[:, 2] = [0.25, 0.30, 0.30]
    P[:, 3] = [0.50, 0.30, -0.05]
    P[:, 4] = [0.50, 0.10, 0.10]

    # Maximum index of the control points (counting from zero)
    n = np.shape(P)[1] - 1

    # Define the array of control point weights
    W = np.asarray([1, 1, 3, 1, 1])

    # Define the order of the basis polynomials
    # Linear (p = 1), Quadratic (p = 2), Cubic (p = 3), etc.
    # Set p = n (number of control points minus one) to obtain a Bezier
    p = 3

    # Define the knot vector (clamped spline)
    # p+1 zeros, n-p equispaced points between 0 and 1, and p+1 ones. In total r+1 points where r=n+p+1
    U = np.concatenate((np.zeros(p), np.linspace(0, 1, n - p + 2), np.ones(p)))

    # Create the NURBS curve
    nurbs3D = curve_backend.NurbsCurve(control_points=P, weights=W, degree=p, knots=U)

    # Define the u-parametrization
    u = np.linspace(0, 1, 101)

    # Compute the NURBS curve values
    C = nurbs3D.get_value(u)
    dC = nurbs3D.get_derivative(u, order=0)

    # Check the error
    error = np.sum((C - dC) ** 2) ** (1 / 2) / u.size
    print("The two-norm of the zeroth derivative error is  :  ", error)
    assert_allclose(dC, C, rtol=1e-12, atol=1e-12)


def test_nurbs_curve_first_derivative_cfd(curve_backend):
    """Test the first derivative against central finite differences"""

    # Define the array of control points
    P = np.zeros((3, 5))
    P[:, 0] = [0.00, 0.00, 0.00]
    P[:, 1] = [0.00, 0.30, 0.05]
    P[:, 2] = [0.25, 0.30, 0.30]
    P[:, 3] = [0.50, 0.30, -0.05]
    P[:, 4] = [0.50, 0.10, 0.10]

    # Maximum index of the control points (counting from zero)
    n = np.shape(P)[1] - 1

    # Define the array of control point weights
    W = np.asarray([1, 1, 3, 1, 1])

    # Define the order of the basis polynomials
    # Linear (p = 1), Quadratic (p = 2), Cubic (p = 3), etc.
    # Set p = n (number of control points minus one) to obtain a Bezier
    p = 3

    # Define the knot vector (clamped spline)
    # p+1 zeros, n-p equispaced points between 0 and 1, and p+1 ones. In total r+1 points where r=n+p+1
    U = np.concatenate((np.zeros(p), np.linspace(0, 1, n - p + 2), np.ones(p)))

    # Create the NURBS curve
    nurbs3D = curve_backend.NurbsCurve(control_points=P, weights=W, degree=p, knots=U)

    # Define a new u-parametrization suitable for finite differences
    h = 1e-5
    hh = h + h**2
    Nu = 1000
    u = np.linspace(
        0.00 + hh, 1.00 - hh, Nu
    )  # Make sure that the limits [0, 1] also work when making changes

    # Compute the NURBS analytic derivative
    dC_analytic = nurbs3D.get_derivative(u, order=1)

    # Compute the NURBS central finite differences derivative
    a = -1 / 2 * nurbs3D.get_value(u - h)
    b = +1 / 2 * nurbs3D.get_value(u + h)
    dC_cfd = (a + b) / h

    # Check the error
    error = np.sum((dC_analytic - dC_cfd) ** 2) ** (1 / 2) / u.size
    print("The two-norm of the first derivative error is   :  ", error)
    assert_allclose(dC_analytic, dC_cfd, rtol=1e-5, atol=1e-5)


def test_nurbs_curve_second_derivative_cfd(curve_backend):
    """Check second derivatives against central differences inside the domain."""

    # Define the array of control points
    P = np.zeros((3, 5))
    P[:, 0] = [0.00, 0.00, 0.00]
    P[:, 1] = [0.00, 0.30, 0.05]
    P[:, 2] = [0.25, 0.30, 0.30]
    P[:, 3] = [0.50, 0.30, -0.05]
    P[:, 4] = [0.50, 0.10, 0.10]

    # Maximum index of the control points (counting from zero)
    n = np.shape(P)[1] - 1

    # Define the array of control point weights
    W = np.asarray([1, 1, 3, 1, 1])

    # Define the order of the basis polynomials
    # Linear (p = 1), Quadratic (p = 2), Cubic (p = 3), etc.
    # Set p = n (number of control points minus one) to obtain a Bezier
    p = 3

    # Define the knot vector (clamped spline)
    # p+1 zeros, n-p equispaced points between 0 and 1, and p+1 ones. In total r+1 points where r=n+p+1
    U = np.concatenate((np.zeros(p), np.linspace(0, 1, n - p + 2), np.ones(p)))

    # Create the NURBS curve
    nurbs3D = curve_backend.NurbsCurve(control_points=P, weights=W, degree=p, knots=U)

    # Define a new u-parametrization suitable for finite differences
    h = 1e-5
    hh = h + h**2
    Nu = 1000
    u = np.linspace(
        0.00 + hh, 1.00 - hh, Nu
    )  # Make sure that the limits [0, 1] also work when making changes

    # Compute the NURBS analytic derivative
    dC_analytic = nurbs3D.get_derivative(u, order=2)

    # Compute the NURBS central finite differences derivative
    a = +1 * nurbs3D.get_value(u - h)
    b = -2 * nurbs3D.get_value(u)
    c = +1 * nurbs3D.get_value(u + h)
    dC_cfd = (a + b + c) / h**2

    # Check the error
    error = np.sum((dC_analytic - dC_cfd) ** 2) ** (1 / 2) / u.size
    print("The two-norm of the second derivative error is  :  ", error)
    assert_allclose(dC_analytic, dC_cfd, rtol=1e-5, atol=1e-5)


def test_nurbs_curve_first_derivative_endpoint(curve_backend):
    """Test the first derivative at the end points of the curve"""

    # Define the array of control points
    P = np.zeros((3, 5))
    P[:, 0] = [0.00, 0.00, 0.00]
    P[:, 1] = [0.00, 0.30, 0.05]
    P[:, 2] = [0.25, 0.30, 0.30]
    P[:, 3] = [0.50, 0.30, -0.05]
    P[:, 4] = [0.50, 0.10, 0.10]

    # Maximum index of the control points (counting from zero)
    n = np.shape(P)[1] - 1

    # Define the array of control point weights
    W = np.asarray([1, 2, 3, 2, 1])

    # Define the order of the basis polynomials
    # Linear (p = 1), Quadratic (p = 2), Cubic (p = 3), etc.
    # Set p = n (number of control points minus one) to obtain a Bezier
    p = 3

    # Define the knot vector (clamped spline)
    # p+1 zeros, n-p equispaced points between 0 and 1, and p+1 ones. In total r+1 points where r=n+p+1
    U = np.concatenate((np.zeros(p), np.linspace(0, 1, n - p + 2), np.ones(p)))

    # Create the NURBS curve
    nurbs3D = curve_backend.NurbsCurve(control_points=P, weights=W, degree=p, knots=U)

    # Start point
    dC_numeric_0 = nurbs3D.get_derivative(u=0.00, order=1)[:, 0]
    dC_analytic_0 = (p / U[p + 1]) * (W[1] / W[0]) * (P[:, 1] - P[:, 0])

    # End point
    dC_numeric_1 = nurbs3D.get_derivative(u=1.00, order=1)[:, 0]
    dC_analytic_1 = (p / (1 - U[n])) * (W[n - 1] / W[n]) * (P[:, n] - P[:, n - 1])

    # Check the error
    error_0 = np.sum((dC_analytic_0 - dC_numeric_0) ** 2) ** (1 / 2)
    error_1 = np.sum((dC_analytic_1 - dC_numeric_1) ** 2) ** (1 / 2)
    print("The start point first derivative error is       :  ", error_0)
    print("The end point first derivative error is         :  ", error_1)
    assert error_0 < 1e-6
    assert error_1 < 1e-6


def test_nurbs_curve_second_derivative_endpoint(curve_backend):
    """Test the second derivative at the end points of the curve"""

    # Define the array of control points
    P = np.zeros((3, 5))
    P[:, 0] = [0.00, 0.00, 0.00]
    P[:, 1] = [0.10, 0.30, 0.00]
    P[:, 2] = [0.25, 0.30, 0.30]
    P[:, 3] = [0.50, 0.30, -0.05]
    P[:, 4] = [0.50, 0.10, 0.10]

    # Maximum index of the control points (counting from zero)
    n = np.shape(P)[1] - 1

    # Define the array of control point weights
    W = np.asarray([1, 2, 3, 2, 1])

    # Define the order of the basis polynomials
    # Linear (p = 1), Quadratic (p = 2), Cubic (p = 3), etc.
    # Set p = n (number of control points minus one) to obtain a Bezier
    p = 3

    # Define the knot vector (clamped spline)
    # p+1 zeros, n-p equispaced points between 0 and 1, and p+1 ones. In total r+1 points where r=n+p+1
    U = np.concatenate((np.zeros(p), np.linspace(0, 1, n - p + 2), np.ones(p)))

    # Create the NURBS curve
    nurbs3D = curve_backend.NurbsCurve(control_points=P, weights=W, degree=p, knots=U)

    # Start point
    ddC_numeric_0 = nurbs3D.get_derivative(u=0.00, order=2)[:, 0]
    ddC_analytic_0 = p * (p - 1) / U[p + 1] * (
        1 / U[p + 2] * (W[2] / W[0]) * (P[:, 2] - P[:, 0])
        - (1 / U[p + 1] + 1 / U[p + 2]) * (W[1] / W[0]) * (P[:, 1] - P[:, 0])
    ) + 2 * (p / U[p + 1]) ** 2 * (W[1] / W[0]) * (1 - W[1] / W[0]) * (
        P[:, 1] - P[:, 0]
    )

    # End point
    ddC_numeric_1 = nurbs3D.get_derivative(u=1.00, order=2)[:, 0]
    ddC_analytic_1 = p * (p - 1) / (1 - U[n]) * (
        1 / (1 - U[n - 1]) * (W[n - 2] / W[n]) * (P[:, n - 2] - P[:, n])
        - (1 / (1 - U[n]) + 1 / (1 - U[n - 1]))
        * (W[n - 1] / W[n])
        * (P[:, n - 1] - P[:, n])
    ) + 2 * (p / (1 - U[n])) ** 2 * (W[n - 1] / W[n]) * (1 - W[n - 1] / W[n]) * (
        P[:, n - 1] - P[:, n]
    )

    # Check the error
    error_0 = np.sum((ddC_analytic_0 - ddC_numeric_0) ** 2) ** (1 / 2)
    error_1 = np.sum((ddC_analytic_1 - ddC_numeric_1) ** 2) ** (1 / 2)
    print("The start point second derivative error is      :  ", error_0)
    print("The end point second derivative error is        :  ", error_1)
    assert error_0 < 1e-6
    assert error_1 < 1e-6
