"""Check the mathematical formulas and examples used in the documentation.

These checks were added when the theory notes were transcribed into
``docs/source/theory``. They exercise the documented formulas numerically to
catch transcription mistakes in knot indices, weight factors, derivative
scaling, and endpoint signs. The continuity check also runs the published demo
to ensure that its construction has the geometric properties described.

Fixed random seeds make each case reproducible when investigating a failure.
Run this file directly to execute the checks without invoking pytest, or call
an individual function with the parameters you want to inspect. The calls at
the bottom cover the same cases as pytest and can be stepped through in a
debugger. Plot display is suppressed only while checking the continuity demo.
"""
from math import comb, factorial
from pathlib import Path
import runpy
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
from numpy.testing import assert_allclose
import pytest
from scipy.interpolate import BSpline

import nurbspy as nrb


@pytest.mark.parametrize("degree", [2, 3, 5])
@pytest.mark.parametrize("interior", [[], [0.13, 0.47, 0.82], [0.3, 0.3, 0.8]])
@pytest.mark.parametrize("rational", [False, True])
def test_documented_endpoint_derivatives_and_curvature(degree, interior, rational):
    """Verify the documented position, derivatives, and curvature at each end.

    Construct clamped curves of several degrees with unit or unequal weights.
    Empty interior knots give Bezier curves; the other cases exercise
    nonuniform spacing and repeated knots. Evaluate the endpoint formulas
    directly from the first three control points, weights, and knot spans,
    then compare them with the library's curve evaluation methods.

    Reversing the data checks the end-point formula as well as the start-point
    formula, including the first derivative's sign. These cases guard against
    accidentally documenting a formula valid only for polynomial Bezier curves.
    """
    rng = np.random.default_rng(42)
    p = degree
    U = np.r_[np.zeros(p + 1), interior, np.ones(p + 1)]
    P = rng.normal(size=(3, len(U) - p - 1))
    W = rng.uniform(0.4, 2.0, P.shape[1]) if rational else np.ones(P.shape[1])
    curve = nrb.NurbsCurve(P, W, p, U)
    for endpoint, points, weights, a, b, sign in [
        (0., P, W, U[p + 1], U[p + 2], 1),
        (1., P[:, ::-1], W[::-1], 1 - U[-p - 2], 1 - U[-p - 3], -1),
    ]:
        edge = points[:, 1] - points[:, 0]
        diagonal = points[:, 2] - points[:, 0]
        alpha, beta = weights[1:3] / weights[0]
        d1 = sign * p * alpha / a * edge
        d2 = (p * (p - 1) / a * (beta / b * diagonal - alpha * (1 / a + 1 / b) * edge)
              + 2 * p**2 / a**2 * alpha * (1 - alpha) * edge)
        kappa = ((p - 1) / p * a / b * beta / alpha**2
                 * np.linalg.norm(np.cross(edge, diagonal)) / np.linalg.norm(edge)**3)
        assert_allclose(curve.get_value(endpoint)[:, 0], points[:, 0], atol=1e-12)
        assert_allclose(curve.get_derivative(endpoint, 1)[:, 0], d1, rtol=1e-12, atol=1e-10)
        assert_allclose(curve.get_derivative(endpoint, 2)[:, 0], d2, rtol=1e-12, atol=1e-10)
        assert_allclose(curve.get_curvature(endpoint), kappa, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("degree", [2, 3, 5])
def test_documented_derivative_control_points(degree):
    """Verify the recursive control-point formula for B-spline derivatives.

    Build each derivative curve by differencing the previous control points,
    applying the documented degree and knot-span factors, and trimming the
    knot vector. Compare its values with SciPy's derivative of the original
    spline, which provides an independent implementation of the calculation.

    Exercise every derivative order through the curve degree, both with and
    without interior knots. This checks the changing indices and knot-vector
    lengths that are easy to miscopy in the general higher-derivative formula.
    """
    rng = np.random.default_rng(7)
    for interior in [[], [0.17, 0.54, 0.91]]:
        p = degree
        U = np.r_[np.zeros(p + 1), interior, np.ones(p + 1)]
        P = rng.normal(size=(3, len(U) - p - 1))
        reference = BSpline(U, P.T, p)
        points = P.copy()
        u = np.linspace(0, 1, 47)
        for k in range(1, p + 1):
            count = points.shape[1] - 1
            denominators = U[p + 1:p + 1 + count] - U[k:k + count]
            points = (p - k + 1) * np.diff(points, axis=1) / denominators
            value = BSpline(U[k:-k], points.T, p - k)(u)
            assert_allclose(value, reference.derivative(k)(u), rtol=1e-12, atol=1e-9)


@pytest.mark.parametrize("order_u,order_v", [(1, 0), (0, 1), (1, 1), (2, 1), (3, 2)])
def test_documented_bezier_surface_differences(order_u, order_v):
    """Check the forward-difference formula for Bezier surface derivatives.

    Use a patch of degree three in u and degree two in v. Compute the expected
    derivative from control-point forward differences, explicit Bernstein
    polynomials of reduced degree, and the documented factorial factors.
    Compare this construction with the library's surface derivatives at
    interior and boundary parameter pairs.

    Pure and mixed derivatives, including the highest orders in both
    directions, check the axis ordering and scaling in the tensor-product
    formula without using the library's basis evaluation as the reference.
    """
    rng = np.random.default_rng(8)
    P = rng.normal(size=(3, 4, 3))
    u = np.array([0., 0.19, 0.63, 1.])
    v = np.array([1., 0.38, 0.71, 0.])
    n, m = 3, 2
    k, l = order_u, order_v
    diff = np.diff(np.diff(P, n=k, axis=1), n=l, axis=2)
    Bu = np.array([comb(n-k, i) * u**i * (1-u)**(n-k-i) for i in range(n-k+1)])
    Bv = np.array([comb(m-l, j) * v**j * (1-v)**(m-l-j) for j in range(m-l+1)])
    expected = (factorial(n) / factorial(n-k) * factorial(m) / factorial(m-l)
                * np.einsum("it,jt,dij->dt", Bu, Bv, diff))
    actual = nrb.NurbsSurface(P).get_derivative(u, v, k, l)
    assert_allclose(actual, expected, rtol=1e-11, atol=1e-11)


def test_documented_rational_mixed_partial():
    """Verify the quotient-rule formula for a rational surface's mixed partial.

    Form a polynomial surface in homogeneous coordinates using weighted
    control points and a separate weight coordinate. Combine its derivatives
    using the documented quotient formula, then compare the result with both
    a central mixed difference of surface values and the library's analytical
    mixed derivative.

    Unequal weights ensure the weight-derivative terms contribute. The finite
    difference uses a looser tolerance because it has truncation and rounding
    errors; it supplies a check that does not depend on analytical derivatives.
    """
    rng = np.random.default_rng(9)
    P = rng.normal(size=(3, 4, 3))
    W = rng.uniform(0.5, 2.0, (4, 3))
    surface = nrb.NurbsSurface(P, W)
    homogeneous = nrb.NurbsSurface(np.concatenate([P * W, W[None]], axis=0))
    u, v = 0.37, 0.61
    value = surface.get_value(u, v)
    hw = homogeneous.get_value(u, v)
    hu = homogeneous.get_derivative(u, v, 1, 0)
    hv = homogeneous.get_derivative(u, v, 0, 1)
    huv = homogeneous.get_derivative(u, v, 1, 1)
    su = (hu[:-1] - hu[-1] * value) / hw[-1]
    sv = (hv[:-1] - hv[-1] * value) / hw[-1]
    suv = (huv[:-1] - hu[-1] * sv - hv[-1] * su - huv[-1] * value) / hw[-1]
    h = 1e-4
    fd = (surface.get_value(u+h, v+h) - surface.get_value(u+h, v-h)
          - surface.get_value(u-h, v+h) + surface.get_value(u-h, v-h)) / (4*h*h)
    assert_allclose(suv, fd, rtol=2e-6, atol=2e-6)
    assert_allclose(suv, surface.get_derivative(u, v, 1, 1), atol=1e-12)


@pytest.mark.parametrize("at_end", [False, True])
def test_documented_radius_construction(at_end):
    """Check that the documented control-point construction sets a given radius.

    Move the adjacent control point along a prescribed unit tangent using the
    documented distance formula, then verify that the endpoint curvature is
    the reciprocal of the requested radius. Unequal weights, nonuniform knot
    spans, and a radius other than one exercise all factors in the formula.

    Repeat the construction at the other end by reversing the curve data.
    This checks that the recipe applies at either endpoint and catches a
    missing weight or knot-span factor that a simpler Bezier example could hide.
    """
    p, rho = 3, 2.7
    U = np.array([0., 0., 0., 0., 0.2, 0.6, 1., 1., 1., 1.])
    P = np.array([[0., 0., 1., 2., 3., 4.],
                  [0., 0., 0.5, 1., 2., 1.],
                  [0., 0., 0., 0.2, 0.4, 0.]])
    W = np.array([1., 0.8, 1.7, 1.2, 0.9, 1.4])
    if at_end:
        # Reverse to apply the same start construction to the original end.
        P, W, U = P[:, ::-1].copy(), W[::-1].copy(), 1 - U[::-1]
    tangent = np.array([0., 1., 0.])
    factor = (p-1)/p * U[p+1]/U[p+2] * W[0]*W[2]/W[1]**2
    distance = np.sqrt(rho * factor * np.linalg.norm(np.cross(tangent, P[:, 2]-P[:, 0])))
    P[:, 1] = P[:, 0] + distance * tangent
    if at_end:
        P, W, U = P[:, ::-1].copy(), W[::-1].copy(), 1 - U[::-1]
    curve = nrb.NurbsCurve(P, W, p, U)
    assert_allclose(curve.get_curvature(float(at_end)), 1/rho, rtol=1e-12)


def test_documented_curvature_continuity():
    """Verify that the published two-curve example has a G2-continuous join.

    Execute the actual documentation demo and compare the joining positions,
    unit tangents, and curvature vectors. Matching scalar curvature alone
    would miss a change in bending direction, so compare the full vectors.
    Also check that the endpoint parameter speeds differ: the example is meant
    to illustrate geometric continuity without requiring C1 parametrization.

    Suppress the demo's display call and close its figures so this check can
    run unattended. The same function can be called directly during debugging
    without needing pytest to supply a fixture.
    """
    script = Path(__file__).resolve().parents[1] / "demos/documentation/curvature_continuity.py"
    try:
        with patch.object(plt, "show", return_value=None):
            example = runpy.run_path(str(script))
        assert_allclose(example["left"].get_value(1.), example["right"].get_value(0.), atol=1e-14)
        assert_allclose(example["ta"], example["tb"], atol=1e-14)
        assert_allclose(example["ka"], example["kb"], atol=1e-14)
        assert not np.isclose(example["sa"], example["sb"])
    finally:
        plt.close("all")


# -------------------------------------------------------------------------------------------------------------------- #
# Check the functions manually
# -------------------------------------------------------------------------------------------------------------------- #
# Run this file directly, or select an individual call and set a breakpoint
# inside the function. Exceptions are left uncaught so the debugger can stop
# at the failing assertion. The loops cover every pytest parameter combination.
if __name__ == "__main__":
    print("Checking endpoint derivatives and curvature...")
    for degree in [2, 3, 5]:
        for interior in [[], [0.13, 0.47, 0.82], [0.3, 0.3, 0.8]]:
            for rational in [False, True]:
                test_documented_endpoint_derivatives_and_curvature(degree, interior, rational)

    print("Checking derivative control points...")
    for degree in [2, 3, 5]:
        test_documented_derivative_control_points(degree)

    print("Checking Bezier surface differences...")
    for order_u, order_v in [(1, 0), (0, 1), (1, 1), (2, 1), (3, 2)]:
        test_documented_bezier_surface_differences(order_u, order_v)

    print("Checking the rational mixed partial...")
    test_documented_rational_mixed_partial()

    print("Checking the endpoint radius construction...")
    for at_end in [False, True]:
        test_documented_radius_construction(at_end)

    print("Checking the curvature-continuity demo...")
    test_documented_curvature_continuity()
    print("All documentation math checks passed.")
