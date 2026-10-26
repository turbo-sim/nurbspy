"""Endpoint derivatives and curvature checked against explicit formulas."""

import numpy as np
from numpy.testing import assert_allclose
import pytest


@pytest.mark.parametrize("degree", [2, 3, 5])
@pytest.mark.parametrize("interior", [[], [0.13, 0.47, 0.82], [0.3, 0.3, 0.8]])
@pytest.mark.parametrize("rational", [False, True])
def test_endpoint_derivatives_and_curvature(curve_backend, degree, interior, rational):
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
    curve = curve_backend.NurbsCurve(P, W, p, U)
    for endpoint, points, weights, a, b, sign in [
        (0.0, P, W, U[p + 1], U[p + 2], 1),
        (1.0, P[:, ::-1], W[::-1], 1 - U[-p - 2], 1 - U[-p - 3], -1),
    ]:
        edge = points[:, 1] - points[:, 0]
        diagonal = points[:, 2] - points[:, 0]
        alpha, beta = weights[1:3] / weights[0]
        d1 = sign * p * alpha / a * edge
        d2 = (
            p * (p - 1) / a * (beta / b * diagonal - alpha * (1 / a + 1 / b) * edge)
            + 2 * p**2 / a**2 * alpha * (1 - alpha) * edge
        )
        kappa = (
            (p - 1)
            / p
            * a
            / b
            * beta
            / alpha**2
            * np.linalg.norm(np.cross(edge, diagonal))
            / np.linalg.norm(edge) ** 3
        )
        assert_allclose(curve.get_value(endpoint)[:, 0], points[:, 0], atol=1e-12)
        assert_allclose(
            curve.get_derivative(endpoint, 1)[:, 0], d1, rtol=1e-12, atol=1e-10
        )
        assert_allclose(
            curve.get_derivative(endpoint, 2)[:, 0], d2, rtol=1e-12, atol=1e-10
        )
        assert_allclose(curve.get_curvature(endpoint), kappa, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("at_end", [False, True])
def test_radius_construction(curve_backend, at_end):
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
    U = np.array([0.0, 0.0, 0.0, 0.0, 0.2, 0.6, 1.0, 1.0, 1.0, 1.0])
    P = np.array(
        [
            [0.0, 0.0, 1.0, 2.0, 3.0, 4.0],
            [0.0, 0.0, 0.5, 1.0, 2.0, 1.0],
            [0.0, 0.0, 0.0, 0.2, 0.4, 0.0],
        ]
    )
    W = np.array([1.0, 0.8, 1.7, 1.2, 0.9, 1.4])
    if at_end:
        # Reverse to apply the same start construction to the original end.
        P, W, U = P[:, ::-1].copy(), W[::-1].copy(), 1 - U[::-1]
    tangent = np.array([0.0, 1.0, 0.0])
    factor = (p - 1) / p * U[p + 1] / U[p + 2] * W[0] * W[2] / W[1] ** 2
    distance = np.sqrt(
        rho * factor * np.linalg.norm(np.cross(tangent, P[:, 2] - P[:, 0]))
    )
    P[:, 1] = P[:, 0] + distance * tangent
    if at_end:
        P, W, U = P[:, ::-1].copy(), W[::-1].copy(), 1 - U[::-1]
    curve = curve_backend.NurbsCurve(P, W, p, U)
    assert_allclose(curve.get_curvature(float(at_end)), 1 / rho, rtol=1e-12)
