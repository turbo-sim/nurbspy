"""Check documented derivations independently of the NURBSpy implementation."""

import numpy as np
from numpy.testing import assert_allclose
import pytest
from scipy.interpolate import BSpline


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
            denominators = U[p + 1 : p + 1 + count] - U[k : k + count]
            points = (p - k + 1) * np.diff(points, axis=1) / denominators
            value = BSpline(U[k:-k], points.T, p - k)(u)
            assert_allclose(value, reference.derivative(k)(u), rtol=1e-12, atol=1e-9)
