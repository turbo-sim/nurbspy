"""Surface partial derivatives checked with independent mathematical formulas."""

from math import comb, factorial

import numpy as np
from numpy.testing import assert_allclose
import pytest

from tests.cases import rational_surface_data


def test_surface_numerical_helpers_with_scalar_parameters(surface_backend):
    """Exercise all public numerical helpers with scalar u and v inputs."""
    surface = surface_backend.NurbsSurface(np.arange(12, dtype=float).reshape(3, 2, 2))
    args = (surface.P, surface.p, surface.q, surface.U, surface.V, 0.23, 0.71)
    values = surface.compute_bspline_coordinates(*args)
    partials = surface.compute_bspline_derivatives(*args, 2, 1)
    rational = surface.compute_nurbs_derivatives(
        surface.P,
        surface.W,
        surface.p,
        surface.q,
        surface.U,
        surface.V,
        0.23,
        0.71,
        2,
        1,
    )
    assert partials.shape == (3, 2, 3, 1)
    assert_allclose(values, surface.get_value(0.23, 0.71))
    assert_allclose(partials[0, 0], values)
    assert_allclose(partials, rational, atol=1e-12)
    assert_allclose(partials[2], 0.0, atol=1e-12)


@pytest.mark.parametrize("order_u, order_v", [(0, 0), (3, 0), (0, 3), (3, 2)])
def test_rational_surface_derivatives_above_degree(surface_backend, order_u, order_v):
    """Differentiate 1/((1+u)*(1+2v)) beyond its degree-one tensor basis."""
    weights = np.outer([1.0, 2.0], [1.0, 3.0])
    surface = surface_backend.NurbsSurface((1 / weights)[None], weights)
    u, v = np.array([0.0, 0.23, 0.71, 1.0]), np.array([1.0, 0.61, 0.37, 0.0])
    # The weighted numerator is identically one, so the reference separates.
    expected = (
        (-1) ** (order_u + order_v)
        * factorial(order_u)
        * factorial(order_v)
        * 2**order_v
        / ((1 + u) ** (order_u + 1) * (1 + 2 * v) ** (order_v + 1))
    )
    assert_allclose(
        surface.get_derivative(u, v, order_u, order_v)[0],
        expected,
        rtol=2e-12,
        atol=2e-12,
    )


@pytest.mark.parametrize("order_u,order_v", [(1, 0), (0, 1), (1, 1), (2, 1), (3, 2)])
def test_bezier_surface_differences(surface_backend, order_u, order_v):
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
    u = np.array([0.0, 0.19, 0.63, 1.0])
    v = np.array([1.0, 0.38, 0.71, 0.0])
    n, m = 3, 2
    k, l = order_u, order_v
    diff = np.diff(np.diff(P, n=k, axis=1), n=l, axis=2)
    Bu = np.array(
        [comb(n - k, i) * u**i * (1 - u) ** (n - k - i) for i in range(n - k + 1)]
    )
    Bv = np.array(
        [comb(m - l, j) * v**j * (1 - v) ** (m - l - j) for j in range(m - l + 1)]
    )
    expected = (
        factorial(n)
        / factorial(n - k)
        * factorial(m)
        / factorial(m - l)
        * np.einsum("it,jt,dij->dt", Bu, Bv, diff)
    )
    actual = surface_backend.NurbsSurface(P).get_derivative(u, v, k, l)
    assert_allclose(actual, expected, rtol=1e-11, atol=1e-11)


def test_rational_mixed_partial(surface_backend):
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
    surface = surface_backend.NurbsSurface(P, W)
    homogeneous = surface_backend.NurbsSurface(np.concatenate([P * W, W[None]], axis=0))
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
    fd = (
        surface.get_value(u + h, v + h)
        - surface.get_value(u + h, v - h)
        - surface.get_value(u - h, v + h)
        + surface.get_value(u - h, v - h)
    ) / (4 * h * h)
    assert_allclose(suv, fd, rtol=2e-6, atol=2e-6)
    assert_allclose(suv, surface.get_derivative(u, v, 1, 1), atol=1e-12)


def test_nurbs_surface_zeroth_derivative(surface_backend):
    """The zeroth mixed derivative must reproduce surface coordinates."""

    # Get the NURBS surface object
    surface = surface_backend.NurbsSurface(**rational_surface_data())

    # Create (u,v) parametrization
    Nu, Nv = 51, 51
    u = np.linspace(0, 1, Nu)
    v = np.linspace(0, 1, Nv)
    u, v = np.meshgrid(u, v, indexing="ij")
    u, v = u.flatten(), v.flatten()

    # Compute the NURBS surface values
    S = surface.get_value(u, v)
    dS = surface.get_derivative(u, v, order_u=0, order_v=0)

    # Check the error
    assert_allclose(dS, S, rtol=0, atol=1e-12)


def test_nurbs_surface_first_derivatives_cfd(surface_backend):
    """First partial derivatives agree with central coordinate differences."""

    # Get the NURBS surface object
    surface = surface_backend.NurbsSurface(**rational_surface_data())

    # Create (u,v) parametrization
    h = 1e-5
    hh = h + h**2
    Nu, Nv = 51, 51
    u = np.linspace(0 + hh, 1 - hh, Nu)
    v = np.linspace(0 + hh, 1 - hh, Nv)
    u, v = np.meshgrid(u, v, indexing="ij")
    u, v = u.flatten(), v.flatten()

    # Compute the NURBS analytic derivative
    dSdu_analytic = surface.get_derivative(u, v, order_u=1, order_v=0)
    dSdv_analytic = surface.get_derivative(u, v, order_u=0, order_v=1)

    # Compute the NURBS central finite-difference derivative
    dSdu_cfd = (surface.get_value(u + h, v) - surface.get_value(u - h, v)) / (2 * h)
    dSdv_cfd = (surface.get_value(u, v + h) - surface.get_value(u, v - h)) / (2 * h)

    # Check the error
    assert_allclose(dSdu_analytic, dSdu_cfd, rtol=1e-6, atol=1e-6)
    assert_allclose(dSdv_analytic, dSdv_cfd, rtol=1e-6, atol=1e-6)


def test_nurbs_surface_second_derivatives_cfd(surface_backend):
    """Pure and mixed second partials agree with central coordinate differences."""

    # Get the NURBS surface object
    surface = surface_backend.NurbsSurface(**rational_surface_data())

    # Create (u,v) parametrization
    h = 1e-5
    hh = h + h**2
    Nu, Nv = 51, 51
    u = np.linspace(0 + hh, 1 - hh, Nu)
    v = np.linspace(0 + hh, 1 - hh, Nv)
    u, v = np.meshgrid(u, v, indexing="ij")
    u, v = u.flatten(), v.flatten()

    # Compute the NURBS analytic derivative
    dSdu_analytic = surface.get_derivative(u, v, order_u=2, order_v=0)
    dSdv_analytic = surface.get_derivative(u, v, order_u=0, order_v=2)
    dSduv_analytic = surface.get_derivative(u, v, order_u=1, order_v=1)

    # Compute the NURBS central finite-difference derivative
    dSdu_cfd = (
        surface.get_value(u + h, v)
        - 2 * surface.get_value(u, v)
        + surface.get_value(u - h, v)
    ) / h**2
    dSdv_cfd = (
        surface.get_value(u, v + h)
        - 2 * surface.get_value(u, v)
        + surface.get_value(u, v - h)
    ) / h**2
    dSduv_cfd = (
        surface.get_value(u + h, v + h)
        - surface.get_value(u - h, v + h)
        - surface.get_value(u + h, v - h)
        + surface.get_value(u - h, v - h)
    ) / (4 * h**2)

    # Check the error
    # Dividing coordinate differences by h squared amplifies floating-point roundoff.
    # Keep a separate absolute tolerance for these second-order checks.
    assert_allclose(dSdu_analytic, dSdu_cfd, rtol=1e-6, atol=5e-5)
    assert_allclose(dSdv_analytic, dSdv_cfd, rtol=1e-6, atol=5e-5)
    assert_allclose(dSduv_analytic, dSduv_cfd, rtol=1e-6, atol=5e-5)
