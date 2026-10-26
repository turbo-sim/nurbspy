"""Contracts for explicit reusable polynomial basis tables in the JAX backend.

Independent SciPy splines and exact geometries check the rational evaluation;
finite differences check that preparing inside AD preserves knot/parameter
sensitivities. Frozen tables intentionally accept only new points and weights.
"""

from importlib.util import find_spec
from math import comb, factorial

import numpy as np
from numpy.testing import assert_allclose
import pytest
from scipy.interpolate import BSpline

if find_spec("jax") is None:
    pytest.skip("Install nurbspy[jax] to run prepared-basis tests.", allow_module_level=True)

import jax
import jax.numpy as jnp
import nurbspy.jax as nrb


def _curve_data(rational=True):
    points = np.array([[0., .2, .65, 1.1, 1.5], [0., .7, -.2, .4, .1]])
    weights = np.array([1., .7, 1.3, .8, 1.1]) if rational else np.ones(5)
    knots = np.array([0., 0., 0., .4, .4, 1., 1., 1.])
    return points, weights, knots


def _scipy_curve_table(points, weights, knots, samples, degree, order):
    """Evaluate homogeneous SciPy splines, then the mathematical quotient rule."""
    samples = np.atleast_1d(samples)
    spline = BSpline(knots, np.vstack([points * weights, weights]).T, degree)
    homogeneous = [
        spline(samples, nu=r).T if r <= degree else np.zeros((points.shape[0] + 1, samples.size))
        for r in range(order + 1)
    ]
    result = []
    for r in range(order + 1):
        value = homogeneous[r][:-1].copy()
        for k in range(1, r + 1):
            value -= comb(r, k) * homogeneous[k][-1:] * result[r - k]
        result.append(value / homogeneous[0][-1:])
    return np.stack(result)


def _surface_data(rational=True):
    x, y = np.meshgrid(np.linspace(0., 1., 5), np.linspace(0., 1., 3), indexing="ij")
    points = np.stack([x, y, .15 * x * x + .1 * y + .2 * x * y])
    weights = 1 + .2 * x + .1 * np.cos(3 * y) if rational else np.ones_like(x)
    knots_u = np.array([0., 0., 0., .4, .4, 1., 1., 1.])
    knots_v = np.array([0., 0., 0., 1., 1., 1.])
    return points, weights, knots_u, knots_v


def _scipy_surface_table(points, weights, knots_u, knots_v, u, v, p, q, ku, kv):
    """Use independent SciPy tensor bases at paired sample locations."""
    u, v = np.atleast_1d(u), np.atleast_1d(v)
    basis_u = BSpline(knots_u, np.eye(points.shape[1]), p)
    basis_v = BSpline(knots_v, np.eye(points.shape[2]), q)
    bu = [basis_u(u, nu=r).T if r <= p else np.zeros((points.shape[1], u.size)) for r in range(ku + 1)]
    bv = [basis_v(v, nu=r).T if r <= q else np.zeros((points.shape[2], v.size)) for r in range(kv + 1)]
    homogeneous_points = np.concatenate([points * weights, weights[None]], axis=0)
    homogeneous = np.empty((ku + 1, kv + 1, points.shape[0] + 1, u.size))
    for a in range(ku + 1):
        for b in range(kv + 1):
            homogeneous[a, b] = np.einsum("dij,ik,jk->dk", homogeneous_points, bu[a], bv[b])
    result = np.zeros((ku + 1, kv + 1, points.shape[0], u.size))
    for a in range(ku + 1):
        for b in range(kv + 1):
            value = homogeneous[a, b, :-1].copy()
            for i in range(a + 1):
                for j in range(b + 1):
                    if i or j:
                        value -= comb(a, i) * comb(b, j) * homogeneous[i, j, -1:] * result[a - i, b - j]
            result[a, b] = value / homogeneous[0, 0, -1:]
    return result


@pytest.mark.parametrize("rational", [False, True])
def test_prepared_curve_reuses_tables_for_changing_geometry(rational):
    """One table supports new P/W batches, including rational orders above degree."""
    points, weights, knots = _curve_data(rational)
    samples = np.array([0., .17, .4, .78, 1.])
    prepared = nrb.prepare_curve(nrb.NurbsCurve(points, weights, 2, knots), samples, max_order=4)
    point_batch = np.stack([points, .8 * points + .15])
    weight_batch = np.stack([weights, weights[::-1]])
    evaluate = jax.jit(jax.vmap(lambda P, W: prepared.get_derivatives(P, W)))
    actual = evaluate(point_batch, weight_batch)
    expected = np.stack([
        _scipy_curve_table(P, W, knots, samples, 2, 4)
        for P, W in zip(point_batch, weight_batch)
    ])
    assert actual.shape == (2, 5, 2, samples.size)
    assert_allclose(actual, expected, rtol=3e-11, atol=2e-10)
    assert_allclose(
        prepared.get_value(point_batch[1], weight_batch[1]), expected[1, 0],
        rtol=3e-12, atol=1e-12,
    )
    assert_allclose(
        prepared.get_derivative(point_batch[1], weight_batch[1], order=3),
        expected[1, 3], rtol=3e-11, atol=2e-10,
    )
    # Omitting W means polynomial unit weights, not the factory curve's W.
    polynomial = _scipy_curve_table(points, np.ones_like(weights), knots, samples, 2, 4)
    assert_allclose(prepared.get_derivatives(points), polynomial, rtol=3e-11, atol=2e-10)


def test_prepared_curve_is_a_dynamic_pytree_and_accepts_scalar_samples():
    """Different prepared arrays reuse a compiled function without stale constants."""
    points, weights, knots = _curve_data()
    evaluate = jax.jit(lambda table, P, W: table.get_value(P, W))
    results = []
    for knot in (.35, .55):
        moved = knots.copy()
        moved[3:5] = knot
        prepared = nrb.prepare_curve(nrb.NurbsCurve(points, weights, 2, moved), .27, max_order=0)
        result = evaluate(prepared, points, weights)
        assert result.shape == (2, 1)
        assert_allclose(
            result, _scipy_curve_table(points, weights, moved, .27, 2, 0)[0],
            rtol=2e-12, atol=1e-12,
        )
        results.append(np.asarray(result))
    assert not np.allclose(results[0], results[1])


def test_prepared_curve_geometry_matches_exact_parabola():
    """Reuse a 2D factory's basis for a 3D curve with exact tangent/curvature."""
    points = np.array([[0., .5, 1.], [0., 0., 1.]])
    samples = np.array([0., .2, .7, 1.])
    prepared = nrb.prepare_curve(nrb.NurbsCurve(points), samples, max_order=2)
    embedded = np.vstack([points, np.zeros(3)])
    expected_tangent = np.stack([np.ones_like(samples), 2 * samples, np.zeros_like(samples)])
    expected_tangent /= np.sqrt(1 + 4 * samples**2)
    assert_allclose(prepared.get_tangent(embedded), expected_tangent, rtol=2e-12, atol=1e-12)
    assert_allclose(prepared.get_curvature(embedded), 2 / (1 + 4 * samples**2)**1.5, rtol=2e-12, atol=1e-12)


@pytest.mark.parametrize("rational", [False, True])
@pytest.mark.parametrize("grid", [False, True], ids=["paired", "grid"])
def test_prepared_surface_tables_match_scipy(rational, grid):
    """Repeated knots and endpoint partials agree in paired and factorized-grid layouts."""
    points, weights, knots_u, knots_v = _surface_data(rational)
    u = np.array([0., .13, .4, .73, 1.])
    v = np.array([0., .21, .4, .66, 1.]) if not grid else np.array([0., .31, 1.])
    surface = nrb.NurbsSurface(points, weights, 2, 2, knots_u, knots_v)
    prepared = nrb.prepare_surface(surface, u, v, max_order_u=2, max_order_v=1, grid=grid)
    if grid:
        uu, vv = np.meshgrid(u, v, indexing="ij")
        expected = _scipy_surface_table(points, weights, knots_u, knots_v, uu.ravel(), vv.ravel(), 2, 2, 2, 1)
        expected = expected.reshape(3, 2, 3, u.size, v.size)
    else:
        expected = _scipy_surface_table(points, weights, knots_u, knots_v, u, v, 2, 2, 2, 1)
    actual = jax.jit(lambda table, P, W: table.get_derivatives(P, W))(prepared, points, weights)
    assert actual.shape == expected.shape
    assert_allclose(actual, expected, rtol=3e-11, atol=2e-10)
    assert_allclose(prepared.get_value(points, weights), expected[0, 0], rtol=3e-12, atol=1e-12)
    assert_allclose(
        prepared.get_derivative(points, weights, order_u=2, order_v=1),
        expected[2, 1], rtol=3e-11, atol=2e-10,
    )
    # Prepared tables capture neither the spatial dimension nor the weights.
    scalar_points = points[2:]
    if grid:
        polynomial = _scipy_surface_table(
            scalar_points, np.ones_like(weights), knots_u, knots_v,
            uu.ravel(), vv.ravel(), 2, 2, 2, 1,
        )
        polynomial = polynomial.reshape(3, 2, 1, u.size, v.size)
    else:
        polynomial = _scipy_surface_table(
            scalar_points, np.ones_like(weights), knots_u, knots_v, u, v, 2, 2, 2, 1,
        )
    assert_allclose(prepared.get_derivatives(scalar_points), polynomial, rtol=3e-11, atol=2e-10)


def _exact_surface_data():
    """A separable rational bilinear parameterization of the graph z=x*y."""
    x, y = np.meshgrid([0., 1.], [0., 1.], indexing="ij")
    return np.stack([x, y, x * y]), np.outer([1., 2.], [1., 3.])


def _rational_coordinate(t, order, slope):
    """Coordinate (1+slope)*t/(1+slope*t), including exact higher derivatives."""
    if order == 0:
        return (1 + slope) * t / (1 + slope * t)
    numerator = (1 + slope) * (-1)**(order + 1) * factorial(order) * slope**(order - 1)
    return numerator / (1 + slope * t)**(order + 1)


@pytest.mark.parametrize("grid", [False, True], ids=["paired", "grid"])
def test_prepared_surface_rational_partials_above_both_degrees(grid):
    """Closed-form separated coordinates check mixed quotient orders beyond degree one."""
    points, weights = _exact_surface_data()
    u, v = np.array([0., .17, 1.]), np.array([0., .63, 1.])
    prepared = nrb.prepare_surface(
        nrb.NurbsSurface(points, weights), u, v,
        max_order_u=3, max_order_v=3, grid=grid,
    )
    uu, vv = np.meshgrid(u, v, indexing="ij") if grid else (u, v)
    expected = np.zeros((4, 4, 3, *uu.shape))
    for a in range(4):
        for b in range(4):
            xd, yd = _rational_coordinate(uu, a, 1), _rational_coordinate(vv, b, 2)
            if b == 0:
                expected[a, b, 0] = xd
            if a == 0:
                expected[a, b, 1] = yd
            expected[a, b, 2] = xd * yd
    assert_allclose(prepared.get_derivatives(points, weights), expected, rtol=5e-11, atol=3e-10)
    x, y = _rational_coordinate(uu, 0, 1), _rational_coordinate(vv, 0, 2)
    metric = 1 + x*x + y*y
    normal = np.stack([-y, -x, np.ones_like(x)]) / np.sqrt(metric)
    assert_allclose(prepared.get_normals(points, weights), normal, rtol=3e-12, atol=1e-12)
    mean, gaussian = prepared.get_curvature(points, weights)
    assert_allclose(mean, -x*y / metric**1.5, rtol=3e-12, atol=1e-12)
    assert_allclose(gaussian, -1 / metric**2, rtol=3e-12, atol=1e-12)


def test_prepared_surface_batches_new_points_and_weights():
    """One factorized grid table serves distinct rational geometries under jit(vmap)."""
    points, weights, knots_u, knots_v = _surface_data()
    u, v = np.array([.13, .72]), np.array([.21, .51, .86])
    surface = nrb.NurbsSurface(points, weights, 2, 2, knots_u, knots_v)
    prepared = nrb.prepare_surface(
        surface, u, v, max_order_u=1, max_order_v=1, grid=True,
    )
    point_batch = np.stack([points, 1.2 * points + .1])
    weight_batch = np.stack([weights, weights * (1 + .1 * np.arange(15).reshape(5, 3) / 15)])
    evaluate = jax.jit(jax.vmap(lambda P, W: prepared.get_derivatives(P, W)))
    uu, vv = np.meshgrid(u, v, indexing="ij")
    expected = np.stack([
        _scipy_surface_table(
            P, W, knots_u, knots_v, uu.ravel(), vv.ravel(), 2, 2, 1, 1,
        ).reshape(2, 2, 3, 2, 3)
        for P, W in zip(point_batch, weight_batch)
    ])
    assert_allclose(evaluate(point_batch, weight_batch), expected, rtol=3e-11, atol=2e-10)


@pytest.mark.parametrize("mode", [jax.jacfwd, jax.jacrev], ids=["forward", "reverse"])
def test_prepared_curve_point_and_weight_gradients_match_scipy(mode):
    """Reusable tables preserve P/W derivatives rather than freezing rational weights."""
    points, weights, knots = _curve_data()
    samples = np.array([.13, .27, .62, .83])
    prepared = nrb.prepare_curve(nrb.NurbsCurve(points, weights, 2, knots), samples, max_order=2)
    scale = np.array([.7, .1, .01])[:, None, None]
    def loss(P, W):
        return jnp.sum(jnp.sin(prepared.get_derivatives(P, W) * scale))

    def reference(P, W):
        return np.sum(np.sin(_scipy_curve_table(P, W, knots, samples, 2, 2) * scale))

    gradients = jax.jit(mode(loss, argnums=(0, 1)))(points, weights)
    for argument, (value, gradient) in enumerate(zip((points, weights), gradients)):
        assert np.isfinite(gradient).all()
        expected = np.empty_like(value)
        for index in np.ndindex(value.shape):
            delta = np.zeros_like(value)
            delta[index] = 1e-6
            plus, minus = [points, weights], [points, weights]
            plus[argument], minus[argument] = value + delta, value - delta
            expected[index] = (reference(*plus) - reference(*minus)) / 2e-6
        assert_allclose(gradient, expected, rtol=3e-6, atol=2e-8)


@pytest.mark.parametrize("mode", [jax.jacfwd, jax.jacrev], ids=["forward", "reverse"])
def test_prepared_surface_point_and_weight_gradients_match_scipy(mode):
    """Forward/reverse gradients of surface partials match independent directions."""
    points, weights, knots_u, knots_v = _surface_data()
    u, v = np.array([.13, .72]), np.array([.21, .81])
    surface = nrb.NurbsSurface(points, weights, 2, 2, knots_u, knots_v)
    prepared = nrb.prepare_surface(surface, u, v, max_order_u=1, max_order_v=1)
    scale = np.array([[.7, .1], [.1, .01]])[:, :, None, None]
    def loss(P, W):
        return jnp.sum(jnp.sin(prepared.get_derivatives(P, W) * scale))

    def reference(P, W):
        table = _scipy_surface_table(P, W, knots_u, knots_v, u, v, 2, 2, 1, 1)
        return np.sum(np.sin(table * scale))

    gradients = jax.jit(mode(loss, argnums=(0, 1)))(points, weights)
    for argument, (value, gradient) in enumerate(zip((points, weights), gradients)):
        direction = np.linspace(-.2, .3, value.size).reshape(value.shape)
        plus, minus = [points, weights], [points, weights]
        plus[argument], minus[argument] = value + 1e-6 * direction, value - 1e-6 * direction
        expected = (reference(*plus) - reference(*minus)) / 2e-6
        assert np.isfinite(gradient).all()
        assert_allclose(np.sum(gradient * direction), expected, rtol=3e-6, atol=2e-8)


@pytest.mark.parametrize("mode", [jax.jacfwd, jax.jacrev], ids=["forward", "reverse"])
def test_curve_preparation_inside_ad_preserves_parameter_and_tied_knot_sensitivities(mode):
    """Preparing inside the loss retains defined u/U sensitivities through safe branches."""
    points, weights, _ = _curve_data()
    samples, knot = np.array([.13, .27, .62, .83]), np.array(.4)
    scale = np.array([.7, .1, .01])[:, None, None]

    def loss(u, interior):
        U = jnp.array([0., 0., 0., interior, interior, 1., 1., 1.])
        prepared = nrb.prepare_curve(nrb.NurbsCurve(points, weights, 2, U), u, max_order=2)
        return jnp.sum(jnp.sin(prepared.get_derivatives(points, weights) * scale))

    def reference(u, interior):
        U = np.array([0., 0., 0., float(interior), float(interior), 1., 1., 1.])
        return np.sum(np.sin(_scipy_curve_table(points, weights, U, u, 2, 2) * scale))

    gradients = jax.jit(mode(loss, argnums=(0, 1)))(samples, knot)
    for argument, (value, gradient) in enumerate(zip((samples, knot), gradients)):
        expected = np.empty_like(value)
        for index in np.ndindex(value.shape):
            delta = np.zeros_like(value)
            delta[index] = 1e-6
            plus, minus = [samples, knot], [samples, knot]
            plus[argument], minus[argument] = value + delta, value - delta
            expected[index] = (reference(*plus) - reference(*minus)) / 2e-6
        assert np.isfinite(gradient).all()
        assert_allclose(gradient, expected, rtol=3e-6, atol=2e-8)


@pytest.mark.parametrize("mode", [jax.jacfwd, jax.jacrev], ids=["forward", "reverse"])
def test_surface_preparation_inside_ad_preserves_both_parameter_and_knot_directions(mode):
    """U/u and V/v stay differentiable when table construction is part of the objective."""
    x, y = np.meshgrid(np.linspace(0., 1., 4), np.linspace(0., 1., 4), indexing="ij")
    points = np.stack([x, y, .15*x*x + .2*x*y])
    weights = 1 + .2*x + .1*y
    inputs = (np.array([.13, .72]), np.array([.21, .81]), np.array(.43), np.array(.57))
    scale = np.array([[.7, .1], [.1, .01]])[:, :, None, None]

    def loss(u, v, ku, kv):
        U, V = jnp.array([0., 0., 0., ku, 1., 1., 1.]), jnp.array([0., 0., 0., kv, 1., 1., 1.])
        surface = nrb.NurbsSurface(points, weights, 2, 2, U, V)
        prepared = nrb.prepare_surface(surface, u, v, max_order_u=1, max_order_v=1)
        return jnp.sum(jnp.sin(prepared.get_derivatives(points, weights) * scale))

    def reference(u, v, ku, kv):
        U, V = np.array([0., 0., 0., float(ku), 1., 1., 1.]), np.array([0., 0., 0., float(kv), 1., 1., 1.])
        return np.sum(np.sin(_scipy_surface_table(points, weights, U, V, u, v, 2, 2, 1, 1) * scale))

    gradients = jax.jit(mode(loss, argnums=(0, 1, 2, 3)))(*inputs)
    for argument, (value, gradient) in enumerate(zip(inputs, gradients)):
        expected = np.empty_like(value)
        for index in np.ndindex(value.shape):
            delta = np.zeros_like(value)
            delta[index] = 1e-6
            plus, minus = list(inputs), list(inputs)
            plus[argument], minus[argument] = value + delta, value - delta
            expected[index] = (reference(*plus) - reference(*minus)) / 2e-6
        assert np.isfinite(gradient).all()
        assert_allclose(gradient, expected, rtol=3e-6, atol=2e-8)


def test_prepared_shapes_orders_and_empty_samples_are_rejected():
    """Explicit validation prevents accidental reuse with incompatible topology."""
    points, weights, knots = _curve_data()
    curve = nrb.NurbsCurve(points, weights, 2, knots)
    for samples in (np.zeros((2, 2)), np.array([])):
        with pytest.raises(ValueError):
            nrb.prepare_curve(curve, samples)
    with pytest.raises((TypeError, ValueError)):
        nrb.prepare_curve(curve, [.2], max_order=-1)
    prepared = nrb.prepare_curve(curve, [.2], max_order=1)
    for P, W in ((points[:, :-1], weights), (points, weights[:-1]), (points.ravel(), weights)):
        with pytest.raises(ValueError):
            prepared.get_value(P, W)
    with pytest.raises((TypeError, ValueError)):
        prepared.get_derivative(points, weights, order=2)
    spoints, sweights, U, V = _surface_data()
    surface = nrb.NurbsSurface(spoints, sweights, 2, 2, U, V)
    with pytest.raises(ValueError):
        nrb.prepare_surface(surface, [.2, .5], [.3])
    with pytest.raises(ValueError):
        nrb.prepare_surface(surface, [], [], grid=True)
    with pytest.raises((TypeError, ValueError)):
        nrb.prepare_surface(surface, [.2], [.3], max_order_v=-1)
    table = nrb.prepare_surface(surface, [.2], [.3], max_order_u=1, max_order_v=1)
    for P, W in ((spoints[:, :-1], sweights), (spoints, sweights[:, :-1]), (spoints[0], sweights)):
        with pytest.raises(ValueError):
            table.get_value(P, W)
    with pytest.raises((TypeError, ValueError)):
        table.get_derivative(spoints, sweights, order_u=0, order_v=2)
