"""Arc-length correctness and differentiation; no timing or benchmark thresholds."""

from importlib.util import find_spec

import numpy as np
from numpy.testing import assert_allclose
import pytest
from scipy.integrate import quad
from scipy.interpolate import BSpline

if find_spec("jax") is None:
    pytest.skip("Install nurbspy[jax] for arc-length tests", allow_module_level=True)

import jax
import jax.numpy as jnp
import nurbspy.jax as nrb


def circle():
    return nrb.NurbsCurve(jnp.array([[1., 1., 0.], [0., 1., 1.]]),
                          jnp.array([1., np.sqrt(.5), 1.]))


def reference_length(curve, a=0., b=1.):
    """Independent homogeneous SciPy spline derivative, split at all knots."""
    spline = BSpline(np.asarray(curve.U),
                      np.vstack((np.asarray(curve.P)*np.asarray(curve.W), curve.W)).T, curve.p)
    def speed(u):
        H, dH = spline(u), spline(u, 1)
        return np.linalg.norm((dH[:-1]*H[-1]-H[:-1]*dH[-1])/H[-1]**2)
    bounds = np.unique(np.r_[a, np.asarray(curve.U)[(curve.U > a) & (curve.U < b)], b])
    return sum(quad(speed, lo, hi, epsabs=2e-12, epsrel=2e-12)[0]
               for lo, hi in zip(bounds[:-1], bounds[1:]))


@pytest.mark.parametrize("method", ["quadrature", "ode"])
def test_exact_lengths_and_interval_orientation(method):
    curve = nrb.NurbsCurve(jnp.array([[1., 4.], [2., 6.]]))
    assert_allclose(curve.get_arclength(method=method), 5., atol=1e-12)
    assert_allclose(curve.get_arclength(.8, .2, method=method), -3., atol=1e-12)
    assert_allclose(curve.get_arclength(.4, .4, method=method), 0., atol=0)
    assert_allclose(circle().get_arclength(tol=1e-10, method=method), np.pi/2, atol=1e-10)


@pytest.mark.parametrize("internal", [[.08, .3, .3, .3, .85], [.1, .10001, .10002, .7, .95]])
def test_adaptive_knots_against_independent_reference(internal):
    P = np.array([[0., .3, .6, .8, 1., 1.5, 2., 2.6, 3.],
                  [0., .2, 1., -.5, 1., -1., .5, .6, 0.]])
    curve = nrb.NurbsCurve(P, degree=3, knots=np.r_[np.zeros(4), internal, np.ones(4)])
    for a, b in [(0., 1.), (.17, .83)]:
        info = curve.get_arclength(a, b, atol=1e-10, rtol=1e-10, return_info=True)
        assert info.status == 0
        assert_allclose(info.value, reference_length(curve, a, b), rtol=1e-10, atol=1e-10)
        assert info.evaluations > 0
        assert info.intervals > 0
    mapping, total = curve.reparametrize_by_arclength(tol=1e-9)
    distances = jnp.linspace(0, total, 17)
    u = np.asarray(mapping(distances))
    assert_allclose([reference_length(curve, 0., float(x)) for x in u], distances,
                    atol=2e-7, rtol=2e-7)
    centers = jnp.array([.13, .43, .77, .91])*total
    h = 1e-5*total
    um, up = np.asarray(mapping(centers-h)), np.asarray(mapping(centers+h))
    keep = np.ones(len(um), dtype=bool)
    for knot in np.unique(internal):
        keep &= ~((um <= knot) & (up >= knot))
    velocities = (curve.get_value(up)-curve.get_value(um))/(2*h)
    assert_allclose(np.linalg.norm(velocities, axis=0)[keep], 1., atol=1e-4)


def test_shifted_domain_and_rational_endpoint_layer():
    curve = nrb.NurbsCurve(jnp.array([[0., 1.], [0., 0.]]), jnp.array([1., 1e12]),
                          degree=1, knots=jnp.array([2., 2., 5., 5.]))
    assert_allclose(curve.get_arclength(2., 5., tol=1e-9), 1., atol=1e-9)
    reversed_curve = nrb.NurbsCurve(curve.P[:, ::-1], curve.W[::-1], degree=1, knots=curve.U)
    assert_allclose(reversed_curve.get_arclength(2., 5., tol=1e-9), 1., atol=1e-9)


@pytest.mark.parametrize("adjoint", ["direct", "leibniz"])
@pytest.mark.parametrize("bounds", [(.2, .8), (.4, .4)])
def test_endpoint_and_geometry_sensitivities(adjoint, bounds):
    def length(end, a, b):
        return nrb.NurbsCurve(jnp.stack((jnp.zeros(2), end), axis=1)).get_arclength(
            a, b, tol=1e-10, adjoint=adjoint)
    end = jnp.array([3., 4.])
    gP, ga, gb = jax.grad(length, argnums=(0, 1, 2))(end, *map(jnp.asarray, bounds))
    assert_allclose(gP, (bounds[1]-bounds[0])*np.array([.6, .8]), atol=1e-10)
    assert_allclose([ga, gb], [-5., 5.], atol=1e-10)


def test_weight_gradient_and_jit_batching():
    c = circle()
    def length(weight):
        return nrb.NurbsCurve(c.P, c.W.at[1].set(weight)).get_arclength(tol=1e-10)
    h = 1e-5
    plus = nrb.NurbsCurve(c.P, c.W.at[1].add(h))
    minus = nrb.NurbsCurve(c.P, c.W.at[1].add(-h))
    independent = (reference_length(plus)-reference_length(minus))/(2*h)
    assert_allclose(jax.grad(length)(c.W[1]), independent, atol=1e-8, rtol=1e-6)
    ends = jnp.array([.2, .5, .8])
    values = jax.jit(jax.vmap(lambda b: c.get_arclength(0., b, tol=1e-9)))(ends)
    assert_allclose(values, [reference_length(c, 0., float(b)) for b in ends], atol=1e-9)


def test_prepared_tables_with_changed_geometry_and_gradients():
    c = circle()
    prepared = nrb.prepare_arclength(c, order=16)
    P, W = c.P*1.2, c.W*jnp.array([1.02, .98, 1.01])
    actual, gradient = jax.value_and_grad(prepared.value, argnums=(0, 1))(P, W)
    independent = reference_length(nrb.NurbsCurve(P, W))
    assert_allclose(actual, independent, atol=1e-10)
    assert all(np.all(np.isfinite(g)) for g in gradient)
    _, tangent = jax.jvp(lambda scale: prepared.value(P*scale, W), (jnp.array(1.),), (jnp.array(1.),))
    assert_allclose(tangent, independent, atol=1e-10)


def test_collapsed_length_and_unavailable_ode_error():
    collapsed = nrb.NurbsCurve(jnp.ones((2, 2)))
    assert collapsed.get_arclength() == 0
    assert np.all(np.isfinite(jax.grad(lambda P: nrb.NurbsCurve(P).get_arclength())(collapsed.P)))
    info = circle().get_arclength_ode(return_info=True)
    assert info.status == 0
    assert np.isnan(info.error_estimate)
    assert info.evaluations == -1


def test_failure_diagnostics_and_scalar_error():
    c = nrb.NurbsCurve(jnp.array([[0., .5, 1.], [0., 1., 0.]]), degree=1)
    info = c.get_arclength(max_intervals=1, return_info=True)
    assert info.status != 0
    with pytest.raises(Exception, match="Arc-length integration failed"):
        c.get_arclength(max_intervals=1)
    assert c.get_arclength(-.1, .8, return_info=True).status != 0
    invalid = nrb.NurbsCurve(c.P, jnp.array([1., -1., 1.]), degree=1)
    assert invalid.get_arclength(return_info=True).status != 0
    assert c.get_arclength(atol=0., rtol=0., return_info=True).status != 0
    with pytest.raises(ValueError, match="method"):
        c.get_arclength(method="unknown")
    with pytest.raises(Exception, match="in-domain limits"):
        nrb.prepare_arclength(c, -.1, .8)
    with pytest.raises(Exception, match="positive weights"):
        nrb.prepare_arclength(invalid)


def test_extreme_weights_cannot_silently_accept_an_inaccurate_length():
    c = nrb.NurbsCurve(jnp.array([[0., 1.], [0., 0.]]), jnp.array([1., 1e16]))
    info = c.get_arclength(tol=1e-9, return_info=True)
    assert int(info.status) != 0 or abs(float(info.value)-1.) <= 1e-9


def test_dense_map_anchors_shape_and_actual_unit_speed():
    c = nrb.NurbsCurve(jnp.array([[0., .2, 2.], [0., .1, .1]]), degree=1,
                       knots=jnp.array([0., 0., .3, 1., 1.]))
    mapping, total = c.reparametrize_by_arclength(tol=1e-9)
    L1, L2 = np.sqrt(.05), 1.8
    distances = jnp.array([[0., .1, L1], [L1+1e-6, 1., float(total)]])
    expected = np.where(distances <= L1, .3*np.asarray(distances)/L1,
                        .3+.7*(np.asarray(distances)-L1)/L2)
    assert_allclose(total, L1+L2, atol=1e-10)
    assert mapping(distances).shape == distances.shape
    assert_allclose(mapping(distances), expected, atol=1e-8)
    assert mapping(jnp.array(0.)) == 0
    assert mapping(total) == 1
    for distance, speed in [(.1, L1/.3), (1., L2/.7)]:
        h = 1e-5
        slope = (mapping(distance+h)-mapping(distance-h))/(2*h)
        assert_allclose(speed*slope, 1., atol=1e-7)
        assert_allclose(jax.grad(mapping)(jnp.array(distance)), 1/speed, atol=1e-7)
    with pytest.raises(Exception, match="queries must lie"):
        mapping(-.1)


def test_dense_map_roundtrip_and_geometry_sensitivity():
    base = jnp.array([[0., .5, 2.], [0., 0., 0.]])
    c = nrb.NurbsCurve(base)
    mapping, total = c.reparametrize_by_arclength(tol=1e-10)
    distances = jnp.linspace(0, total, 17)
    u = mapping(distances)
    assert_allclose(jax.jit(mapping)(distances), u, atol=1e-12)
    assert_allclose(u+u**2, distances, atol=2e-9)
    assert np.all(np.diff(u) > 0)
    def inverse_at_distance(scale):
        inverse, _ = nrb.NurbsCurve(base*scale).reparametrize_by_arclength(tol=1e-11, steps_per_span=64)
        return inverse(jnp.array(.4))
    expected = -.4/np.sqrt(1+4*.4)
    assert_allclose(jax.grad(inverse_at_distance)(jnp.array(1.)), expected, atol=1e-7)


def test_integer_rational_control_points():
    curve = nrb.NurbsCurve(jnp.array([[1, 1, 0], [0, 1, 1]]),
                          jnp.array([1., np.sqrt(.5), 1.]))
    assert_allclose(curve.get_arclength(tol=1e-10), np.pi/2, atol=1e-10)


def test_dense_map_on_shifted_active_domain():
    curve = nrb.NurbsCurve(jnp.array([[0., 3.], [0., 4.]]), degree=1,
                          knots=jnp.array([2., 2., 5., 5.]))
    mapping, total = curve.reparametrize_by_arclength()
    s = jnp.linspace(0, total, 9)
    assert_allclose(mapping(s), 2.+.6*s, atol=1e-10)


@pytest.mark.parametrize("x", [[0., 0., 0., 1.], [-1., 1., -1., 1.], [1., 1., 1., 1.]])
def test_dense_map_rejects_sampled_stationary_speed(x):
    c = nrb.NurbsCurve(jnp.array([x, [0.]*4]))
    with pytest.raises(Exception, match="positive speed"):
        c.reparametrize_by_arclength()


def test_dense_map_rejects_a_flat_knot_span():
    c = nrb.NurbsCurve(jnp.array([[0., 0., 1.], [0., 0., 0.]]), degree=1,
                       knots=jnp.array([0., 0., .4, 1., 1.]))
    with pytest.raises(Exception, match="positive speed"):
        c.reparametrize_by_arclength()
