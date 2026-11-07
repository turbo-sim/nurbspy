"""Readable Point construction preserves both numerical backends and array layout."""
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from nurbspy import Point


def test_point_coordinates_and_immutability():
    point = Point(1, 2, 3)
    assert (point.x, point.y, point.z) == (1, 2, 3)
    assert tuple(point) == (1, 2, 3)
    assert repr(point) == "Point(1, 2, 3)"
    with pytest.raises(FrozenInstanceError):
        point.x = 10
    with pytest.raises(AttributeError, match="2D"):
        _ = Point(1, 2).z
    scalar = np.array(4.)
    point = Point(scalar, 0)
    scalar[...] = 8
    assert point.x == 4


@pytest.mark.parametrize("coordinate", [[1, 2], "1", None, np.zeros(2)])
def test_point_rejects_non_scalar_coordinates(coordinate):
    with pytest.raises(TypeError, match="numeric scalars"):
        Point(coordinate, 0)


@pytest.mark.parametrize("dimension", [2, 3])
@pytest.mark.parametrize("options", [
    {}, {"weights": [1., .5, 2., 1.]}, {"degree": 2},
    {"degree": 2, "knots": [0, 0, 0, .5, 1, 1, 1]},
    {"degree": 2, "weights": [1., .5, 2., 1.], "knots": [0, 0, 0, .5, 1, 1, 1]},
])
def test_point_curve_matches_legacy_arrays(curve_backend, dimension, options):
    points = np.array([[0., 1., 2., 3.], [0., 2., -1., 0.], [0., .5, 1., 0.]])[:dimension]
    model = curve_backend.NurbsCurve(tuple(Point(*p) for p in points.T), **options)
    legacy = curve_backend.NurbsCurve(points, **options)
    u = np.linspace(.01, .99, 11)
    np.testing.assert_array_equal(model.P, points)
    np.testing.assert_allclose(model.get_value(u), legacy.get_value(u), atol=1e-12)
    np.testing.assert_allclose(model.get_derivative(u, 1), legacy.get_derivative(u, 1), atol=1e-12)


def test_numeric_lists_keep_coordinate_first_layout(curve_backend):
    points = [[0., 1., 2.], [3., 4., 5.], [6., 7., 8.]]
    for data in (points, tuple(map(tuple, points)), np.asarray(points)):
        np.testing.assert_array_equal(curve_backend.NurbsCurve(data).P, points)
    numeric = np.arange(20.).reshape(5, 4)
    assert curve_backend.NurbsCurve(numeric).ndim == 5


@pytest.mark.parametrize("points", [
    [Point(0, 0), (1, 2)], [Point(0, 0), Point(1, 2, 3)],
    [[Point(0, 0), Point(1, 2)]], [], [1, 2, 3],
])
def test_invalid_point_sequences_are_clear(curve_backend, points):
    with pytest.raises(ValueError, match="control_points|dimension"):
        curve_backend.NurbsCurve(points)


def test_complex_point_coordinates_preserve_complex_step(numpy_backend):
    def value(x):
        return numpy_backend.NurbsCurve([Point(0, 0), Point(x, 1), Point(2, 0)]).get_value(.4)[0, 0]
    assert np.imag(value(1 + 1e-30j)) / 1e-30 == pytest.approx(.48)


def test_point_input_is_copied_and_weights_remain_separate():
    from nurbspy import NurbsCurve
    points = [Point(0, 0), Point(1, 1), Point(2, 0)]
    first, second = NurbsCurve(points), NurbsCurve(points)
    first.P[0, 0] = 10
    assert second.P[0, 0] == points[0].x == 0
    assert not hasattr(points[0], "weight")


def test_jax_point_scalars_preserve_jit_and_gradients(jax_backend):
    import jax
    import jax.numpy as jnp
    def point_value(x):
        return jax_backend.NurbsCurve([Point(0., 0.), Point(x, 1.), Point(2., 0.)]).get_value(.4)[0, 0]
    def array_value(x):
        return jax_backend.NurbsCurve(jnp.array([[0., x, 2.], [0., 1., 0.]])).get_value(.4)[0, 0]
    assert jax.jit(point_value)(1.) == pytest.approx(array_value(1.))
    assert jax.jit(jax.grad(point_value))(1.) == pytest.approx(jax.grad(array_value)(1.))
