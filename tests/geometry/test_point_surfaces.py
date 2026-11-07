"""Point grids and geometric locations preserve historical parameter ordering."""
import numpy as np
import pytest

from nurbspy import Point


def data(dimension=3):
    u, v = np.meshgrid(np.arange(4.), np.arange(3.), indexing="ij")
    return np.stack((u, v, .2*u*v + .1*u**2))[:dimension]


def grid(points):
    return tuple(tuple(Point(*points[:, i, j]) for j in range(points.shape[2]))
                 for i in range(points.shape[1]))


@pytest.mark.parametrize("dimension", [2, 3])
@pytest.mark.parametrize("options", [
    {}, {"weights": [[1., 2., 1.]] * 4},
    {"u_degree": 2, "v_degree": 1},
    {"u_degree": 2, "v_degree": 1, "u_knots": [0, 0, 0, .5, 1, 1, 1], "v_knots": [0, 0, .5, 1, 1]},
    {"u_degree": 2, "v_degree": 1, "weights": [[1., 2., 1.]] * 4},
])
def test_point_grid_matches_arrays_and_partial_derivatives(surface_backend, dimension, options):
    points = data(dimension)
    model = surface_backend.NurbsSurface(grid(points), **options)
    legacy = surface_backend.NurbsSurface(points, **options)
    np.testing.assert_array_equal(model.P, points)
    u, v = np.linspace(.1, .9, 7), np.linspace(.2, .8, 7)
    np.testing.assert_allclose(model.get_value(u, v), legacy.get_value(u, v), atol=1e-12)
    for ku, kv in [(1, 0), (0, 1), (1, 1)]:
        np.testing.assert_allclose(model.get_derivative(u, v, ku, kv),
                                   legacy.get_derivative(u, v, ku, kv), atol=1e-11)


def test_numeric_grids_keep_coordinate_first_layout(surface_backend):
    points = np.arange(27.).reshape(3, 3, 3)
    for values in (points, points.tolist()):
        np.testing.assert_array_equal(surface_backend.NurbsSurface(values).P, points)
    assert surface_backend.NurbsSurface(np.zeros((5, 3, 2))).ndim == 5


@pytest.mark.parametrize("points", [
    [[Point(0, 0)], [Point(1, 1), Point(2, 2)]],
    [[Point(0, 0)], [Point(1, 1, 1)]],
    [[Point(0, 0)], [(1, 1)]], [Point(0, 0), Point(1, 1)], [],
])
def test_invalid_grids_fail_clearly(surface_backend, points):
    with pytest.raises(ValueError, match="control_points|dimension"):
        surface_backend.NurbsSurface(points)


def test_wrong_weight_grid_is_rejected(surface_backend):
    with pytest.raises(ValueError, match="weights"):
        surface_backend.NurbsSurface(grid(data()), weights=np.ones((3, 4)))


def test_individual_point_helpers_preserve_geometry(surface_backend):
    numeric = [np.array(p) for p in [(0., 0., 0.), (2., 0., 1.), (0., 3., 2.), (2., 3., 4.)]]
    points = [Point(*p) for p in numeric]
    old = surface_backend.NurbsSurfaceBilinear(*numeric).NurbsSurface
    new = surface_backend.NurbsSurfaceBilinear(*points).NurbsSurface
    np.testing.assert_allclose(new.P, old.P)
    # Preserve the historical P01=S(1,0), P10=S(0,1) helper convention.
    np.testing.assert_allclose(new.get_value(1., 0.)[:, 0], numeric[1])
    np.testing.assert_allclose(new.get_value(0., 1.)[:, 0], numeric[2])
    arc = surface_backend.CircularArc(Point(0, 0, 0), np.array([1., 0, 0]),
                                      np.array([0., 1, 0]), 1., 0., np.pi/2).NurbsCurve
    np.testing.assert_allclose(arc.get_value(0.)[:, 0], [1, 0, 0], atol=1e-12)
    curve = surface_backend.NurbsCurve([Point(1., 0., 0.), Point(1., 0., 1.)])
    kwargs = dict(generatrix=curve, axis_direction=np.array([0., 0., 1.]), angle_start=0., angle_end=np.pi/2)
    old = surface_backend.NurbsSurfaceRevolution(axis_point=np.zeros(3), **kwargs).NurbsSurface
    new = surface_backend.NurbsSurfaceRevolution(axis_point=Point(0, 0, 0), **kwargs).NurbsSurface
    np.testing.assert_allclose(new.P, old.P)


def test_complex_grid_preserves_complex_step(numpy_backend):
    def value(x):
        return numpy_backend.NurbsSurface([
            [Point(0., 0., 0.), Point(0., 1., 0.)],
            [Point(1., 0., 0.), Point(1., 1., x)],
        ]).get_value(.3, .4)[2, 0]
    assert np.imag(value(1 + 1e-30j)) / 1e-30 == pytest.approx(.12)


def test_jax_grid_and_corner_gradients(jax_backend):
    import jax
    def grid_value(x):
        return jax_backend.NurbsSurface([
            [Point(0., 0., 0.), Point(0., 1., 0.)],
            [Point(1., 0., 0.), Point(1., 1., x)],
        ]).get_value(.3, .4)[2, 0]
    def corner_value(x):
        return jax_backend.NurbsSurfaceBilinear(Point(0., 0., 0.), Point(1., 0., 0.),
            Point(0., 1., 0.), Point(1., 1., x)).NurbsSurface.get_value(.3, .4)[2, 0]
    assert jax.jit(jax.grad(grid_value))(1.) == pytest.approx(.12)
    assert jax.jit(jax.grad(corner_value))(1.) == pytest.approx(.12)


def test_integer_point_grid_keeps_fractional_weights(surface_backend):
    points = [[Point(0, 0, 0), Point(0, 1, 0)],
              [Point(1, 0, 0), Point(1, 1, 1)]]
    weights = [[1., .25], [.5, 1.]]
    surface = surface_backend.NurbsSurface(points, weights=weights, u_degree=1, v_degree=1)
    np.testing.assert_array_equal(surface.W, weights)
    reference = surface_backend.NurbsSurface(np.array([[[0., 0.], [1., 1.]],
        [[0., 1.], [0., 1.]], [[0., 0.], [0., 1.]]]), weights=weights, u_degree=1, v_degree=1)
    np.testing.assert_allclose(surface.get_value(.3, .4), reference.get_value(.3, .4))
