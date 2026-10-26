"""Generator boundaries and exact geometric properties shared across backends."""

import numpy as np
from numpy.testing import assert_allclose
import pytest

from tests.cases import surface_generator_case


@pytest.mark.parametrize(
    "kind", ["bilinear", "ruled", "extruded", "revolution", "coons"]
)
def test_surface_generator_matches_numpy(surface_backend, kind):
    """Compare complete rational patches at boundaries and interior parameters."""
    import nurbspy.numpy as numpy_backend

    parameters = np.array([1.2, 0.8, 0.4, 1.1])
    surface = surface_generator_case(surface_backend, kind, parameters)
    reference = surface_generator_case(numpy_backend, kind, parameters)
    u = np.array([0.0, 0.13, 0.47, 0.81, 1.0])
    v = np.array([1.0, 0.72, 0.31, 0.19, 0.0])
    assert_allclose(
        surface.get_value(u, v), reference.get_value(u, v), rtol=1e-12, atol=1e-12
    )
    assert_allclose(
        surface.get_derivative(u, v, 1, 1),
        reference.get_derivative(u, v, 1, 1),
        rtol=1e-11,
        atol=1e-11,
    )


def test_bilinear_corner_interpolation(surface_backend):
    """Check the unusual P00/P01/P10/P11 naming against the exact bilinear map."""
    corners = np.array(
        [[0.2, -0.1, 0.3], [1.3, 0.2, 0.1], [0.1, 1.2, 0.4], [1.1, 1.4, 0.8]]
    )
    patch = surface_backend.NurbsSurfaceBilinear(*corners).NurbsSurface
    u, v = np.meshgrid([0.0, 0.27, 1.0], [0.0, 0.63, 1.0], indexing="ij")
    u, v = u.ravel(), v.ravel()
    expected = corners.T @ np.array(
        [(1 - u) * (1 - v), u * (1 - v), (1 - u) * v, u * v]
    )
    assert_allclose(patch.get_value(u, v), expected, rtol=1e-12, atol=1e-12)


def test_ruled_boundaries_and_rational_rulings(surface_backend):
    """Unequal boundary weights give rational interpolation along straight rulings."""
    points = np.array([[0.0, 0.5, 1.0], [0.0, 0.2, 0.1], [0.0, 0.4, 0.0]])
    first = surface_backend.NurbsCurve(points, np.array([1.0, 0.7, 1.0]))
    second = surface_backend.NurbsCurve(
        points + np.array([0.2, 1.0, 0.3])[:, None], np.array([1.0, 1.4, 1.0])
    )
    patch = surface_backend.NurbsSurfaceRuled(first, second).NurbsSurface
    u, v = np.meshgrid([0.0, 0.31, 0.79, 1.0], [0.0, 0.43, 1.0], indexing="ij")
    u, v = u.ravel(), v.ravel()
    basis = np.array([(1 - u) ** 2, 2 * u * (1 - u), u * u])
    w1, w2 = np.asarray(first.W) @ basis, np.asarray(second.W) @ basis
    expected = ((1 - v) * w1 * first.get_value(u) + v * w2 * second.get_value(u)) / (
        (1 - v) * w1 + v * w2
    )
    assert_allclose(patch.get_value(u, v), expected, rtol=1e-12, atol=1e-12)


def test_extrusion_matches_translated_curve(surface_backend):
    """A nonunit direction is normalized and a signed distance controls extrusion."""
    curve = surface_backend.NurbsCurve(
        np.array([[0.0, 0.4, 1.0], [0.0, 0.7, 0.2], [0.0, 0.1, 0.4]]),
        np.array([1.0, 0.8, 1.0]),
    )
    direction, distance = np.array([0.3, -0.2, 2.0]), -0.7
    patch = surface_backend.NurbsSurfaceExtruded(
        curve, direction, distance
    ).NurbsSurface
    u, v = np.array([0.0, 0.31, 0.8, 1.0]), np.array([0.0, 0.42, 1.0, 0.2])
    expected = (
        curve.get_value(u)
        + distance * direction[:, None] / np.linalg.norm(direction) * v
    )
    assert_allclose(patch.get_value(u, v), expected, rtol=1e-12, atol=1e-12)


def test_coons_preserves_all_boundaries(surface_backend):
    """Homogeneous blending interpolates four compatible rational boundary curves."""
    patch = surface_generator_case(
        surface_backend, "coons", np.array([1.2, 0.8, 0.4, 1.1])
    )
    t = np.array([0.0, 0.13, 0.47, 0.82, 1.0])
    weights = np.array([1.0, 0.8, 1.0])
    south = np.array([[0.0, 0.6, 1.2], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    north = south + np.array([0.0, 1.0, 0.0])[:, None]
    west = np.array([[0.0, 0.0, 0.0], [0.0, 0.5, 1.0], [0.0, 0.4, 0.0]])
    east = np.array([[1.2, 1.2, 1.2], [0.0, 0.5, 1.0], [0.0, 1.1, 0.0]])
    for points, u, v in [
        (south, t, 0 * t),
        (north, t, 0 * t + 1),
        (west, 0 * t, t),
        (east, 0 * t + 1, t),
    ]:
        curve = surface_backend.NurbsCurve(points, weights)
        assert_allclose(
            patch.get_value(u, v), curve.get_value(t), rtol=1e-12, atol=1e-12
        )


@pytest.mark.parametrize(
    "start,end",
    [
        (0.0, np.pi / 3),
        (0.2, 0.2 + np.pi),
        (0.0, 1.5 * np.pi),
        (0.0, 2 * np.pi),
        (5.5, 0.3),
    ],
)
def test_revolution_exact_cylinder(surface_backend, start, end):
    """Revolution preserves radius/height and reaches exact rotated endpoints."""
    curve = surface_backend.NurbsCurve(np.array([[1.2, 1.2], [0.3, 0.3], [0.0, 2.0]]))
    patch = surface_backend.NurbsSurfaceRevolution(
        curve, np.zeros(3), np.array([0.0, 0.0, 2.0]), start, end
    ).NurbsSurface
    u, v = np.meshgrid(np.linspace(0, 1, 9), [0.0, 0.23, 1.0], indexing="ij")
    values = np.asarray(patch.get_value(u.ravel(), v.ravel()))
    assert_allclose(
        values[0] ** 2 + values[1] ** 2, 1.2**2 + 0.3**2, rtol=1e-12, atol=1e-12
    )
    assert_allclose(values[2], 2 * v.ravel(), rtol=1e-12, atol=1e-12)
    for parameter, angle in [(0.0, start), (1.0, end)]:
        expected = [
            1.2 * np.cos(angle) - 0.3 * np.sin(angle),
            1.2 * np.sin(angle) + 0.3 * np.cos(angle),
            0.8,
        ]
        assert_allclose(patch.get_value(parameter, 0.4)[:, 0], expected, atol=1e-12)


def test_revolution_preserves_radial_direction_with_zero_coordinate_sum(
    surface_backend,
):
    """A radial vector (1,-1,0) is not on the axis despite its zero component sum."""
    curve = surface_backend.NurbsCurve(np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]]))
    patch = surface_backend.NurbsSurfaceRevolution(
        curve, np.zeros(3), np.array([0.0, 0.0, 1.0]), 0.0, np.pi / 2
    ).NurbsSurface
    assert_allclose(patch.get_value(0.0, 0.0)[:, 0], [1.0, -1.0, 0.0], atol=1e-12)
    assert_allclose(patch.get_value(1.0, 0.0)[:, 0], [1.0, 1.0, 0.0], atol=1e-12)
    assert_allclose(patch.get_value(0.37, 1.0)[:, 0], [0.0, 0.0, 1.0], atol=1e-12)


def test_coons_rejects_partially_matching_corner(surface_backend):
    """Every coordinate of a meeting corner must agree, not just x or z."""
    south = surface_backend.NurbsCurve(np.array([[0.0, 1.0], [0.0, 0.0], [0.0, 0.0]]))
    north = surface_backend.NurbsCurve(np.array([[0.0, 1.0], [1.0, 1.0], [0.0, 0.0]]))
    west = surface_backend.NurbsCurve(np.array([[0.0, 0.0], [0.2, 1.0], [0.0, 0.0]]))
    east = surface_backend.NurbsCurve(np.array([[1.0, 1.0], [0.0, 1.0], [0.0, 0.0]]))
    with pytest.raises(Exception, match="[Cc]orner"):
        surface_backend.NurbsSurfaceCoons(south, north, west, east)
