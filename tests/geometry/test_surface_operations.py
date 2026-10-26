"""Public surface operations supported by the NumPy/Numba backend."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from tests.cases import rational_surface_data


@pytest.fixture
def rational_surface(surface_backend):
    """Return a fresh rational patch so operations cannot affect another test."""
    return surface_backend.NurbsSurface(**rational_surface_data())


@pytest.mark.parametrize("fixed_axis", ["u", "v"])
@pytest.mark.parametrize("fixed_parameter", [0.0, 0.17, 0.5, 0.89, 1.0])
def test_isocurve_matches_surface_section(
    rational_surface, fixed_axis, fixed_parameter
):
    """Extracted isocurves preserve coordinates and tangents, including boundaries."""
    parameters = np.array([0.0, 0.13, 0.5, 0.84, 1.0])
    fixed_parameters = np.full_like(parameters, fixed_parameter)

    # Fixing u leaves a curve in v; fixing v leaves a curve in u.
    if fixed_axis == "u":
        curve = rational_surface.get_isocurve_u(fixed_parameter)
        expected_values = rational_surface.get_value(fixed_parameters, parameters)
        expected_derivatives = rational_surface.get_derivative(
            fixed_parameters, parameters, order_u=0, order_v=1
        )
    else:
        curve = rational_surface.get_isocurve_v(fixed_parameter)
        expected_values = rational_surface.get_value(parameters, fixed_parameters)
        expected_derivatives = rational_surface.get_derivative(
            parameters, fixed_parameters, order_u=1, order_v=0
        )

    assert_allclose(
        curve.get_value(parameters), expected_values, rtol=1e-12, atol=1e-12
    )
    assert_allclose(
        curve.get_derivative(parameters, order=1),
        expected_derivatives,
        rtol=1e-12,
        atol=1e-12,
    )


@pytest.mark.parametrize("direction", ["u", "v"])
def test_attachment_preserves_both_patches_and_seam(
    surface_backend, rational_surface, direction
):
    """Attaching compatible patches maps each to half the domain without a seam gap."""
    first = rational_surface
    second_points = first.P.copy()
    second_weights = first.W.copy()

    # Make a distinct second patch and copy its joining edge in homogeneous form.
    # Matching both coordinates and weights preserves the entire rational seam.
    if direction == "u":
        second_points[0] += 4.0
        second_points[:, 0, :] = first.P[:, -1, :]
        second_weights[0, :] = first.W[-1, :]
    else:
        second_points[2] += 3.0
        second_points[:, :, 0] = first.P[:, :, -1]
        second_weights[:, 0] = first.W[:, -1]
    second = surface_backend.NurbsSurface(
        control_points=second_points,
        weights=second_weights,
        u_degree=first.p,
        v_degree=first.q,
        u_knots=first.U.copy(),
        v_knots=first.V.copy(),
    )

    # Include both endpoints and internal knots when comparing the original patches.
    u, v = np.meshgrid(
        [0.0, 0.19, 0.5, 0.81, 1.0], [0.0, 0.27, 0.73, 1.0], indexing="ij"
    )
    u, v = u.ravel(), v.ravel()
    seam_parameter = np.linspace(0.0, 1.0, 13)
    if direction == "u":
        attached = first.attach_nurbs_udir(second)
        first_values = attached.get_value(u / 2, v)
        second_values = attached.get_value(0.5 + u / 2, v)
        seam_values = attached.get_value(
            np.full_like(seam_parameter, 0.5), seam_parameter
        )
        expected_seam = first.get_value(np.ones_like(seam_parameter), seam_parameter)
        second_seam = second.get_value(np.zeros_like(seam_parameter), seam_parameter)
    else:
        attached = first.attach_nurbs_vdir(second)
        first_values = attached.get_value(u, v / 2)
        second_values = attached.get_value(u, 0.5 + v / 2)
        seam_values = attached.get_value(
            seam_parameter, np.full_like(seam_parameter, 0.5)
        )
        expected_seam = first.get_value(seam_parameter, np.ones_like(seam_parameter))
        second_seam = second.get_value(seam_parameter, np.zeros_like(seam_parameter))

    assert_allclose(first_values, first.get_value(u, v), rtol=1e-12, atol=1e-12)
    assert_allclose(second_values, second.get_value(u, v), rtol=1e-12, atol=1e-12)
    assert_allclose(seam_values, expected_seam, rtol=1e-12, atol=1e-12)
    assert_allclose(seam_values, second_seam, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("offset", [-0.4, 0.0, 0.25])
def test_planar_offset_has_requested_signed_distance(surface_backend, offset):
    """A plane's offset is an exact translation opposite its oriented unit normal."""
    origin = np.array([0.5, -1.0, 2.0])
    tangent_u = np.array([2.0, 0.5, 0.0])
    tangent_v = np.array([0.25, 1.0, 1.0])
    control_points = np.empty((3, 2, 2))
    control_points[:, 0, 0] = origin
    control_points[:, 1, 0] = origin + tangent_u
    control_points[:, 0, 1] = origin + tangent_v
    control_points[:, 1, 1] = origin + tangent_u + tangent_v
    plane = surface_backend.NurbsSurface(control_points=control_points)
    normal = np.cross(tangent_u, tangent_v)
    normal /= np.linalg.norm(normal)

    # The general offset algorithm is approximate; a bilinear plane is an exact case.
    offset_surface = surface_backend.make_offset_surface(plane, offset)
    u, v = np.meshgrid(
        np.linspace(0.0, 1.0, 7), np.linspace(0.0, 1.0, 9), indexing="ij"
    )
    u, v = u.ravel(), v.ravel()
    expected_values = plane.get_value(u, v) - offset * normal[:, np.newaxis]
    expected_normals = np.broadcast_to(normal[:, np.newaxis], expected_values.shape)

    assert_allclose(offset_surface.get_value(u, v), expected_values, rtol=0, atol=1e-12)
    assert_allclose(
        offset_surface.get_normals(u, v), expected_normals, rtol=0, atol=1e-12
    )


@pytest.mark.parametrize(
    "unconstrained_parameters",
    [
        (0.3, 0.6),
        (-0.2, 0.6),
        (1.2, 0.6),
        (0.3, -0.2),
        (0.3, 1.2),
        (-0.2, -0.3),
        (-0.2, 1.3),
        (1.2, -0.3),
        (1.2, 1.3),
    ],
    ids=[
        "interior",
        "u-min",
        "u-max",
        "v-min",
        "v-max",
        "corner-00",
        "corner-01",
        "corner-10",
        "corner-11",
    ],
)
def test_projection_finds_closest_point_on_bounded_plane(
    surface_backend, unconstrained_parameters
):
    """Off-surface points project to the known interior, edge, or corner minimum."""
    control_points = np.array(
        [
            [[1.0, 1.0], [3.0, 3.0]],
            [[-2.0, 1.0], [-2.0, 1.0]],
            [[0.75, 0.75], [0.75, 0.75]],
        ]
    )
    plane = surface_backend.NurbsSurface(control_points=control_points)
    parameter_u, parameter_v = unconstrained_parameters
    target = np.array([1.0 + 2.0 * parameter_u, -2.0 + 3.0 * parameter_v, 1.95])

    # Orthogonal projection onto this rectangle clips each in-plane coordinate.
    expected_parameters = np.clip(unconstrained_parameters, 0.0, 1.0)
    expected_point = np.array(
        [1.0 + 2.0 * expected_parameters[0], -2.0 + 3.0 * expected_parameters[1], 0.75]
    )
    parameters = plane.project_point_to_surface(target, ftol=1e-12, gtol=1e-10)
    projected_point = plane.get_value(*parameters).ravel()

    assert_allclose(parameters, expected_parameters, rtol=0, atol=1e-6)
    assert_allclose(projected_point, expected_point, rtol=0, atol=3e-6)
    assert_allclose(
        np.linalg.norm(projected_point - target),
        np.linalg.norm(expected_point - target),
        rtol=0,
        atol=1e-10,
    )


def test_point_projection_recovers_rational_surface_parameters(rational_surface):
    """Projecting a point already on a rational surface recovers its parameters."""
    # Compute a point on the NURBS surface.
    u0, v0 = 1 / 3, 5 / 7
    point = rational_surface.get_value(u0, v0)

    # Project the point on the NURBS surface.
    u_opt, v_opt = rational_surface.project_point_to_surface(point)

    # Check the recovered parameters independently, rather than averaging the errors.
    assert_allclose([u_opt, v_opt], [u0, v0], rtol=0, atol=1e-6)
