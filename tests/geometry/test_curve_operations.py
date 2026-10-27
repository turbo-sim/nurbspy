"""Curve projection, merging, arc length, and inverse parametrizations."""

import numpy as np
from numpy.testing import assert_allclose
import pytest


def _project_points(backend, curve, queries):
    """Adapt the two public projection APIs to a common batch of query columns."""
    if backend.__name__ == "nurbspy.jax":
        return np.asarray(curve.project_points(queries))
    return np.array(
        [
            curve.project_point_to_curve(query, ftol=1e-14, gtol=1e-12)
            for query in queries.T
        ]
    )


def test_segment_projection_interior_and_endpoints(curve_backend):
    """Find the global closest point, including queries beyond both endpoints."""
    curve = curve_backend.NurbsCurve(np.array([[0.0, 1.0], [0.0, 0.0]]))
    queries = np.array([[-0.4, 0.0, 0.35, 1.0, 1.4], [0.2, -0.3, 0.2, 0.3, -0.2]])
    expected = np.clip(queries[0], 0.0, 1.0)

    # The closest point on this horizontal segment has a clipped x coordinate.
    actual = _project_points(curve_backend, curve, queries)
    assert_allclose(actual, expected, rtol=0, atol=1e-8)
    expected_points = np.vstack([expected, np.zeros_like(expected)])
    assert_allclose(curve.get_value(actual), expected_points, rtol=0, atol=1e-8)


def test_projection_recovers_known_normal_offsets(curve_backend):
    """Recover known closest points for short normal offsets from a parabola."""
    # C(u) = (u, u**2); these offsets stay inside the local radius of curvature.
    curve = curve_backend.NurbsCurve(np.array([[0.0, 0.5, 1.0], [0.0, 0.0, 1.0]]))
    expected = np.array([0.15, 0.45, 0.8])
    normals = np.stack([-2 * expected, np.ones_like(expected)])
    normals = normals / np.linalg.norm(normals, axis=0)
    queries = np.asarray(curve.get_value(expected)) + 0.03 * normals

    actual = _project_points(curve_backend, curve, queries)
    assert actual.shape == expected.shape
    assert_allclose(actual, expected, rtol=0, atol=1e-8)
    distances = np.linalg.norm(np.asarray(curve.get_value(actual)) - queries, axis=0)
    assert_allclose(distances, 0.03, rtol=0, atol=1e-10)


def test_straight_line_arclength(curve_backend):
    """Integrate a line with exact total and partial lengths of five and three."""
    curve = curve_backend.NurbsCurve(np.array([[1.0, 4.0], [2.0, 6.0]]))
    assert_allclose(curve.get_arclength(), 5.0, rtol=1e-10)
    assert_allclose(curve.get_arclength(0.2, 0.8), 3.0, rtol=1e-10)


def test_jax_straight_line_quadrature(jax_backend):
    """Check JAX's additional fixed-quadrature arc-length API on an exact line."""
    curve = jax_backend.NurbsCurve(np.array([[1.0, 4.0], [2.0, 6.0]]))
    assert_allclose(curve.get_arclength_quad_fixed(), 5.0, rtol=1e-10)
    assert_allclose(curve.get_arclength_quad_fixed(0.2, 0.8), 3.0, rtol=1e-10)


@pytest.mark.parametrize("n_points", [17, 32, 41, 42])
def test_jax_quadrature_node_requests_on_exact_circle(jax_backend, n_points):
    """Support requested counts on both sides of an embedded-rule boundary."""
    curve = jax_backend.NurbsCurve(
        np.array([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]]),
        weights=np.array([1.0, np.sqrt(0.5), 1.0]),
    )
    assert_allclose(curve.get_arclength_quad_fixed(n_points=n_points), np.pi / 2, rtol=0, atol=1e-10)


def test_merged_curves_preserve_segment_geometry(curve_backend):
    """Merging two connected curves preserves the geometry of each segment."""
    first = curve_backend.NurbsCurve(np.array([[0.0, 0.5, 1.0], [0.0, 0.0, 0.0]]))
    second = curve_backend.NurbsCurve(np.array([[1.0, 1.0, 1.0], [0.0, 0.5, 1.0]]))
    merged = curve_backend.merge_nurbs_curves([first, second])

    # Equal knot ranges give each original curve half of the merged interval.
    samples = np.array([0.0, 0.25, 0.75, 1.0])
    expected = np.array([[0.0, 0.5, 1.0, 1.0], [0.0, 0.0, 0.5, 1.0]])
    assert_allclose(merged.get_value(samples), expected, rtol=0, atol=1e-7)


def test_jax_coordinate_reparametrization(jax_backend):
    """Invert a monotone quadratic coordinate against its analytic inverse."""
    # This quadratic Bezier has x(u) = u + u**2 and y(u) = u.
    curve = jax_backend.NurbsCurve(np.array([[0.0, 0.5, 2.0], [0.0, 0.5, 1.0]]))
    inverse = curve.reparametrize_by_coordinate(dim=0, tol=1e-11)
    coordinates = np.array([0.1, 0.8, 1.7])
    expected = (np.sqrt(1 + 4 * coordinates) - 1) / 2
    assert_allclose(inverse(coordinates), expected, rtol=0, atol=1e-10)


def test_jax_arclength_reparametrization(jax_backend):
    """Invert distance along a straight curve whose parameter speed varies."""
    # x(u) = u + u**2: distance from the origin equals x despite varying speed.
    curve = jax_backend.NurbsCurve(np.array([[0.0, 0.5, 2.0], [0.0, 0.0, 0.0]]))
    inverse, total = curve.reparametrize_by_arclength(tol=1e-10)
    distances = np.array([0.0, 0.2, 0.8, 1.5, 2.0])
    expected = (np.sqrt(1 + 4 * distances) - 1) / 2
    assert_allclose(total, 2.0, rtol=0, atol=1e-9)
    assert_allclose(inverse(distances), expected, rtol=0, atol=1e-8)


def test_nurbs_curve_point_projection(curve_backend):
    """Check stationarity of an interior projection onto a spatial rational curve."""

    # Define the array of control points
    Q = np.zeros((3, 5))
    Q[:, 0] = [0.00, 0.00, 0.00]
    Q[:, 1] = [0.10, 0.30, 0.00]
    Q[:, 2] = [0.25, 0.30, 0.30]
    Q[:, 3] = [0.50, 0.30, -0.05]
    Q[:, 4] = [0.50, 0.10, 0.10]

    # Maximum index of the control points (counting from zero)
    n = np.shape(Q)[1] - 1

    # Define the array of control point weights
    W = np.asarray([1, 2, 3, 2, 1])

    # Define the order of the basis polynomials
    # Linear (p = 1), Quadratic (p = 2), Cubic (p = 3), etc.
    # Set p = n (number of control points minus one) to obtain a Bezier
    p = 3

    # Define the knot vector (clamped spline)
    # p+1 zeros, n-p equispaced points between 0 and 1, and p+1 ones. In total r+1 points where r=n+p+1
    U = np.concatenate((np.zeros(p), np.linspace(0, 1, n - p + 2), np.ones(p)))

    # Create the NURBS curve
    nurbs3D = curve_backend.NurbsCurve(control_points=Q, weights=W, degree=p, knots=U)

    # Compute a point on the NURBS curve
    u0 = 5 / 7
    Q = 1.1 * nurbs3D.get_value(u0).squeeze()

    # Project the point back onto the NURBS curve
    if curve_backend.__name__ == "nurbspy.jax":
        u_proj = nurbs3D.project_points(Q[:, np.newaxis])[0]
    else:
        u_proj = nurbs3D.project_point_to_curve(Q, ftol=1e-14, gtol=1e-12)
    Q_proj = nurbs3D.get_value(u_proj).squeeze()
    dC_proj = nurbs3D.get_derivative(u_proj, 1).squeeze()

    # Check orthogonality condition: (C(u*) − Q) · C'(u*) = 0
    residual = np.sum((Q_proj - Q) * dC_proj)
    res_norm = float(np.abs(residual))
    print(f"The orthogonality residual is            : {res_norm:.3e}")

    # Assert orthogonality
    assert res_norm < 1e-8, "Projection orthogonality condition not satisfied."
