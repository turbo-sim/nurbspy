"""Curve evaluation and geometric invariants shared by NumPy and JAX."""

import numpy as np
from numpy.testing import assert_allclose


def test_curve_accepts_float_parameter(curve_backend):
    """Evaluate a line at one floating-point parameter with a singleton output."""

    # Define the array of control points
    P = np.zeros((2, 2))
    P[:, 0] = [0.00, 0.00]
    P[:, 1] = [1.00, 1.00]

    # Create the NURBS curve
    bezierCurve = curve_backend.NurbsCurve(control_points=P)

    # Check real
    values_real = bezierCurve.get_value(u=0.5).flatten()
    assert np.sum((values_real - np.asarray([0.5, 0.5])) ** 2) ** (1 / 2) < 1e-6


def test_curve_accepts_integer_parameter(curve_backend):
    """Accept an integer endpoint parameter without changing the curve value."""

    # Define the array of control points
    P = np.zeros((2, 2))
    P[:, 0] = [0.00, 0.00]
    P[:, 1] = [1.00, 1.00]

    # Create the NURBS curve
    bezierCurve = curve_backend.NurbsCurve(control_points=P)

    # Check u=0
    values_real = bezierCurve.get_value(u=0).flatten()
    assert np.sum((values_real - np.asarray([0.0, 0.0])) ** 2) ** (1 / 2) < 1e-6


def test_clamped_curve_interpolates_endpoints(curve_backend):
    """Test the NURBS curve end-point interpolation property"""

    # Define the array of control points
    P = np.zeros((3, 4))
    P[:, 0] = [1.00, 2.00, 1.00]
    P[:, 1] = [2.00, 1.00, 2.00]
    P[:, 2] = [3.00, 2.00, 3.00]
    P[:, 3] = [4.00, 1.00, 4.00]

    # Define the array of control point weights
    W = np.asarray([2.00, 3.00, 1.00, 2.00])

    # Maximum index of the control points (counting from zero)
    n = np.shape(P)[1] - 1

    # Define the order of the basis polynomials
    p = 2

    # Define the knot vector (clamped spline)
    # p+1 zeros, n-p equispaced points between 0 and 1, and p+1 ones. In total r+1 points where r=n+p+1
    U = np.concatenate((np.zeros(p), np.linspace(0, 1, n - p + 2), np.ones(p)))

    # Create the NURBS curve
    nurbsCurve = curve_backend.NurbsCurve(
        control_points=P, weights=W, degree=p, knots=U
    )

    # Check the corner point values
    assert (
        np.sum((nurbsCurve.get_value(u=0.00).flatten() - P[:, 0]) ** 2) ** (1 / 2)
        < 1e-6
    )
    assert (
        np.sum((nurbsCurve.get_value(u=1.00).flatten() - P[:, -1]) ** 2) ** (1 / 2)
        < 1e-6
    )


def test_bspline_matches_nurbs_book_example(curve_backend):
    """Test the B-Spline curve computation against a known example (Ex2.2 from the NURBS book)"""

    # Define the array of control points
    P2 = np.asarray([1, 3, 5])
    P3 = np.asarray([2, 1, 4])
    P4 = np.asarray([3, 0, 6])
    P = np.zeros((3, 8))
    P[:, 2] = P2
    P[:, 3] = P3
    P[:, 4] = P4

    # Define the order of the basis polynomials
    p = 2

    # Define the knot vector (clamped spline)
    # p+1 zeros, n-p equispaced points between 0 and 1, and p+1 ones. In total r+1 points where r=n+p+1
    #                 u0    u1    u2    u3    u4    u5    u6    u7    u8    u9   u10
    U = np.asarray([0.00, 0.00, 0.00, 1.00, 2.00, 3.00, 4.00, 4.00, 5.00, 5.00, 5.00])

    # Create the B-Spline curve
    myBSpline = curve_backend.NurbsCurve(control_points=P, knots=U, degree=p)

    # Evaluate the B-Spline curve numerically
    u = 5 / 2  # u-parameter
    values_numeric = myBSpline.get_value(u)

    # Evaluate the B-Spline curve analytically (NURBS book page 82)
    values_analytic = (1 / 8 * P2 + 6 / 8 * P3 + 1 / 8 * P4)[:, np.newaxis]

    # Check the error
    error = np.sum((values_analytic - values_numeric) ** 2) ** (1 / 2)
    print("The two-norm of the evaluation error is         :  ", error)
    assert error < 1e-8


def test_circle_from_four_quarter_arcs(curve_backend):
    """Test the NURBS curve value against a known example (NURBS book section 7.5)"""

    # Create a circle using 4 arcs of 90 degrees
    # Define the array of control points
    P = np.zeros((2, 9))
    P[:, 0] = [1.00, 0.00]
    P[:, 1] = [1.00, 1.00]
    P[:, 2] = [0.00, 1.00]
    P[:, 3] = [-1.00, 1.00]
    P[:, 4] = [-1.00, 0.00]
    P[:, 5] = [-1.00, -1.00]
    P[:, 6] = [0.00, -1.00]
    P[:, 7] = [1.00, -1.00]
    P[:, 8] = [1.00, 0.00]

    # Define the array of control point weights
    W = np.asarray(
        [1, np.sqrt(2) / 2, 1, np.sqrt(2) / 2, 1, np.sqrt(2) / 2, 1, np.sqrt(2) / 2, 1]
    )

    # Define the order of the basis polynomials
    p = 2

    # Define the knot vector (clamped spline)
    # p+1 zeros, n minus p equispaced points between 0 and 1, and p+1 ones.  In total r+1 points where r=n+p+1
    U = np.asarray([0, 0, 0, 1 / 4, 1 / 4, 1 / 2, 1 / 2, 3 / 4, 3 / 4, 1, 1, 1])

    # Create the NURBS curve
    myCircle = curve_backend.NurbsCurve(P, W, p, U)

    # Define the u-parametrization
    u = np.linspace(0, 1, 101)

    # Check the radius error
    coords = myCircle.get_value(u)
    radius_error = np.sum((np.sum((coords) ** 2, axis=0) - 1) ** 2) ** (1 / 2)
    print("The two-norm of the evaluation error is         :  ", radius_error)
    assert radius_error < 1e-8


def test_circle_from_three_third_arcs(curve_backend):
    """Test the NURBS curve value against a known example (NURBS book section 7.5)"""

    # Create a circle using 3 arcs of 120 degrees
    # Define the array of control points
    a = np.cos(np.pi / 6)
    P = np.zeros((2, 7))
    P[:, 0] = [a, 1 / 2]
    P[:, 1] = [0, 2]
    P[:, 2] = [-a, 1 / 2]
    P[:, 3] = [-2 * a, -1]
    P[:, 4] = [0, -1]
    P[:, 5] = [2 * a, -1]
    P[:, 6] = [a, 1 / 2]

    # Define the array of control point weights
    W = np.asarray([1, 1 / 2, 1, 1 / 2, 1, 1 / 2, 1])

    # Define the order of the basis polynomials
    p = 2

    # Define the knot vector (clamped spline)
    # p+1 zeros, n minus p equispaced points between 0 and 1, and p+1 ones.  In total r+1 points where r=n+p+1
    U = np.asarray([0, 0, 0, 1 / 3, 1 / 3, 2 / 3, 2 / 3, 1, 1, 1])

    # Create the NURBS curve
    myCircle = curve_backend.NurbsCurve(P, W, p, U)

    # Define the u-parametrization
    u = np.linspace(0, 1, 101)

    # Check the radius error
    coords = myCircle.get_value(u)
    radius_error = np.sum((np.sum((coords) ** 2, axis=0) - 1) ** 2) ** (1 / 2)
    print("The two-norm of the evaluation error is         :  ", radius_error)
    assert radius_error < 1e-8


def test_planar_circular_arc_geometry(curve_backend):
    """Test the computation of a circular NURBS curve value, curvature and arc-length in 2D"""

    # Set defining parameters
    O = np.asarray([0.00, 1.00])  # Circle center
    X = np.asarray([1.00, 0.00])  # Abscissa direction
    Y = np.asarray([0.00, 1.00])  # Ordinate direction
    R = 0.5  # Circle radius
    theta_start = 1 / 6 * np.pi  # Start angle
    theta_end = 3 / 2 * np.pi - 1 / 6 * np.pi  # End angle

    # Create the circular arc
    my_circular_arc = curve_backend.CircularArc(
        O, X, Y, R, theta_start, theta_end
    ).NurbsCurve

    # Define the u-parametrization
    u = np.linspace(0, 1, 101)

    # Check the radius error
    coords = my_circular_arc.get_value(u)
    radius_error = np.sum(
        (np.sum((coords - O[:, np.newaxis]) ** 2, axis=0) - R**2) ** 2
    ) ** (1 / 2)
    print("The two-norm of the radius error is             :  ", radius_error)
    assert radius_error < 1e-8

    # Check the curvature error
    curvature = my_circular_arc.get_curvature(u)
    curvature_error = np.sum((curvature - 1 / R) ** 2) ** (1 / 2)
    print("The two-norm of the curvature error is          :  ", curvature_error)
    assert curvature_error < 1e-8

    # Check the arc length error
    arc_length = my_circular_arc.get_arclength()
    arc_length_error = arc_length - R * np.abs(theta_end - theta_start)
    print("The arc length computation error is             :  ", arc_length_error)
    # The default NumPy quadrature uses one fixed rule across multiple spans.
    assert_allclose(arc_length, R * np.abs(theta_end - theta_start), rtol=0, atol=1e-2)


def test_spatial_circular_arc_geometry_and_frame(curve_backend):
    """Test the computation of a circular NURBS curve value, curvature and arc-length in 3D"""

    # Set defining parameters
    O = np.asarray([0.00, 0.00, 0.50])  # Circle center
    X = np.asarray([3.00, 0.00, 0.00])  # Abscissa direction
    Y = np.asarray([0.00, 1.00, 0.00])  # Ordinate direction
    R = 0.5  # Circle radius
    theta_start = 0.00  # Start angle
    theta_end = np.pi / 2  # End angle

    # Create the circular arc
    my_circular_arc = curve_backend.CircularArc(
        O, X, Y, R, theta_start, theta_end
    ).NurbsCurve

    # Define the u-parametrization
    u = np.linspace(0, 1, 101)

    # Check the radius error
    coords = my_circular_arc.get_value(u)
    radius_error = np.sum(
        (np.sum((coords - O[:, np.newaxis]) ** 2, axis=0) - R**2) ** 2
    ) ** (1 / 2)
    print("The two-norm of the radius error is             :  ", radius_error)
    assert radius_error < 1e-8

    # Check the curvature error
    curvature = my_circular_arc.get_curvature(u)
    curvature_error = np.sum((curvature - 1 / R) ** 2) ** (1 / 2)
    print("The two-norm of the curvature error is          :  ", curvature_error)
    assert curvature_error < 1e-8

    # Check the arc length error
    arc_length = my_circular_arc.get_arclength()
    arc_length_error = arc_length - R * np.abs(theta_end - theta_start)
    print("The arc length computation error is             :  ", arc_length_error)
    assert_allclose(arc_length, R * np.abs(theta_end - theta_start), rtol=0, atol=1e-5)

    # Check the Frenet-Serret frame of reference computation (starting point)
    assert (
        np.sum(
            (my_circular_arc.get_tangent(u=0.00) - np.asarray([0, 1, 0])[:, np.newaxis])
            ** 2
        )
        ** (1 / 2)
        < 1e-6
    )
    assert (
        np.sum(
            (my_circular_arc.get_normal(u=0.00) - np.asarray([-1, 0, 0])[:, np.newaxis])
            ** 2
        )
        ** (1 / 2)
        < 1e-6
    )
    assert (
        np.sum(
            (
                my_circular_arc.get_binormal(u=0.00)
                - np.asarray([0, 0, 1])[:, np.newaxis]
            )
            ** 2
        )
        ** (1 / 2)
        < 1e-6
    )

    # Check the Frenet-Serret frame of reference computation (end point)
    assert (
        np.sum(
            (
                my_circular_arc.get_tangent(u=1.00)
                - np.asarray([-1, 0, 0])[:, np.newaxis]
            )
            ** 2
        )
        ** (1 / 2)
        < 1e-6
    )
    assert (
        np.sum(
            (my_circular_arc.get_normal(u=1.00) - np.asarray([0, -1, 0])[:, np.newaxis])
            ** 2
        )
        ** (1 / 2)
        < 1e-6
    )
    assert (
        np.sum(
            (
                my_circular_arc.get_binormal(u=1.00)
                - np.asarray([0, 0, 1])[:, np.newaxis]
            )
            ** 2
        )
        ** (1 / 2)
        < 1e-6
    )
