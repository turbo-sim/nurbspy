""" Unit tests for the NURBS curve computations """

# -------------------------------------------------------------------------------------------------------------------- #
# Importing packages
# -------------------------------------------------------------------------------------------------------------------- #
import sys
import os
import time
import pdb
import numpy as np
import nurbspy.jax as nrb
import scipy.integrate


# -------------------------------------------------------------------------------------------------------------------- #
# Prepare the NURBS curve test suite
# -------------------------------------------------------------------------------------------------------------------- #
def test_nurbs_curve_float_input():
    """Test the Bezier curve computation in scalar mode"""

    # Define the array of control points
    P = np.zeros((2, 2))
    P[:, 0] = [0.00, 0.00]
    P[:, 1] = [1.00, 1.00]

    # Create the NURBS curve
    bezierCurve = nrb.NurbsCurve(control_points=P)

    # Check real
    values_real = bezierCurve.get_value(u=0.5).flatten()
    assert np.sum((values_real - np.asarray([0.5, 0.5])) ** 2) ** (1 / 2) < 1e-6


def test_nurbs_curve_integer_input():
    """ Test the Bezier curve computation in scalar mode"""

    # Define the array of control points
    P = np.zeros((2, 2))
    P[:, 0] = [0.00, 0.00]
    P[:, 1] = [1.00, 1.00]

    # Create the NURBS curve
    bezierCurve = nrb.NurbsCurve(control_points=P)

    # Check u=0
    values_real = bezierCurve.get_value(u=0).flatten()
    assert np.sum((values_real - np.asarray([0.0, 0.0])) ** 2) ** (1 / 2) < 1e-6


def test_nurbs_curve_endpoint_interpolation():
    """ Test the NURBS curve end-point interpolation property """

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
    nurbsCurve = nrb.NurbsCurve(control_points=P, weights=W, degree=p, knots=U)

    # Check the corner point values
    assert np.sum((nurbsCurve.get_value(u=0.00).flatten() - P[:,  0]) ** 2) ** (1 / 2) < 1e-6
    assert np.sum((nurbsCurve.get_value(u=1.00).flatten() - P[:, -1]) ** 2) ** (1 / 2) < 1e-6


def test_nurbs_curve_endpoint_curvature():
    """ Test the NURBS curve end-point curvature property """

    # Define the array of control points
    P = np.zeros((3,11))
    P[:, 0]  = [0.00, 0.00, 0.00]
    P[:, 1]  = [0.10, 0.50, 0.00]
    P[:, 2]  = [0.20, 0.00, 0.50]
    P[:, 3]  = [0.30, 0.50, 1.00]
    P[:, 4]  = [0.40, 0.00, 0.50]
    P[:, 5]  = [0.50, 0.50, 0.00]
    P[:, 6]  = [0.60, 0.00, 0.50]
    P[:, 7]  = [0.70, 0.50, 1.00]
    P[:, 8]  = [0.80, 0.00, 0.50]
    P[:, 9]  = [0.90, 0.00, 0.00]
    P[:, 10] = [1.00, 0.00, 0.00]

    # Maximum index of the control points (counting from zero)
    n = np.shape(P)[1] - 1

    # Define the array of control point weights
    W = np.asarray([1, 2, 3, 2, 1, 2, 3, 2, 1, 2, 3])

    # Define the order of the basis polynomials
    p = 4

    # Define the knot vector (clamped spline)
    # p+1 zeros, n-p equispaced points between 0 and 1, and p+1 ones. In total r+1 points where r=n+p+1
    U = np.concatenate((np.zeros(p), np.linspace(0, 1, n - p + 2), np.ones(p)))

    # Create the NURBS curve
    myCurve = nrb.NurbsCurve(control_points=P, weights=W, degree=p, knots=U)

    # Get NURBS curve parameters
    p = myCurve.p
    U = myCurve.U
    P = myCurve.P
    W = myCurve.W
    n = np.shape(P)[1] - 1

    # Get the endpoint curvature numerically
    curvature_1a = myCurve.get_curvature(u=0.00)[0]
    curvature_1b = myCurve.get_curvature(u=1.00)[0]

    # Get the endpoint curvature analytically
    curvature_2a = (p - 1) / p * (U[p+1] / U[p+2]) * (W[2] * W[0] / W[1]**2) * \
                    np.sum(np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])**2)**(1/2) * \
                    np.sum((P[:, 1] - P[:, 0])**2)**(-3/2)

    curvature_2b = (p - 1) / p * (1 - U[n]) / (1 - U[n-1]) * (W[n] * W[n-2] / W[n-1]**2) * \
                    np.sum(np.cross(P[:, n-1] - P[:, n], P[:, n-2] - P[:, n])**2)**(1/2) * \
                    np.sum((P[:, n-1] - P[:, n])**2)**(-3/2)

    # Check the error
    error_curvature_start = np.sqrt((curvature_1a - curvature_2a) ** 2)
    error_curvature_end   = np.sqrt((curvature_1b - curvature_2b) ** 2)
    print('Start point curvature error                     :  ', error_curvature_start)
    print('End point curvature error                       :  ', error_curvature_end)
    assert error_curvature_start < 1e-10
    assert error_curvature_end < 1e-6


def test_nurbs_curve_example_1():
    """ Test the B-Spline curve computation against a known example (Ex2.2 from the NURBS book) """

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
    myBSpline = nrb.NurbsCurve(control_points=P, knots=U, degree=p)

    # Evaluate the B-Spline curve numerically
    u = 5/2   # u-parameter
    values_numeric = myBSpline.get_value(u)

    # Evaluate the B-Spline curve analytically (NURBS book page 82)
    values_analytic = (1/8*P2 + 6/8*P3 + 1/8*P4)[:, np.newaxis]

    # Check the error
    error = np.sum((values_analytic - values_numeric) ** 2) ** (1 / 2)
    print('The two-norm of the evaluation error is         :  ', error)
    assert error < 1e-8


def test_nurbs_curve_example_2():
    """ Test the NURBS curve value against a known example (NURBS book section 7.5) """

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
    W = np.asarray([1, np.sqrt(2) / 2, 1, np.sqrt(2) / 2, 1, np.sqrt(2) / 2, 1, np.sqrt(2) / 2, 1])

    # Define the order of the basis polynomials
    p = 2

    # Define the knot vector (clamped spline)
    # p+1 zeros, n minus p equispaced points between 0 and 1, and p+1 ones.  In total r+1 points where r=n+p+1
    U = np.asarray([0, 0, 0, 1 / 4, 1 / 4, 1 / 2, 1 / 2, 3 / 4, 3 / 4, 1, 1, 1])

    # Create the NURBS curve
    myCircle = nrb.NurbsCurve(P, W, p, U)

    # Define the u-parametrization
    u = np.linspace(0, 1, 101)

    # Check the radius error
    coords = myCircle.get_value(u)
    radius_error = np.sum((np.sum((coords)**2, axis=0) - 1) ** 2) ** (1/2)
    print('The two-norm of the evaluation error is         :  ', radius_error)
    assert radius_error < 1e-8


def test_nurbs_curve_example_3():
    """ Test the NURBS curve value against a known example (NURBS book section 7.5) """

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
    myCircle = nrb.NurbsCurve(P, W, p, U)

    # Define the u-parametrization
    u = np.linspace(0, 1, 101)

    # Check the radius error
    coords = myCircle.get_value(u)
    radius_error = np.sum((np.sum((coords)**2, axis=0) - 1) ** 2) ** (1/2)
    print('The two-norm of the evaluation error is         :  ', radius_error)
    assert radius_error < 1e-8


def test_nurbs_curve_example_4():
    """ Test the computation of a circular NURBS curve value, curvature and arc-length in 2D """

    # Set defining parameters
    O = np.asarray([0.00, 1.00])                 # Circle center
    X = np.asarray([1.00, 0.00])                 # Abscissa direction
    Y = np.asarray([0.00, 1.00])                 # Ordinate direction
    R = 0.5                                      # Circle radius
    theta_start = 1 / 6 * np.pi                  # Start angle
    theta_end = 3 / 2 * np.pi - 1 / 6 * np.pi    # End angle

    # Create and plot the circular arc
    my_circular_arc = nrb.CircularArc(O, X, Y, R, theta_start, theta_end).NurbsCurve

    # Define the u-parametrization
    u = np.linspace(0, 1, 101)

    # Check the radius error
    coords = my_circular_arc.get_value(u)
    radius_error = np.sum((np.sum((coords-O[:, np.newaxis])**2, axis=0) - R**2) ** 2) ** (1/2)
    print('The two-norm of the radius error is             :  ', radius_error)
    assert radius_error < 1e-8

    # CHeck the curvature error
    curvature = my_circular_arc.get_curvature(u)
    curvature_error = np.sum((curvature - 1 / R) ** 2) ** (1 / 2)
    print("The two-norm of the curvature error is          :  ", curvature_error)
    assert curvature_error < 1e-8

    # Check the arc length error
    arc_length = my_circular_arc.get_arclength()
    arc_length_error = arc_length - R * np.abs(theta_end - theta_start)
    print("The arc length computation error is             :  ", arc_length_error)
    assert arc_length_error < 1e-2


def test_nurbs_curve_example_5():
    """ Test the computation of a circular NURBS curve value, curvature and arc-length in 3D """

    # Set defining parameters
    O = np.asarray([0.00, 0.00, 0.50])      # Circle center
    X = np.asarray([3.00, 0.00, 0.00])      # Abscissa direction
    Y = np.asarray([0.00, 1.00, 0.00])      # Ordinate direction
    R = 0.5                                 # Circle radius
    theta_start = 0.00                      # Start angle
    theta_end = np.pi/2                     # End angle

    # Create and plot the circular arc
    my_circular_arc = nrb.CircularArc(O, X, Y , R, theta_start, theta_end).NurbsCurve

    # Define the u-parametrization
    u = np.linspace(0, 1, 101)

    # Check the radius error
    coords = my_circular_arc.get_value(u)
    radius_error = np.sum((np.sum((coords-O[:, np.newaxis])**2, axis=0) - R**2) ** 2) ** (1/2)
    print('The two-norm of the radius error is             :  ', radius_error)
    assert radius_error < 1e-8

    # CHeck the curvature error
    curvature = my_circular_arc.get_curvature(u)
    curvature_error = np.sum((curvature - 1 / R) ** 2) ** (1 / 2)
    print("The two-norm of the curvature error is          :  ", curvature_error)
    assert curvature_error < 1e-8

    # Check the arc length error
    arc_length = my_circular_arc.get_arclength()
    arc_length_error = arc_length - R * np.abs(theta_end - theta_start)
    print("The arc length computation error is             :  ", arc_length_error)
    assert arc_length_error < 1e-2

    # Check the Frenet-Serret frame of reference computation (starting point)
    assert np.sum((my_circular_arc.get_tangent(u=0.00)  - np.asarray([0, 1, 0])[:, np.newaxis]) ** 2) ** (1 / 2) < 1e-6
    assert np.sum((my_circular_arc.get_normal(u=0.00)  - np.asarray([-1, 0, 0])[:, np.newaxis]) ** 2) ** (1 / 2) < 1e-6
    assert np.sum((my_circular_arc.get_binormal(u=0.00) - np.asarray([0, 0, 1])[:, np.newaxis]) ** 2) ** (1 / 2) < 1e-6

    # Check the Frenet-Serret frame of reference computation (end point)
    assert np.sum((my_circular_arc.get_tangent(u=1.00)  - np.asarray([-1, 0, 0])[:, np.newaxis]) ** 2) ** (1 / 2) < 1e-6
    assert np.sum((my_circular_arc.get_normal(u=1.00)   - np.asarray([0, -1, 0])[:, np.newaxis]) ** 2) ** (1 / 2) < 1e-6
    assert np.sum((my_circular_arc.get_binormal(u=1.00) - np.asarray([0, 0, 1])[:, np.newaxis]) ** 2) ** (1 / 2) < 1e-6


def test_nurbs_curve_zeroth_derivative():
    """ Test that the zero-th derivative agrees with the function evaluation """

    # Define the array of control points
    P = np.zeros((3, 5))
    P[:, 0] = [0.00, 0.00, 0.00]
    P[:, 1] = [0.00, 0.30, 0.05]
    P[:, 2] = [0.25, 0.30, 0.30]
    P[:, 3] = [0.50, 0.30, -0.05]
    P[:, 4] = [0.50, 0.10, 0.10]

    # Maximum index of the control points (counting from zero)
    n = np.shape(P)[1] - 1

    # Define the array of control point weights
    W = np.asarray([1, 1, 3, 1, 1])

    # Define the order of the basis polynomials
    # Linear (p = 1), Quadratic (p = 2), Cubic (p = 3), etc.
    # Set p = n (number of control points minus one) to obtain a Bezier
    p = 3

    # Define the knot vector (clamped spline)
    # p+1 zeros, n-p equispaced points between 0 and 1, and p+1 ones. In total r+1 points where r=n+p+1
    U = np.concatenate((np.zeros(p), np.linspace(0, 1, n - p + 2), np.ones(p)))

    # Create the NURBS curve
    nurbs3D = nrb.NurbsCurve(control_points=P, weights=W, degree=p, knots=U)

    # Define the u-parametrization
    u = np.linspace(0, 1, 101)

    # Compute the NURBS curve values
    C  = nurbs3D.get_value(u)
    dC = nurbs3D.get_derivative(u, order=0)

    # Check the error
    error = np.sum((C - dC) ** 2) ** (1 / 2) / u.size
    print('The two-norm of the zeroth derivative error is  :  ', error)
    assert error < 1e-8


def test_nurbs_curve_first_derivative_cfd():
    """ Test the first derivative against central finite differences """

    # Define the array of control points
    P = np.zeros((3, 5))
    P[:, 0] = [0.00, 0.00, 0.00]
    P[:, 1] = [0.00, 0.30, 0.05]
    P[:, 2] = [0.25, 0.30, 0.30]
    P[:, 3] = [0.50, 0.30, -0.05]
    P[:, 4] = [0.50, 0.10, 0.10]

    # Maximum index of the control points (counting from zero)
    n = np.shape(P)[1] - 1

    # Define the array of control point weights
    W = np.asarray([1, 1, 3, 1, 1])

    # Define the order of the basis polynomials
    # Linear (p = 1), Quadratic (p = 2), Cubic (p = 3), etc.
    # Set p = n (number of control points minus one) to obtain a Bezier
    p = 3

    # Define the knot vector (clamped spline)
    # p+1 zeros, n-p equispaced points between 0 and 1, and p+1 ones. In total r+1 points where r=n+p+1
    U = np.concatenate((np.zeros(p), np.linspace(0, 1, n - p + 2), np.ones(p)))

    # Create the NURBS curve
    nurbs3D = nrb.NurbsCurve(control_points=P, weights=W, degree=p, knots=U)

    # Define a new u-parametrization suitable for finite differences
    h = 1e-5
    hh = h + h ** 2
    Nu = 1000
    u = np.linspace(0.00 + hh, 1.00 - hh, Nu)  # Make sure that the limits [0, 1] also work when making changes

    # Compute the NURBS analytic derivative
    dC_analytic = nurbs3D.get_derivative(u, order=1)

    # Compute the NURBS central finite differences derivative
    a = -1 / 2 * nurbs3D.get_value(u - h)
    b = +1 / 2 * nurbs3D.get_value(u + h)
    dC_cfd = (a + b) / h

    # Check the error
    error = np.sum((dC_analytic - dC_cfd) ** 2) ** (1 / 2) / u.size
    print('The two-norm of the first derivative error is   :  ', error)
    assert error < 1e-6


def test_nurbs_curve_second_derivative_cfd():
    """ Test the first derivative against central finite differences """

    # Define the array of control points
    P = np.zeros((3, 5))
    P[:, 0] = [0.00, 0.00, 0.00]
    P[:, 1] = [0.00, 0.30, 0.05]
    P[:, 2] = [0.25, 0.30, 0.30]
    P[:, 3] = [0.50, 0.30, -0.05]
    P[:, 4] = [0.50, 0.10, 0.10]

    # Maximum index of the control points (counting from zero)
    n = np.shape(P)[1] - 1

    # Define the array of control point weights
    W = np.asarray([1, 1, 3, 1, 1])

    # Define the order of the basis polynomials
    # Linear (p = 1), Quadratic (p = 2), Cubic (p = 3), etc.
    # Set p = n (number of control points minus one) to obtain a Bezier
    p = 3

    # Define the knot vector (clamped spline)
    # p+1 zeros, n-p equispaced points between 0 and 1, and p+1 ones. In total r+1 points where r=n+p+1
    U = np.concatenate((np.zeros(p), np.linspace(0, 1, n - p + 2), np.ones(p)))

    # Create the NURBS curve
    nurbs3D = nrb.NurbsCurve(control_points=P, weights=W, degree=p, knots=U)

    # Define a new u-parametrization suitable for finite differences
    h = 1e-5
    hh = h + h ** 2
    Nu = 1000
    u = np.linspace(0.00 + hh, 1.00 - hh, Nu)  # Make sure that the limits [0, 1] also work when making changes

    # Compute the NURBS analytic derivative
    dC_analytic = nurbs3D.get_derivative(u, order=2)

    # Compute the NURBS central finite differences derivative
    a = +1 * nurbs3D.get_value(u - h)
    b = -2 * nurbs3D.get_value(u)
    c = +1 * nurbs3D.get_value(u + h)
    dC_cfd = (a + b + c) / h**2

    # Check the error
    error = np.sum((dC_analytic - dC_cfd) ** 2) ** (1 / 2) / u.size
    print('The two-norm of the second derivative error is  :  ', error)
    assert error < 1e-6


def test_nurbs_curve_first_derivative_endpoint():
    """ Test the first derivative at the end points of the curve """

    # Define the array of control points
    P = np.zeros((3, 5))
    P[:, 0] = [0.00, 0.00, 0.00]
    P[:, 1] = [0.00, 0.30, 0.05]
    P[:, 2] = [0.25, 0.30, 0.30]
    P[:, 3] = [0.50, 0.30, -0.05]
    P[:, 4] = [0.50, 0.10, 0.10]

    # Maximum index of the control points (counting from zero)
    n = np.shape(P)[1] - 1

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
    nurbs3D = nrb.NurbsCurve(control_points=P, weights=W, degree=p, knots=U)

    # Start point
    dC_numeric_0 = nurbs3D.get_derivative(u=0.00, order=1)[:, 0]
    dC_analytic_0 = (p / U[p+1]) * (W[1] / W[0]) * (P[:, 1] - P[:, 0])

    # End point
    dC_numeric_1 = nurbs3D.get_derivative(u=1.00, order=1)[:, 0]
    dC_analytic_1 = (p / (1-U[n])) * (W[n-1] / W[n]) * (P[:, n] - P[:, n-1])

    # Check the error
    error_0 = np.sum((dC_analytic_0 - dC_numeric_0) ** 2) ** (1 / 2)
    error_1 = np.sum((dC_analytic_1 - dC_numeric_1) ** 2) ** (1 / 2)
    print('The start point first derivative error is       :  ', error_0)
    print('The end point first derivative error is         :  ', error_1)
    assert error_0 < 1e-6
    assert error_1 < 1e-6


def test_nurbs_curve_second_derivative_endpoint():
    """ Test the second derivative at the end points of the curve """

    # Define the array of control points
    P = np.zeros((3, 5))
    P[:, 0] = [0.00, 0.00, 0.00]
    P[:, 1] = [0.10, 0.30, 0.00]
    P[:, 2] = [0.25, 0.30, 0.30]
    P[:, 3] = [0.50, 0.30, -0.05]
    P[:, 4] = [0.50, 0.10, 0.10]

    # Maximum index of the control points (counting from zero)
    n = np.shape(P)[1] - 1

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
    nurbs3D = nrb.NurbsCurve(control_points=P, weights=W, degree=p, knots=U)

    # Start point
    ddC_numeric_0 = nurbs3D.get_derivative(u=0.00, order=2)[:, 0]
    ddC_analytic_0 = p*(p-1) / U[p+1] * (1/U[p+2] * (W[2] / W[0]) * (P[:, 2] - P[:, 0]) - (1/U[p+1] + 1/U[p+2]) * (W[1] / W[0]) * (P[:, 1] - P[:, 0]) ) + 2 * (p / U[p + 1]) ** 2 * (W[1] / W[0]) * (1 - W[1] / W[0]) * (P[:, 1] - P[:, 0])

    # End point
    ddC_numeric_1 = nurbs3D.get_derivative(u=1.00, order=2)[:, 0]
    ddC_analytic_1 = p*(p-1) / (1 - U[n]) * (1/(1 - U[n-1]) * (W[n-2] / W[n]) * (P[:, n-2] - P[:, n]) - (1/(1 - U[n]) + 1/(1 - U[n-1])) * (W[n-1] / W[n]) * (P[:, n-1] - P[:, n]) ) + 2 * (p / (1 - U[n])) ** 2 * (W[n-1] / W[n]) * (1 - W[n-1] / W[n]) * (P[:, n-1] - P[:, n])

    # Check the error
    error_0 = np.sum((ddC_analytic_0 - ddC_numeric_0) ** 2) ** (1 / 2)
    error_1 = np.sum((ddC_analytic_1 - ddC_numeric_1) ** 2) ** (1 / 2)
    print('The start point second derivative error is      :  ', error_0)
    print('The end point second derivative error is        :  ', error_1)
    assert error_0 < 1e-6
    assert error_1 < 1e-6


def test_nurbs_curve_point_projection():

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
    nurbs3D = nrb.NurbsCurve(control_points=Q, weights=W, degree=p, knots=U)

    # Compute a point on the NURBS curve
    u0 = 5/7
    Q = 1.1*nurbs3D.get_value(u0).squeeze()

    # Project the point back onto the NURBS curve
    u_proj = nurbs3D.project_point_to_curve(Q)
    Q_proj = nurbs3D.get_value(u_proj).squeeze()
    dC_proj = nurbs3D.get_derivative(u_proj, 1).squeeze()

    # Check orthogonality condition: (C(u*) − Q) · C'(u*) = 0
    residual = np.sum((Q_proj - Q) * dC_proj)
    res_norm = float(np.abs(residual))
    print(f"The orthogonality residual is            : {res_norm:.3e}")

    # Assert orthogonality
    assert res_norm < 1e-8, "Projection orthogonality condition not satisfied."

    # # Plot the NURBS curve and both points
    # import matplotlib.pyplot as plt
    # fig, ax = nurbs3D.plot(frenet_serret=False)
    # ax.plot(Q[0], Q[1], Q[2], "ko", label="Original point (offset)")
    # ax.plot(Q_proj[0], Q_proj[1], Q_proj[2], "ro", label="Projected point")
    # ax.legend()
    # plt.show()



# -------------------------------------------------------------------------------------------------------------------- #
# Check the functions manually
# -------------------------------------------------------------------------------------------------------------------- #
test_nurbs_curve_float_input()
test_nurbs_curve_integer_input()
test_nurbs_curve_endpoint_interpolation()
test_nurbs_curve_endpoint_curvature()
test_nurbs_curve_example_1()
test_nurbs_curve_example_2()
test_nurbs_curve_example_3()
test_nurbs_curve_example_4()
test_nurbs_curve_example_5()
test_nurbs_curve_zeroth_derivative()
test_nurbs_curve_first_derivative_cfd()
test_nurbs_curve_second_derivative_cfd()
test_nurbs_curve_first_derivative_endpoint()
test_nurbs_curve_second_derivative_endpoint()
test_nurbs_curve_point_projection()
