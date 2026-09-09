""" Unit tests for the NURBS surface computations """

# -------------------------------------------------------------------------------------------------------------------- #
# Importing packages
# -------------------------------------------------------------------------------------------------------------------- #
import sys
import os
import time
import pdb
import numpy as np
import nurbspy as nrb


# -------------------------------------------------------------------------------------------------------------------- #
# Prepare the NURBS surface test suite
# -------------------------------------------------------------------------------------------------------------------- #
def test_nurbs_surface_scalar_input():

    """ Test the NURBS surface computation in scalar mode for real and complex input """

    # Define the array of control points
    n_dim, n, m = 3, 2, 2
    P = np.zeros((n_dim, n, m))

    # First row
    P[:, 0, 0] = [0.00, 0.00, 0.00]
    P[:, 1, 0] = [1.00, 0.00, 0.00]

    # Second row
    P[:, 0, 1] = [0.00, 1.00, 0.00]
    P[:, 1, 1] = [1.00, 1.00, 0.00]

    # Create and plot the Bezier surface
    bezierSurface = nrb.NurbsSurface(control_points=P)

    # Check real
    values_real = bezierSurface.get_value(u=0.5, v=0.5).flatten()
    assert np.sum((values_real - np.asarray([0.5, 0.5, 0.0])) ** 2) ** (1 / 2) < 1e-6

    # Check complex
    values_complex = bezierSurface.get_value(u=0.5 + 0.5j, v=0.5 + 0.5j).flatten()
    assert np.sum((values_complex - np.asarray([0.5 + 0.5j, 0.5 + 0.5j, 0.0])) ** 2) ** (1 / 2) < 1e-6


def test_nurbs_surface_endpoint_property():

    """ Test the NURBS surface end-point interpolation property """

    # Define the array of control points
    n_dim, n, m = 3, 2, 2
    P = np.zeros((n_dim, n, m))

    # First row
    P[:, 0, 0] = [0.00, 0.00, 0.00]
    P[:, 1, 0] = [1.00, 0.00, 0.00]

    # Second row
    P[:, 0, 1] = [0.00, 1.00, 0.00]
    P[:, 1, 1] = [1.00, 1.00, 0.00]

    # Create and plot the Bezier surface
    bezierSurface = nrb.NurbsSurface(control_points=P)

    # Check the corner point values
    assert np.sum((bezierSurface.get_value(u=0.00, v=0.00).flatten() - P[:, 0, 0]) ** 2) ** (1 / 2) < 1e-6
    assert np.sum((bezierSurface.get_value(u=1.00, v=0.00).flatten() - P[:, 1, 0]) ** 2) ** (1 / 2) < 1e-6
    assert np.sum((bezierSurface.get_value(u=0.00, v=1.00).flatten() - P[:, 0, 1]) ** 2) ** (1 / 2) < 1e-6
    assert np.sum((bezierSurface.get_value(u=1.00, v=1.00).flatten() - P[:, 1, 1]) ** 2) ** (1 / 2) < 1e-6


def test_nurbs_surface_example_1():

    """ Test the coordinates, normals, and curvature of a planar NURBS surface """

    # Define the array of control points
    n_dim, n, m = 3, 2, 2
    P = np.zeros((n_dim, n, m))

    # First row
    P[:, 0, 0] = [0.00, 0.00, 1.00]
    P[:, 1, 0] = [1.00, 0.00, 1.00]

    # Second row
    P[:, 0, 1] = [0.00, 1.00, 1.00]
    P[:, 1, 1] = [1.00, 1.00, 1.00]

    # Create and plot the Bezier surface
    planarSurface = nrb.NurbsSurface(control_points=P)

    # Create (u,v) parametrization
    Nu, Nv = 51, 51
    u = np.linspace(0, 1, Nu)
    v = np.linspace(0, 1, Nv)
    u, v = np.meshgrid(u, v, indexing='ij')
    u, v = u.flatten(), v.flatten()

    # Check value error
    values_numeric = planarSurface.get_value(u, v)
    values_analytic = np.concatenate((u[np.newaxis, :], v[np.newaxis, :], (1.00 + 0*u)[np.newaxis, :]))
    values_error = np.sum((values_analytic - values_numeric) ** 2) ** (1 / 2) / (Nu * Nv)
    print('The two-norm of the values error is             :  ', values_error)
    assert values_error < 1e-12

    # Check the error of the unitary normal vectors
    normals = planarSurface.get_normals(u, v)
    normals_error = np.sum((normals[2, :] - np.ones((u.size,)))**2) ** (1/2) / (Nu * Nv)
    print('The two-norm of the normals error is            :  ', normals_error)
    assert  normals_error < 1e-12

    # Check curvature error
    mean_curvature, gaussian_curvature = planarSurface.get_curvature(u, v)
    mean_curvature_error = np.sum(mean_curvature**2) ** (1/2) / (Nu * Nv)
    gaussian_curvature_error = np.sum(gaussian_curvature ** 2) ** (1 / 2) / (Nu * Nv)
    print('The two-norm of the mean curvature error is     :  ', mean_curvature_error)
    print('The two-norm of the gaussian curvature error is :  ', gaussian_curvature_error)
    assert  mean_curvature_error < 1e-12
    assert  gaussian_curvature_error < 1e-12


def test_nurbs_surface_example_2():

    """ Test the coordinates, normals, and curvature of a cylindrical NURBS surface """

    # Define a circular arc
    O = np.asarray([2.00, 1.00, 0.00])      # Circle center
    X = np.asarray([1.00, 0.00, 0.00])      # Abscissa direction
    Y = np.asarray([0.00, 1.00, 0.00])      # Ordinate direction
    R = 0.50                                # Circle radius
    theta_start = 0.00                      # Start angle
    theta_end = 2*np.pi                     # End angle

    # Create and plot the circular arc
    nurbsCurve = nrb.CircularArc(O, X, Y , R, theta_start, theta_end).NurbsCurve

    # Set the extrusion direction (can be unitary or not)
    direction = np.asarray([0, 0, 1])

    # Set the extrusion length
    length = 1

    # Create and plot the ruled NURBS surface
    cylinderSurface = nrb.NurbsSurfaceExtruded(nurbsCurve, direction, length).NurbsSurface

    # Create (u,v) parametrization
    Nu, Nv = 51, 51
    u = np.linspace(0, 1, Nu)
    v = np.linspace(0, 1, Nv)
    u, v = np.meshgrid(u, v, indexing='ij')
    u, v = u.flatten(), v.flatten()

    # Check radius error
    values_numeric = cylinderSurface.get_value(u, v) - O[:, np.newaxis]
    values_error = np.sum(((values_numeric[0, :]**2 + values_numeric[1, :]**2) - R**2) **2) **(1 / 2) / (Nu * Nv)
    print('The two-norm of the values error is             :  ', values_error)
    assert values_error < 1e-12

    # Check the error of the unitary normal vectors
    normals_numeric = cylinderSurface.get_normals(u, v)
    normals_analytic = cylinderSurface.get_value(u, v) - O[:, np.newaxis]
    normals_analytic = normals_analytic / (np.sum((normals_analytic[[0, 1], :])**2, axis=0) ** (1/2))[np.newaxis,:]
    normals_analytic[2, :] = 0.0
    normals_error = np.sum((normals_analytic - normals_numeric)**2) ** (1/2) / (Nu * Nv)
    print('The two-norm of the normals error is            :  ', normals_error)
    assert  normals_error < 1e-12

    # Check curvature error
    mean_curvature, gaussian_curvature = cylinderSurface.get_curvature(u, v)
    mean_curvature, gaussian_curvature = np.abs(mean_curvature), np.abs(gaussian_curvature)
    mean_curvature_error = np.sum((mean_curvature - 1/(2*R))**2) ** (1 / 2) / (Nu * Nv)
    gaussian_curvature_error = np.sum(gaussian_curvature ** 2) ** (1 / 2) / (Nu * Nv)
    print('The two-norm of the mean curvature error is     :  ', mean_curvature_error)
    print('The two-norm of the gaussian curvature error is :  ', gaussian_curvature_error)
    assert  mean_curvature_error < 1e-12
    assert  gaussian_curvature_error < 1e-12


def test_nurbs_surface_example_3():

    """ Test the coordinates, normals, and curvature of a spherical NURBS surface """

    # Create the generatrix NURBS curve
    O = np.asarray([0.00, 0.00, 0.00])    # Circle center
    X = np.asarray([1.00, 0.00, 0.00])    # Abscissa direction (negative to see normals pointing outwards)
    Y = np.asarray([0.00, 0.00, 1.00])    # Ordinate direction
    R = 0.50
    theta_start, theta_end = np.pi/2, 3/2*np.pi
    nurbsGeneratrix = nrb.CircularArc(O, X, Y, R, theta_start, theta_end).NurbsCurve

    # Set the a point to define the axis of revolution
    axis_point = np.asarray([0.0, 0.0, 0.0])

    # Set a direction to define the axis of revolution (needs not be unitary)
    axis_direction = np.asarray([0.0, 0.0, 1.0])

    # Set the revolution angle
    theta_start, theta_end = 0.00, 2*np.pi

    # Create and plot the NURBS surface
    sphericSurface = nrb.NurbsSurfaceRevolution(nurbsGeneratrix, axis_point, axis_direction, theta_start, theta_end).NurbsSurface

    # Create (u,v) parametrization
    Nu, Nv = 51, 51
    u = np.linspace(0.00, 1.00, Nu)
    v = np.linspace(0.01, 0.99, Nv)     # Avoid the poles
    u, v = np.meshgrid(u, v, indexing='ij')
    u, v = u.flatten(), v.flatten()

    # Check radius error
    values_numeric = sphericSurface.get_value(u, v)
    values_error = np.sum((np.sum(values_numeric**2, axis=0) - R**2) ** 2) ** (1 / 2) / (Nu * Nv)
    print('The two-norm of the values error is             :  ', values_error)
    assert values_error < 1e-12

    # Check the error of the unitary normal vectors
    normals_numeric = -sphericSurface.get_normals(u, v)
    normals_analytic = sphericSurface.get_value(u, v)
    normals_analytic = normals_analytic / (np.sum(normals_analytic**2, axis=0) ** (1/2))[np.newaxis,:]
    normals_error = np.sum((normals_analytic - normals_numeric)**2) ** (1/2) / (Nu * Nv)
    print('The two-norm of the normals error is            :  ', normals_error)
    assert  normals_error < 1e-12

    # Check curvature error
    mean_curvature, gaussian_curvature = sphericSurface.get_curvature(u, v)
    mean_curvature, gaussian_curvature = np.abs(mean_curvature), np.abs(gaussian_curvature)
    mean_curvature_error = np.sum((mean_curvature - 1/R) ** 2) ** (1 / 2) / (Nu * Nv)
    gaussian_curvature_error = np.sum((gaussian_curvature - 1/R**2) ** 2) ** (1 / 2) / (Nu * Nv)
    print('The two-norm of the mean curvature error is     :  ', mean_curvature_error)
    print('The two-norm of the gaussian curvature error is :  ', gaussian_curvature_error)
    assert  mean_curvature_error < 1e-12
    assert  gaussian_curvature_error < 1e-12


def test_nurbs_surface_bilinear():

    """ Test the bilinear NURBS surface constructor """

    # Set the bilinear surface defining points
    P00 = np.asarray([0.00, 0.00, 0.00])
    P01 = np.asarray([2.00, 0.50, 0.00])
    P10 = np.asarray([0.20, 1.00, 0.00])
    P11 = np.asarray([1.80, 1.50, 0.00])

    # Create the NURBS surface and evaluate the coordinates of a known case
    bilinearSurface = nrb.NurbsSurfaceBilinear(P00, P01, P10, P11).NurbsSurface
    coordinates = bilinearSurface.get_value(u=0.50, v=0.50)
    values_error = np.sum((coordinates.flatten() - np.asarray([1.00, 0.75, 0.00]))**2)**(1/2)
    print('The two-norm of the values error is             :  ', values_error)
    assert  values_error < 1e-8


def test_nurbs_surface_ruled():

    """ Test the ruled NURBS surface constructor """

    # Define the lower NURBS curve (rational Bézier curve)
    P1 = np.zeros((3, 5))
    P1[:, 0] = [0.00, 0.00, 0.00]
    P1[:, 1] = [0.25, 0.00, 0.50]
    P1[:, 2] = [0.50, 0.00, 0.50]
    P1[:, 3] = [0.75, 0.00, 0.00]
    P1[:, 4] = [1.00, 0.00, 0.00]
    W1 = np.asarray([1, 1, 2, 1, 1])
    nurbsCurve1 = nrb.NurbsCurve(control_points=P1, weights=W1)

    # Define the lower NURBS curve (rational Bézier curve)
    P2 = np.zeros((3, 5))
    P2[:, 0] = [0.00, 1.00, 0.50]
    P2[:, 1] = [0.25, 1.00, 0.00]
    P2[:, 2] = [0.50, 1.00, 0.00]
    P2[:, 3] = [0.75, 1.00, 0.50]
    P2[:, 4] = [1.00, 1.00, 0.50]
    W2 = np.asarray([1, 1, 2, 1, 1])
    nurbsCurve2 = nrb.NurbsCurve(control_points=P2, weights=W2)

    # Create the NURBS surface and evaluate the coordinates of a known case
    ruledSurface = nrb.NurbsSurfaceRuled(nurbsCurve1, nurbsCurve2).NurbsSurface
    coordinates = ruledSurface.get_value(u=0.50, v=0.50)
    values_error = np.sum((coordinates.flatten() - np.asarray([0.50, 0.50, 0.25]))**2)**(1/2)
    print('The two-norm of the values error is             :  ', values_error)
    assert  values_error < 1e-8


def test_nurbs_surface_extruded():

    """ Test the extruded NURBS surface constructor """

    # Define the base NURBS curve (rational Bézier curve)
    P = np.zeros((3, 5))
    P[:, 0] = [0.00, 0.00, 0.00]
    P[:, 1] = [0.50, 0.80, 0.00]
    P[:, 2] = [0.75, 0.60, 0.00]
    P[:, 3] = [1.00, 0.30, 0.00]
    P[:, 4] = [0.80, 0.10, 0.00]
    W = np.asarray([1, 1, 1, 1, 1])
    nurbsCurve = nrb.NurbsCurve(control_points=P, weights=W)

    # Set the extrusion direction (can be unitary or not)
    direction = np.asarray([1, 1, 5])

    # Set the extrusion length
    length = 0.75

    # Create the NURBS surface and evaluate the coordinates of a known case
    extrudedSurface = nrb.NurbsSurfaceExtruded(nurbsCurve, direction, length).NurbsSurface
    coordinates = extrudedSurface.get_value(u=0.50, v=0.50)
    values_error = np.sum((coordinates.flatten() - np.asarray([0.77841878, 0.57841878, 0.36084392]))**2)**(1/2)
    print('The two-norm of the values error is             :  ', values_error)
    assert  values_error < 1e-8


def test_nurbs_surface_revolution():

    """ Test the revolution NURBS surface constructor """

    # Define the array of control points
    P = np.zeros((3, 5))
    P[:, 0] = [0.20, 0.00, 0.00]
    P[:, 1] = [0.50, 0.00, 0.25]
    P[:, 2] = [0.55, 0.00, 0.50]
    P[:, 3] = [0.45, 0.00, 0.75]
    P[:, 4] = [0.30, 0.00, 1.00]

    # Define the array of control point weights
    W = np.asarray([1, 2, 3, 2, 1])

    # Create the generatrix NURBS curve
    nurbsGeneratrix = nrb.NurbsCurve(control_points=P, weights=W)

    # Set the a point to define the axis of revolution
    axis_point = np.asarray([0.0, 0.0, 0.0])

    # Set a direction to define the axis of revolution (needs not be unitary)
    axis_direction = np.asarray([0.2, -0.2, 1.0])

    # Set the revolution angle
    theta_start, theta_end = 0.00, 2 * np.pi

    # Create the NURBS surface and evaluate the coordinates of a known case
    revolutionSurface = nrb.NurbsSurfaceRevolution(nurbsGeneratrix, axis_point, axis_direction, theta_start, theta_end).NurbsSurface
    coordinates = revolutionSurface.get_value(u=0.50, v=0.50)
    values_error = np.sum((coordinates.flatten() - np.asarray([-0.27777778, -0.22222222, 0.61111111]))**2)**(1/2)
    print('The two-norm of the values error is             :  ', values_error)
    assert  values_error < 1e-8


def test_nurbs_surface_coons():

    """ Test the Coons NURBS surface constructor """

    # Define the south boundary (rational Bézier curve)
    P = np.zeros((3, 4))
    P[:, 0] = [0.00, 0.00, 0.00]
    P[:, 1] = [0.33, 0.00, -0.40]
    P[:, 2] = [0.66, 0.10, 0.60]
    P[:, 3] = [1.00, 0.20, 0.40]
    W = np.asarray([1, 2, 2, 1])
    nurbsCurve_south = nrb.NurbsCurve(control_points=P, weights=W)

    # Define the north boundary (rational Bézier curve)
    P = np.zeros((3, 4))
    P[:, 0] = [0.05, 1.00, 0.00]
    P[:, 1] = [0.33, 1.15, 0.40]
    P[:, 2] = [0.66, 1.15, 0.00]
    P[:, 3] = [1.05, 1.25, 0.40]
    W = np.asarray([1, 2, 2, 1])
    nurbsCurve_north = nrb.NurbsCurve(control_points=P, weights=W)

    # Define the west boundary (rational Bézier curve)
    P = np.zeros((3, 3))
    P[:, 0] = nurbsCurve_south.P[:, 0]
    P[:, 1] = [-0.20, 0.50, -0.40]
    P[:, 2] = nurbsCurve_north.P[:, 0]
    W = np.asarray([nurbsCurve_south.W[0], 1, nurbsCurve_north.W[0]])
    nurbsCurve_west = nrb.NurbsCurve(control_points=P, weights=W)

    # Define the east boundary (rational Bézier curve)
    P = np.zeros((3, 3))
    P[:, 0] = nurbsCurve_south.P[:, -1]
    P[:, 1] = [1.15, 0.50, 0.30]
    P[:, 2] = nurbsCurve_north.P[:, -1]
    W = np.asarray([nurbsCurve_south.W[-1], 1, nurbsCurve_north.W[-1]])
    nurbsCurve_east = nrb.NurbsCurve(control_points=P, weights=W)

    # Create the NURBS surface and evaluate the coordinates of a known case
    coonsSurface = nrb.NurbsSurfaceCoons(nurbsCurve_south, nurbsCurve_north, nurbsCurve_west, nurbsCurve_east).NurbsSurface
    coordinates = coonsSurface.get_value(u=0.50, v=0.50)
    values_error = np.sum((coordinates.flatten() - np.asarray([0.48500000, 0.56964286, 0.08571429]))**2)**(1/2)
    print('The two-norm of the values error is             :  ', values_error)
    assert  values_error < 1e-8


def test_nurbs_surface_zeroth_derivative():

    # Get the NURBS surface object
    nurbsSurface = get_nurbs_test_surface()

    # Create (u,v) parametrization
    Nu, Nv = 51, 51
    u = np.linspace(0, 1, Nu)
    v = np.linspace(0, 1, Nv)
    u, v = np.meshgrid(u, v, indexing='ij')
    u, v = u.flatten(), v.flatten()

    # Compute the NURBS curve values
    S  = nurbsSurface.get_value(u, v)
    dS = nurbsSurface.get_derivative(u, v, order_u=0, order_v=0)

    # Check the error
    error = np.sum((S - dS) ** 2) ** (1 / 2) / u.size
    print('Derivative (0,0) two-norm error is              :  ', error)
    assert error < 1e-12


def test_nurbs_surface_first_derivatives_cs():

    # Get the NURBS surface object
    nurbsSurface = get_nurbs_test_surface()

    # Create (u,v) parametrization
    h = 1e-12
    hh = h + h ** 2
    Nu, Nv = 51, 51
    u = np.linspace(0 + hh, 1 - hh, Nu)
    v = np.linspace(0 + hh, 1 - hh, Nv)
    u, v = np.meshgrid(u, v, indexing='ij')
    u, v = u.flatten(), v.flatten()

    # Compute the NURBS analytic derivative
    dSdu_analytic  = nurbsSurface.get_derivative(u, v, order_u=1, order_v=0)
    dSdv_analytic  = nurbsSurface.get_derivative(u, v, order_u=0, order_v=1)

    # Compute the NURBS complex step derivative
    dSdu_complex_step = np.imag(nurbsSurface.get_value(u + h * 1j, v)) / h
    dSdv_complex_step = np.imag(nurbsSurface.get_value(u, v + h * 1j)) / h

    # Check the error
    error_u  = np.sum((dSdu_analytic  - dSdu_complex_step)  ** 2) ** (1 / 2) / (Nu * Nv)
    error_v  = np.sum((dSdv_analytic  - dSdv_complex_step)  ** 2) ** (1 / 2) / (Nu * Nv)
    print('Derivative (1,0) two-norm error is              :  ', error_u)
    print('Derivative (0,1) two-norm error is              :  ', error_v)
    assert error_u < 1e-12
    assert error_v < 1e-12


def test_nurbs_surface_first_derivatives_cfd():

    # Get the NURBS surface object
    nurbsSurface = get_nurbs_test_surface()

    # Create (u,v) parametrization
    h = 1e-5
    hh = h + h ** 2
    Nu, Nv = 51, 51
    u = np.linspace(0 + hh, 1 - hh, Nu)
    v = np.linspace(0 + hh, 1 - hh, Nv)
    u, v = np.meshgrid(u, v, indexing='ij')
    u, v = u.flatten(), v.flatten()

    # Compute the NURBS analytic derivative
    dSdu_analytic  = nurbsSurface.get_derivative(u, v, order_u=1, order_v=0)
    dSdv_analytic  = nurbsSurface.get_derivative(u, v, order_u=0, order_v=1)

    # Compute the NURBS complex step derivative
    dSdu_cfd = (nurbsSurface.get_value(u+h, v) - nurbsSurface.get_value(u-h, v)) / (2*h)
    dSdv_cfd = (nurbsSurface.get_value(u, v+h) - nurbsSurface.get_value(u, v-h)) / (2*h)

    # Check the error
    error_u  = np.sum((dSdu_analytic  - dSdu_cfd)  ** 2) ** (1 / 2) / (Nu * Nv)
    error_v  = np.sum((dSdv_analytic  - dSdv_cfd)  ** 2) ** (1 / 2) / (Nu * Nv)
    print('Derivative (1,0) two-norm error is              :  ', error_u)
    print('Derivative (0,1) two-norm error is              :  ', error_v)
    assert error_u < 1e-6
    assert error_v < 1e-6


def test_nurbs_surface_second_derivatives_cfd():

    # Get the NURBS surface object
    nurbsSurface = get_nurbs_test_surface()

    # Create (u,v) parametrization
    h = 1e-5
    hh = h + h ** 2
    Nu, Nv = 51, 51
    u = np.linspace(0 + hh, 1 - hh, Nu)
    v = np.linspace(0 + hh, 1 - hh, Nv)
    u, v = np.meshgrid(u, v, indexing='ij')
    u, v = u.flatten(), v.flatten()

    # Compute the NURBS analytic derivative
    dSdu_analytic  = nurbsSurface.get_derivative(u, v, order_u=2, order_v=0)
    dSdv_analytic  = nurbsSurface.get_derivative(u, v, order_u=0, order_v=2)
    dSduv_analytic = nurbsSurface.get_derivative(u, v, order_u=1, order_v=1)

    # Compute the NURBS complex step derivative
    dSdu_cfd = (nurbsSurface.get_value(u + h, v) - 2 * nurbsSurface.get_value(u, v) + nurbsSurface.get_value(u - h, v)) / h ** 2
    dSdv_cfd = (nurbsSurface.get_value(u, v + h) - 2 * nurbsSurface.get_value(u, v) + nurbsSurface.get_value(u, v - h)) / h ** 2
    dSduv_cfd = (nurbsSurface.get_value(u + h, v + h) - nurbsSurface.get_value(u - h, v + h) -
                 nurbsSurface.get_value(u + h, v - h) + nurbsSurface.get_value(u - h, v - h)) / (4 * h ** 2)

    # Check the error
    error_u  = np.sum((dSdu_analytic  - dSdu_cfd)  ** 2) ** (1 / 2) / (Nu * Nv)
    error_v  = np.sum((dSdv_analytic  - dSdv_cfd)  ** 2) ** (1 / 2) / (Nu * Nv)
    error_uv = np.sum((dSduv_analytic - dSduv_cfd) ** 2) ** (1 / 2) / (Nu * Nv)
    print('Derivative (2,0) two-norm error is              :  ', error_u)
    print('Derivative (0,2) two-norm error is              :  ', error_v)
    print('Derivative (1,1) two-norm error is              :  ', error_uv)
    assert error_u < 1e-6
    assert error_v < 1e-6
    assert error_uv < 1e-6


def test_nurbs_surface_point_projection():

    # Create test NURBS surface
    nurbs_surface = get_nurbs_test_surface()

    # Compute a point on the NURBS surface
    u0, v0 = 1 / 3, 5 / 7
    P = nurbs_surface.get_value(u0, v0)

    # Project the point on the NURBS surface
    u_opt, v_opt = nurbs_surface.project_point_to_surface(P)

    # Check the error
    error = np.sqrt((u0 - u_opt) ** 2 + (v0 - v_opt) ** 2)
    print('The error of the projected parameter value is   : ', error)
    assert error < 1e-6


def get_nurbs_test_surface():

    # Define the array of control points
    n_dim, n, m = 3, 5, 4
    P = np.zeros((n_dim, n, m))

    # First row
    P[:, 0, 0] = [0.00, 3.00, 0.00]
    P[:, 1, 0] = [1.00, 2.00, 0.00]
    P[:, 2, 0] = [2.00, 1.50, 0.00]
    P[:, 3, 0] = [3.00, 2.00, 0.00]
    P[:, 4, 0] = [4.00, 3.00, 0.00]

    # Second row
    P[:, 0, 1] = [0.00, 3.00, 1.00]
    P[:, 1, 1] = [1.00, 2.00, 1.00]
    P[:, 2, 1] = [2.00, 1.50, 1.00]
    P[:, 3, 1] = [3.00, 2.00, 1.00]
    P[:, 4, 1] = [4.00, 3.00, 1.00]

    # Third row
    P[:, 0, 2] = [0.00, 3.00, 2.00]
    P[:, 1, 2] = [1.00, 2.00, 2.00]
    P[:, 2, 2] = [2.00, 1.50, 2.00]
    P[:, 3, 2] = [3.00, 2.00, 2.00]
    P[:, 4, 2] = [4.00, 3.00, 2.00]

    # Fourth row
    P[:, 0, 3] = [0.50, 3.00, 3.00]
    P[:, 1, 3] = [1.00, 2.50, 3.00]
    P[:, 2, 3] = [2.00, 2.00, 3.00]
    P[:, 3, 3] = [3.00, 2.50, 3.00]
    P[:, 4, 3] = [3.50, 3.00, 3.00]

    # Define the array of control point weights
    W = np.zeros((n, m))
    W[:, 0] = np.asarray([1, 1, 5, 1, 1])
    W[:, 1] = np.asarray([1, 1, 10, 1, 1])
    W[:, 2] = np.asarray([1, 1, 10, 1, 1])
    W[:, 3] = np.asarray([1, 1, 5, 1, 1])

    # Maximum index of the control points (counting from zero)
    n = np.shape(P)[1] - 1
    m = np.shape(P)[2] - 1

    # Define the order of the basis polynomials
    # Linear (p = 1), Quadratic (p = 2), Cubic (p = 3), etc.
    # Set p = n (number of control points minus one) to obtain a Bezier
    p = 3
    q = 3

    # Define the knot vectors (clamped spline)
    # p+1 zeros, n-p equispaced points between 0 and 1, and p+1 ones.  In total r+1 points where r=n+p+1
    # q+1 zeros, m-p equispaced points between 0 and 1, and q+1 ones. In total s+1 points where s=m+q+1
    U = np.concatenate((np.zeros(p), np.linspace(0, 1, n - p + 2), np.ones(p)))
    V = np.concatenate((np.zeros(q), np.linspace(0, 1, m - q + 2), np.ones(q)))

    # Create and plot the NURBS surface
    nurbsSurface = nrb.NurbsSurface(control_points=P, weights=W, u_degree=p, v_degree=q, u_knots=U, v_knots=V)

    return nurbsSurface



# -------------------------------------------------------------------------------------------------------------------- #
# Check the functions manually
# -------------------------------------------------------------------------------------------------------------------- #
# Run this file directly to inspect the checks or set debugger breakpoints.
if __name__ == "__main__":
    test_nurbs_surface_scalar_input()
    test_nurbs_surface_endpoint_property()
    test_nurbs_surface_example_1()
    test_nurbs_surface_example_2()
    test_nurbs_surface_example_3()
    test_nurbs_surface_bilinear()
    test_nurbs_surface_ruled()
    test_nurbs_surface_extruded()
    test_nurbs_surface_revolution()
    test_nurbs_surface_coons()
    test_nurbs_surface_zeroth_derivative()
    test_nurbs_surface_first_derivatives_cs()
    test_nurbs_surface_first_derivatives_cfd()
    test_nurbs_surface_second_derivatives_cfd()
    test_nurbs_surface_point_projection()