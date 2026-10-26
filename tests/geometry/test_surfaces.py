"""Surface constructors and geometry supported by the NumPy/Numba backend."""

# -------------------------------------------------------------------------------------------------------------------- #
# Importing packages
# -------------------------------------------------------------------------------------------------------------------- #
import numpy as np
from numpy.testing import assert_allclose


# -------------------------------------------------------------------------------------------------------------------- #
# Prepare the NURBS surface test suite
# -------------------------------------------------------------------------------------------------------------------- #
def test_nurbs_surface_scalar_input(surface_backend):
    """Test the NURBS surface computation in scalar mode for real and complex input."""

    # Define the array of control points
    n_dim, n, m = 3, 2, 2
    P = np.zeros((n_dim, n, m))

    # First row
    P[:, 0, 0] = [0.00, 0.00, 0.00]
    P[:, 1, 0] = [1.00, 0.00, 0.00]

    # Second row
    P[:, 0, 1] = [0.00, 1.00, 0.00]
    P[:, 1, 1] = [1.00, 1.00, 0.00]

    # Create the Bezier surface
    bezier_surface = surface_backend.NurbsSurface(control_points=P)

    # Check real
    values_real = bezier_surface.get_value(u=0.5, v=0.5).flatten()
    assert_allclose(values_real, [0.5, 0.5, 0.0], rtol=0, atol=1e-12)

    # Check complex
    values_complex = bezier_surface.get_value(u=0.5 + 0.5j, v=0.5 + 0.5j).flatten()
    assert_allclose(values_complex, [0.5 + 0.5j, 0.5 + 0.5j, 0.0], rtol=0, atol=1e-12)


def test_nurbs_surface_endpoint_property(surface_backend):
    """Test the NURBS surface end-point interpolation property."""

    # Define the array of control points
    n_dim, n, m = 3, 2, 2
    P = np.zeros((n_dim, n, m))

    # First row
    P[:, 0, 0] = [0.00, 0.00, 0.00]
    P[:, 1, 0] = [1.00, 0.00, 0.00]

    # Second row
    P[:, 0, 1] = [0.00, 1.00, 0.00]
    P[:, 1, 1] = [1.00, 1.00, 0.00]

    # Create the Bezier surface
    bezier_surface = surface_backend.NurbsSurface(control_points=P)

    # Check the corner point values
    assert_allclose(
        bezier_surface.get_value(u=0.00, v=0.00).ravel(), P[:, 0, 0], rtol=0, atol=1e-12
    )
    assert_allclose(
        bezier_surface.get_value(u=1.00, v=0.00).ravel(), P[:, 1, 0], rtol=0, atol=1e-12
    )
    assert_allclose(
        bezier_surface.get_value(u=0.00, v=1.00).ravel(), P[:, 0, 1], rtol=0, atol=1e-12
    )
    assert_allclose(
        bezier_surface.get_value(u=1.00, v=1.00).ravel(), P[:, 1, 1], rtol=0, atol=1e-12
    )


def test_planar_surface_geometry(surface_backend):
    """Test the coordinates, normals, and curvature of a planar NURBS surface."""

    # Define the array of control points
    n_dim, n, m = 3, 2, 2
    P = np.zeros((n_dim, n, m))

    # First row
    P[:, 0, 0] = [0.00, 0.00, 1.00]
    P[:, 1, 0] = [1.00, 0.00, 1.00]

    # Second row
    P[:, 0, 1] = [0.00, 1.00, 1.00]
    P[:, 1, 1] = [1.00, 1.00, 1.00]

    # Create the Bezier surface
    planar_surface = surface_backend.NurbsSurface(control_points=P)

    # Create (u,v) parametrization
    Nu, Nv = 51, 51
    u = np.linspace(0, 1, Nu)
    v = np.linspace(0, 1, Nv)
    u, v = np.meshgrid(u, v, indexing="ij")
    u, v = u.flatten(), v.flatten()

    # Check value error
    values_numeric = planar_surface.get_value(u, v)
    values_analytic = np.concatenate(
        (u[np.newaxis, :], v[np.newaxis, :], (1.00 + 0 * u)[np.newaxis, :])
    )
    assert_allclose(values_numeric, values_analytic, rtol=0, atol=1e-12)

    # Check the error of the unitary normal vectors
    normals = planar_surface.get_normals(u, v)
    expected_normals = np.zeros_like(normals)
    expected_normals[2, :] = 1.0
    assert_allclose(normals, expected_normals, rtol=0, atol=1e-12)

    # Check curvature error
    mean_curvature, gaussian_curvature = planar_surface.get_curvature(u, v)
    assert_allclose(mean_curvature, 0.0, rtol=0, atol=1e-12)
    assert_allclose(gaussian_curvature, 0.0, rtol=0, atol=1e-12)


def test_cylindrical_surface_geometry(surface_backend):
    """Test the coordinates, normals, and curvature of a cylindrical NURBS surface."""

    # Define a circular arc
    O = np.asarray([2.00, 1.00, 0.00])  # Circle center
    X = np.asarray([1.00, 0.00, 0.00])  # Abscissa direction
    Y = np.asarray([0.00, 1.00, 0.00])  # Ordinate direction
    R = 0.50  # Circle radius
    theta_start = 0.00  # Start angle
    theta_end = 2 * np.pi  # End angle

    # Create the circular arc
    curve = surface_backend.CircularArc(O, X, Y, R, theta_start, theta_end).NurbsCurve

    # Set the extrusion direction (can be unitary or not)
    direction = np.asarray([0, 0, 1])

    # Set the extrusion length
    length = 1

    # Create the ruled NURBS surface
    cylinder_surface = surface_backend.NurbsSurfaceExtruded(
        curve, direction, length
    ).NurbsSurface

    # Create (u,v) parametrization
    Nu, Nv = 51, 51
    u = np.linspace(0, 1, Nu)
    v = np.linspace(0, 1, Nv)
    u, v = np.meshgrid(u, v, indexing="ij")
    u, v = u.flatten(), v.flatten()

    # Check radius error
    values_numeric = cylinder_surface.get_value(u, v) - O[:, np.newaxis]
    radius_squared = values_numeric[0, :] ** 2 + values_numeric[1, :] ** 2
    assert_allclose(radius_squared, R**2, rtol=0, atol=1e-12)
    assert_allclose(values_numeric[2, :], length * v, rtol=0, atol=1e-12)

    # Check the error of the unitary normal vectors
    normals_numeric = cylinder_surface.get_normals(u, v)
    normals_analytic = cylinder_surface.get_value(u, v) - O[:, np.newaxis]
    normals_analytic = (
        normals_analytic
        / (np.sum((normals_analytic[[0, 1], :]) ** 2, axis=0) ** (1 / 2))[np.newaxis, :]
    )
    normals_analytic[2, :] = 0.0
    assert_allclose(normals_numeric, normals_analytic, rtol=0, atol=1e-12)

    # Check curvature error
    mean_curvature, gaussian_curvature = cylinder_surface.get_curvature(u, v)
    mean_curvature, gaussian_curvature = np.abs(mean_curvature), np.abs(
        gaussian_curvature
    )
    assert_allclose(mean_curvature, 1 / (2 * R), rtol=0, atol=1e-12)
    assert_allclose(gaussian_curvature, 0.0, rtol=0, atol=1e-12)


def test_spherical_surface_geometry(surface_backend):
    """Test the coordinates, normals, and curvature of a spherical NURBS surface."""

    # Create the generatrix NURBS curve
    O = np.asarray([0.00, 0.00, 0.00])  # Circle center
    X = np.asarray(
        [1.00, 0.00, 0.00]
    )  # Abscissa direction (negative to see normals pointing outwards)
    Y = np.asarray([0.00, 0.00, 1.00])  # Ordinate direction
    R = 0.50
    theta_start, theta_end = np.pi / 2, 3 / 2 * np.pi
    generatrix = surface_backend.CircularArc(
        O, X, Y, R, theta_start, theta_end
    ).NurbsCurve

    # Set a point to define the axis of revolution
    axis_point = np.asarray([0.0, 0.0, 0.0])

    # Set a direction to define the axis of revolution (needs not be unitary)
    axis_direction = np.asarray([0.0, 0.0, 1.0])

    # Set the revolution angle
    theta_start, theta_end = 0.00, 2 * np.pi

    # Create the NURBS surface
    spherical_surface = surface_backend.NurbsSurfaceRevolution(
        generatrix, axis_point, axis_direction, theta_start, theta_end
    ).NurbsSurface

    # Create (u,v) parametrization
    Nu, Nv = 51, 51
    u = np.linspace(0.00, 1.00, Nu)
    v = np.linspace(0.01, 0.99, Nv)  # Avoid the poles
    u, v = np.meshgrid(u, v, indexing="ij")
    u, v = u.flatten(), v.flatten()

    # Check radius error
    values_numeric = spherical_surface.get_value(u, v)
    assert_allclose(np.sum(values_numeric**2, axis=0), R**2, rtol=0, atol=1e-12)

    # Check the error of the unitary normal vectors
    normals_numeric = -spherical_surface.get_normals(u, v)
    normals_analytic = spherical_surface.get_value(u, v)
    normals_analytic = (
        normals_analytic
        / (np.sum(normals_analytic**2, axis=0) ** (1 / 2))[np.newaxis, :]
    )
    assert_allclose(normals_numeric, normals_analytic, rtol=0, atol=1e-12)

    # Check curvature error
    mean_curvature, gaussian_curvature = spherical_surface.get_curvature(u, v)
    mean_curvature, gaussian_curvature = np.abs(mean_curvature), np.abs(
        gaussian_curvature
    )
    assert_allclose(mean_curvature, 1 / R, rtol=0, atol=1e-12)
    assert_allclose(gaussian_curvature, 1 / R**2, rtol=0, atol=1e-11)


def test_nurbs_surface_bilinear(surface_backend):
    """Test the bilinear NURBS surface constructor."""

    # Set the bilinear surface defining points
    P00 = np.asarray([0.00, 0.00, 0.00])
    P01 = np.asarray([2.00, 0.50, 0.00])
    P10 = np.asarray([0.20, 1.00, 0.00])
    P11 = np.asarray([1.80, 1.50, 0.00])

    # Create the NURBS surface and evaluate the coordinates of a known case
    bilinear_surface = surface_backend.NurbsSurfaceBilinear(
        P00, P01, P10, P11
    ).NurbsSurface
    coordinates = bilinear_surface.get_value(u=0.50, v=0.50)
    assert_allclose(coordinates.ravel(), [1.00, 0.75, 0.00], rtol=0, atol=1e-8)


def test_nurbs_surface_ruled(surface_backend):
    """Test the ruled NURBS surface constructor."""

    # Define the lower NURBS curve (rational Bézier curve)
    P1 = np.zeros((3, 5))
    P1[:, 0] = [0.00, 0.00, 0.00]
    P1[:, 1] = [0.25, 0.00, 0.50]
    P1[:, 2] = [0.50, 0.00, 0.50]
    P1[:, 3] = [0.75, 0.00, 0.00]
    P1[:, 4] = [1.00, 0.00, 0.00]
    W1 = np.asarray([1, 1, 2, 1, 1])
    lower_curve = surface_backend.NurbsCurve(control_points=P1, weights=W1)

    # Define the lower NURBS curve (rational Bézier curve)
    P2 = np.zeros((3, 5))
    P2[:, 0] = [0.00, 1.00, 0.50]
    P2[:, 1] = [0.25, 1.00, 0.00]
    P2[:, 2] = [0.50, 1.00, 0.00]
    P2[:, 3] = [0.75, 1.00, 0.50]
    P2[:, 4] = [1.00, 1.00, 0.50]
    W2 = np.asarray([1, 1, 2, 1, 1])
    upper_curve = surface_backend.NurbsCurve(control_points=P2, weights=W2)

    # Create the NURBS surface and evaluate the coordinates of a known case
    ruled_surface = surface_backend.NurbsSurfaceRuled(
        lower_curve, upper_curve
    ).NurbsSurface
    coordinates = ruled_surface.get_value(u=0.50, v=0.50)
    assert_allclose(coordinates.ravel(), [0.50, 0.50, 0.25], rtol=0, atol=1e-8)


def test_nurbs_surface_extruded(surface_backend):
    """Test the extruded NURBS surface constructor."""

    # Define the base NURBS curve (rational Bézier curve)
    P = np.zeros((3, 5))
    P[:, 0] = [0.00, 0.00, 0.00]
    P[:, 1] = [0.50, 0.80, 0.00]
    P[:, 2] = [0.75, 0.60, 0.00]
    P[:, 3] = [1.00, 0.30, 0.00]
    P[:, 4] = [0.80, 0.10, 0.00]
    W = np.asarray([1, 1, 1, 1, 1])
    curve = surface_backend.NurbsCurve(control_points=P, weights=W)

    # Set the extrusion direction (can be unitary or not)
    direction = np.asarray([1, 1, 5])

    # Set the extrusion length
    length = 0.75

    # Create the NURBS surface and evaluate the coordinates of a known case
    extruded_surface = surface_backend.NurbsSurfaceExtruded(
        curve, direction, length
    ).NurbsSurface
    coordinates = extruded_surface.get_value(u=0.50, v=0.50)
    assert_allclose(
        coordinates.ravel(), [0.77841878, 0.57841878, 0.36084392], rtol=0, atol=1e-8
    )


def test_nurbs_surface_revolution(surface_backend):
    """Test the revolution NURBS surface constructor."""

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
    generatrix = surface_backend.NurbsCurve(control_points=P, weights=W)

    # Set a point to define the axis of revolution
    axis_point = np.asarray([0.0, 0.0, 0.0])

    # Set a direction to define the axis of revolution (needs not be unitary)
    axis_direction = np.asarray([0.2, -0.2, 1.0])

    # Set the revolution angle
    theta_start, theta_end = 0.00, 2 * np.pi

    # Create the NURBS surface and evaluate the coordinates of a known case
    revolution_surface = surface_backend.NurbsSurfaceRevolution(
        generatrix, axis_point, axis_direction, theta_start, theta_end
    ).NurbsSurface
    coordinates = revolution_surface.get_value(u=0.50, v=0.50)
    assert_allclose(
        coordinates.ravel(), [-0.27777778, -0.22222222, 0.61111111], rtol=0, atol=1e-8
    )


def test_nurbs_surface_coons(surface_backend):
    """Test the Coons NURBS surface constructor."""

    # Define the south boundary (rational Bézier curve)
    P = np.zeros((3, 4))
    P[:, 0] = [0.00, 0.00, 0.00]
    P[:, 1] = [0.33, 0.00, -0.40]
    P[:, 2] = [0.66, 0.10, 0.60]
    P[:, 3] = [1.00, 0.20, 0.40]
    W = np.asarray([1, 2, 2, 1])
    south_curve = surface_backend.NurbsCurve(control_points=P, weights=W)

    # Define the north boundary (rational Bézier curve)
    P = np.zeros((3, 4))
    P[:, 0] = [0.05, 1.00, 0.00]
    P[:, 1] = [0.33, 1.15, 0.40]
    P[:, 2] = [0.66, 1.15, 0.00]
    P[:, 3] = [1.05, 1.25, 0.40]
    W = np.asarray([1, 2, 2, 1])
    north_curve = surface_backend.NurbsCurve(control_points=P, weights=W)

    # Define the west boundary (rational Bézier curve)
    P = np.zeros((3, 3))
    P[:, 0] = south_curve.P[:, 0]
    P[:, 1] = [-0.20, 0.50, -0.40]
    P[:, 2] = north_curve.P[:, 0]
    W = np.asarray([south_curve.W[0], 1, north_curve.W[0]])
    west_curve = surface_backend.NurbsCurve(control_points=P, weights=W)

    # Define the east boundary (rational Bézier curve)
    P = np.zeros((3, 3))
    P[:, 0] = south_curve.P[:, -1]
    P[:, 1] = [1.15, 0.50, 0.30]
    P[:, 2] = north_curve.P[:, -1]
    W = np.asarray([south_curve.W[-1], 1, north_curve.W[-1]])
    east_curve = surface_backend.NurbsCurve(control_points=P, weights=W)

    # Create the NURBS surface and evaluate the coordinates of a known case
    coons_surface = surface_backend.NurbsSurfaceCoons(
        south_curve, north_curve, west_curve, east_curve
    ).NurbsSurface
    coordinates = coons_surface.get_value(u=0.50, v=0.50)
    assert_allclose(
        coordinates.ravel(), [0.48500000, 0.56964286, 0.08571429], rtol=0, atol=1e-8
    )
