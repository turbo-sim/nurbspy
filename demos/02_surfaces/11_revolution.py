"""Generate surfaces by revolving a NURBS profile about an axis.

Build a general revolved profile, sphere, and torus; verify sphere curvatures
and colour the torus by signed Gaussian curvature.
"""

import numpy as np
import nurbspy as nrb


def main(show=True):
    figures = {}
    isocurve_display = dict(
        control_points=False,
        control_polygon=False,
        geometry_color="#536d7d",
        curve_depth_test=False,
    )
    # Revolution surface example

    # Define the control points
    P = [
        nrb.Point(0.2, 0.0, 0.0),
        nrb.Point(0.5, 0.0, 0.25),
        nrb.Point(0.55, 0.0, 0.5),
        nrb.Point(0.45, 0.0, 0.75),
        nrb.Point(0.3, 0.0, 1.0),
    ]

    # Define the array of control point weights
    W = np.asarray([1, 2, 3, 2, 1])

    # Create the generatrix NURBS curve
    nurbsGeneratrix = nrb.NurbsCurve(control_points=P, weights=W)

    # Set the a point to define the axis of revolution
    axis_point = nrb.Point(0.0, 0.0, 0.0)

    # Set a direction to define the axis of revolution (needs not be unitary)
    axis_direction = np.asarray([0.2, -0.2, 1.0])

    # Set the revolution angle
    theta_start, theta_end = 0.00, 2 * np.pi

    nurbsRevolutionSurface = nrb.NurbsSurfaceRevolution(
        nurbsGeneratrix, axis_point, axis_direction, theta_start, theta_end
    ).NurbsSurface
    figures["revolved_profile"] = nrb.Figure(
        nurbsRevolutionSurface,
        title="Surface of revolution",
        display=dict(
            control_points=True,
            control_polygon=True,
            grid=True,
            geometry_color="#0072bd",
        ),
    )

    # Revolution surface example: A sphere

    # Create the generatrix NURBS curve
    O = nrb.Point(0.0, 0.0, 0.0)
    X = np.asarray(
        [1.00, 0.00, 0.00]
    )  # Abscissa direction (negative to see normals pointing outwards)
    Y = np.asarray([0.00, 0.00, 1.00])  # Ordinate direction
    R = 0.5
    theta_start, theta_end = np.pi / 2, 3 / 2 * np.pi
    nurbsGeneratrix = nrb.CircularArc(O, X, Y, R, theta_start, theta_end).NurbsCurve

    # Set the a point to define the axis of revolution
    axis_point = nrb.Point(0.0, 0.0, 0.0)

    # Set a direction to define the axis of revolution (needs not be unitary)
    axis_direction = np.asarray([0.0, 0.0, 1.0])

    # Set the revolution angle
    theta_start, theta_end = np.pi / 2, 2 * np.pi

    nurbsRevolutionSurface = nrb.NurbsSurfaceRevolution(
        nurbsGeneratrix, axis_point, axis_direction, theta_start, theta_end
    ).NurbsSurface
    figures["sphere"] = nrb.Figure(
        nurbsRevolutionSurface,
        title="Sphere by revolution",
        display=dict(
            control_points=True,
            control_polygon=True,
            grid=True,
            geometry_color="#0072bd",
        ),
    )

    # Validate the computation of the mean and gaussian curvatures
    Nu, Nv, h = 50, 50, 1e-6
    u = np.linspace(0 + h, 1 - h, Nu)  # Small offset to avoid the poles
    v = np.linspace(0 + h, 1 - h, Nv)  # Small offset to avoid the poles
    [uu, vv] = np.meshgrid(u, v, indexing="ij")
    u, v = uu.flatten(), vv.flatten()
    K_mean, K_gauss = nurbsRevolutionSurface.get_curvature(u, v)
    error_mean = np.sum(1 / (Nu * Nv) * (K_mean - 1 / R) ** 2) ** (1 / 2)
    error_gauss = np.sum(1 / (Nu * Nv) * (K_gauss - 1 / R**2) ** 2) ** (1 / 2)
    print(
        "The two-norm error of the mean curvature computation is      :  ", error_mean
    )
    print(
        "The two-norm error of the gaussian curvature computation is  :  ", error_gauss
    )

    # Revolution surface example: A torus

    # Create the generatrix NURBS curve
    O = nrb.Point(1.0, 0.0, 0.0)
    X = np.asarray([1.00, 0.00, 0.00])  # Abscissa direction
    Y = np.asarray([0.00, 0.00, 1.00])  # Ordinate direction
    R = 0.5
    theta_start, theta_end = 0, 2 * np.pi
    nurbsGeneratrix = nrb.CircularArc(O, X, Y, R, theta_start, theta_end).NurbsCurve

    # Set the a point to define the axis of revolution
    axis_point = nrb.Point(0.0, 0.0, 0.0)

    # Set a direction to define the axis of revolution (needs not be unitary)
    axis_direction = np.asarray([0.0, 0.0, 1.0])

    # Set the revolution angle
    theta_start, theta_end = 0.00, 2 * np.pi

    nurbsRevolutionSurface = nrb.NurbsSurfaceRevolution(
        nurbsGeneratrix, axis_point, axis_direction, theta_start, theta_end
    ).NurbsSurface
    figures["torus_curvature"] = nrb.Figure(
        nurbsRevolutionSurface,
        title="Torus: signed Gaussian curvature",
        display=dict(
            control_points=True,
            control_polygon=True,
            grid=True,
            geometry_color="#0072bd",
        ),
    )
    # A symmetric scale places zero at the neutral colour of a signed quantity.
    u_grid, v_grid = np.meshgrid(
        np.linspace(0.001, 0.999, 41), np.linspace(0.001, 0.999, 41), indexing="ij"
    )
    _, gaussian = nurbsRevolutionSurface.get_curvature(u_grid.ravel(), v_grid.ravel())
    limit = float(np.max(np.abs(gaussian[np.isfinite(gaussian)])))
    figures["torus_curvature"].update(
        display=dict(
            quantity="gaussian_curvature",
            colormap="coolwarm",
            limits=[-limit, limit],
            boundary=False,
        )
    )
    for value in np.linspace(0.08, 0.92, 8):
        figures["sphere"].add(
            figures["sphere"].geometry().get_isocurve_u(float(value)),
            name=f"u={value:.2f}",
            display=isocurve_display,
        )
    for value in np.linspace(0.08, 0.92, 8):
        figures["sphere"].add(
            figures["sphere"].geometry().get_isocurve_v(float(value)),
            name=f"v={value:.2f}",
            display=isocurve_display,
        )
    figures["sphere"].update(display=dict(boundary=False))
    if show:
        for figure in figures.values():
            figure.show()
    return figures


if __name__ == "__main__":
    main()
