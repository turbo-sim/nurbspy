"""Generate a spherical patch by revolving a circular arc.

Check analytical mean and Gaussian curvature before opening the editor.
Display the control net and initial parameter curves over a reference grid.
"""

import numpy as np
import nurbspy as nrb


def main(show=True):
    isocurve_display = dict(
        control_points=False,
        control_polygon=False,
        geometry_color="#536d7d",
        curve_depth_test=False,
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
    figure = nrb.Figure(
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

    for value in np.linspace(0.08, 0.92, 8):
        figure.add(
            figure.geometry().get_isocurve_u(float(value)),
            name=f"u={value:.2f}",
            display=isocurve_display,
        )
    for value in np.linspace(0.08, 0.92, 8):
        figure.add(
            figure.geometry().get_isocurve_v(float(value)),
            name=f"v={value:.2f}",
            display=isocurve_display,
        )
    figure.update(display=dict(boundary=False))
    if show:
        figure.edit()
    return figure


if __name__ == "__main__":
    main()
