"""Revolve a circular profile to construct and edit a rational torus.

Use signed Gaussian-curvature colouring to distinguish the inner and outer
sides. Display the control net and retain the spatial reference grid.
"""

import numpy as np
import nurbspy as nrb


def main(show=True):
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
    figure = nrb.Figure(
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
    figure.update(
        display=dict(
            quantity="gaussian_curvature",
            colormap="coolwarm",
            limits=[-limit, limit],
            boundary=False,
        )
    )
    if show:
        figure.edit()
    return figure


if __name__ == "__main__":
    main()
