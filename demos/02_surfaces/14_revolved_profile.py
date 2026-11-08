"""Revolve a weighted profile about an inclined axis.

Define the profile, revolution axis, and angle explicitly, then edit one surface
with its control net and a background reference grid.
"""

import numpy as np
import nurbspy as nrb


def main(show=True):
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
    figure = nrb.Figure(
        nurbsRevolutionSurface,
        title="Surface of revolution",
        display=dict(
            control_points=True,
            control_polygon=True,
            grid=True,
            geometry_color="#0072bd",
        ),
    )
    if show:
        figure.edit()
    return figure


if __name__ == "__main__":
    main()
