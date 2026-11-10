"""Extrude an exact circular profile along a fixed direction and edit the surface.

Display one extrusion with its control net and initial generating curve and
live parameter curves. The background grid gives a sense of space.
"""

import numpy as np
import nurbspy as nrb


def main(show=True):
    boundary_display = dict(
        geometry_color="#d97823",
        control_points=False,
        control_polygon=False,
        curve_depth_test=False,
    )
    # Extruded surface example: A circular cylinder

    # Define a circular arc
    O = nrb.Point(0.0, 0.0, 0.0)
    X = np.asarray([1.00, 0.00, 0.00])  # Abscissa direction
    Y = np.asarray([0.00, 1.00, 0.00])  # Ordinate direction
    R = 0.5  # Circle radius
    theta_start = 0.00  # Start angle
    theta_end = 2 * np.pi  # End angle

    nurbsCurve = nrb.CircularArc(O, X, Y, R, theta_start, theta_end).NurbsCurve

    # Set the extrusion direction (can be unitary or not)
    direction = np.asarray([0, 0, 1])

    # Set the extrusion length
    length = 1

    extrudedNurbsSurface = nrb.NurbsSurfaceExtruded(
        nurbsCurve, direction, length
    ).NurbsSurface
    figure = nrb.Figure(
        extrudedNurbsSurface,
        title="Circular cylinder by extrusion",
        display=dict(
            control_points=True,
            control_polygon=True,
            grid=True,
            isocurves=True,
            geometry_color="#0072bd",
        ),
    )
    figure.add(
        nurbsCurve,
        name="Generating curve",
        display=boundary_display,
    )
    if show:
        figure.edit()
    return figure


if __name__ == "__main__":
    main()
