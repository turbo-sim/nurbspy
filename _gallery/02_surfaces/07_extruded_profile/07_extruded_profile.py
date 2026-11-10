"""Extrude a NURBS profile along a fixed direction and edit the surface.

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
    # Extruded surface example:

    # Define the base NURBS curve (rational Bézier curve)
    P = [
        nrb.Point(0.0, 0.0, 0.0),
        nrb.Point(0.5, 0.8, 0.0),
        nrb.Point(0.75, 0.6, 0.0),
        nrb.Point(1.0, 0.3, 0.0),
        nrb.Point(0.8, 0.1, 0.0),
    ]
    W = np.asarray([1, 1, 1, 1, 1])
    nurbsCurve = nrb.NurbsCurve(control_points=P, weights=W)

    # Set the extrusion direction (can be unitary or not)
    direction = np.asarray([1, 1, 5])

    # Set the extrusion length
    length = 1

    extrudedNurbsSurface = nrb.NurbsSurfaceExtruded(
        nurbsCurve, direction, length
    ).NurbsSurface
    figure = nrb.Figure(
        extrudedNurbsSurface,
        title="Extruded rational profile",
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
