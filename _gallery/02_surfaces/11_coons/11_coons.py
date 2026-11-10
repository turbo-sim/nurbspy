"""Interpolate four boundary curves with a Coons surface patch.

Construct compatible north, south, east, and west curves, then show the
interpolating surface with its boundaries, parameter curves, and control net.

Open one construction in the editor with a reference grid.
Isocurves follow edits. Other annotations illustrate the initial construction.
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
    # Coons surface example

    # Define the south boundary (rational Bézier curve)
    P = [
        nrb.Point(0.0, 0.0, 0.0),
        nrb.Point(0.33, 0.0, -0.4),
        nrb.Point(0.66, 0.1, 0.6),
        nrb.Point(1.0, 0.2, 0.4),
    ]
    W = np.asarray([1, 2, 2, 1])
    nurbsCurve_south = nrb.NurbsCurve(control_points=P, weights=W)

    # Define the north boundary (rational Bézier curve)
    P = [
        nrb.Point(0.05, 1.0, 0.0),
        nrb.Point(0.33, 1.15, 0.4),
        nrb.Point(0.66, 1.15, 0.0),
        nrb.Point(1.05, 1.25, 0.4),
    ]
    W = np.asarray([1, 2, 2, 1])
    nurbsCurve_north = nrb.NurbsCurve(control_points=P, weights=W)

    # Define the west boundary (rational Bézier curve)
    P = [
        nrb.Point(*nurbsCurve_south.P[:, 0]),
        nrb.Point(-0.2, 0.5, -0.4),
        nrb.Point(*nurbsCurve_north.P[:, 0]),
    ]
    W = np.asarray([nurbsCurve_south.W[0], 1, nurbsCurve_north.W[0]])
    nurbsCurve_west = nrb.NurbsCurve(control_points=P, weights=W)

    # Define the east boundary (rational Bézier curve)
    P = [
        nrb.Point(*nurbsCurve_south.P[:, -1]),
        nrb.Point(1.15, 0.5, 0.3),
        nrb.Point(*nurbsCurve_north.P[:, -1]),
    ]
    W = np.asarray([nurbsCurve_south.W[-1], 1, nurbsCurve_north.W[-1]])
    nurbsCurve_east = nrb.NurbsCurve(control_points=P, weights=W)

    coonsNurbsSurface = nrb.NurbsSurfaceCoons(
        nurbsCurve_south, nurbsCurve_north, nurbsCurve_west, nurbsCurve_east
    ).NurbsSurface
    figure = nrb.Figure(
        coonsNurbsSurface,
        title="Coons patch interpolating four boundary curves",
        display=dict(
            control_points=True,
            control_polygon=True,
            grid=True,
            isocurves=True,
            geometry_color="#0072bd",
        ),
    )
    figure.add(
        nurbsCurve_south,
        name="South boundary",
        display=boundary_display,
    )
    figure.add(
        nurbsCurve_north,
        name="North boundary",
        display=boundary_display,
    )
    figure.add(
        nurbsCurve_west,
        name="West boundary",
        display=boundary_display,
    )
    figure.add(
        nurbsCurve_east,
        name="East boundary",
        display=boundary_display,
    )
    if show:
        figure.edit()
    return figure


if __name__ == "__main__":
    main()
