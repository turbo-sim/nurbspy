"""Span straight lines between two crossed circular arcs.

Construct and edit one ruled surface with its control net and a reference grid.
Boundary curves illustrate the initial construction; isocurves follow edits.
"""

import numpy as np
import nurbspy as nrb


def main(show=True):
    boundary_display = dict(
        geometry_color="#21856a",
        control_points=False,
        control_polygon=False,
        curve_depth_test=False,
    )
    # Ruled surface example

    # Create the first circular arc
    O = nrb.Point(0.0, 0.0, 0.0)
    X = np.asarray([1.00, 0.00, 0.00])  # Abscissa direction
    Y = np.asarray([0.00, 1.00, 0.00])  # Ordinate direction
    R = 0.5  # Circle radius
    theta_start = 0.00  # Start angle
    theta_end = 2 * np.pi  # End angle
    nurbsCurve1 = nrb.CircularArc(O, X, Y, R, theta_start, theta_end).NurbsCurve

    # Create the second circular arc
    O = nrb.Point(0.0, 0.0, 0.0)
    X = np.asarray([1.00, 0.00, 0.00])  # Abscissa direction
    Y = np.asarray([0.00, 0.00, 1.00])  # Ordinate direction
    R = 0.5  # Circle radius
    theta_start = np.pi / 2  # Start angle
    theta_end = 2 * np.pi + np.pi / 2  # End angle
    nurbsCurve2 = nrb.CircularArc(O, X, Y, R, theta_start, theta_end).NurbsCurve

    ruledNurbsSurface = nrb.NurbsSurfaceRuled(nurbsCurve1, nurbsCurve2).NurbsSurface
    figure = nrb.Figure(
        ruledNurbsSurface,
        title="Ruled surface between crossed circles",
        display=dict(
            control_points=True,
            control_polygon=True,
            grid=True,
            isocurves=True,
            geometry_color="#0072bd",
        ),
    )
    figure.add(
        nurbsCurve1,
        name="First boundary",
        display=boundary_display,
    )
    figure.add(
        nurbsCurve2,
        name="Second boundary",
        display=boundary_display,
    )
    if show:
        figure.edit()
    return figure


if __name__ == "__main__":
    main()
