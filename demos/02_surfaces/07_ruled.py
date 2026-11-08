"""Generate ruled surfaces by spanning straight lines between two curves.

Compare weighted boundary curves and crossed circular arcs, displaying the
boundaries, control nets, and parameter curves of the resulting patches.
"""

import numpy as np
import nurbspy as nrb


def main(show=True):
    figures = {}
    boundary_display = dict(
        geometry_color="#d97823",
        control_points=False,
        control_polygon=False,
        curve_depth_test=False,
    )
    boundary_display = dict(
        geometry_color="#21856a",
        control_points=False,
        control_polygon=False,
        curve_depth_test=False,
    )
    isocurve_display = dict(
        control_points=False,
        control_polygon=False,
        geometry_color="#536d7d",
        curve_depth_test=False,
    )
    # Ruled surface example

    # Define the lower NURBS curve (rational Bézier curve)
    P1 = [
        nrb.Point(0.0, 0.0, 0.0),
        nrb.Point(0.25, 0.0, 0.5),
        nrb.Point(0.5, 0.0, 0.5),
        nrb.Point(0.75, 0.0, 0.0),
        nrb.Point(1.0, 0.0, 0.0),
    ]
    W1 = np.asarray([1, 1, 2, 1, 1])
    nurbsCurve1 = nrb.NurbsCurve(control_points=P1, weights=W1)

    # Define the lower NURBS curve (rational Bézier curve)
    P2 = [
        nrb.Point(0.0, 1.0, 0.5),
        nrb.Point(0.25, 1.0, 0.0),
        nrb.Point(0.5, 1.0, 0.0),
        nrb.Point(0.75, 1.0, 0.5),
        nrb.Point(1.0, 1.0, 0.5),
    ]
    W2 = np.asarray([1, 1, 2, 1, 1])
    nurbsCurve2 = nrb.NurbsCurve(control_points=P2, weights=W2)

    ruledNurbsSurface = nrb.NurbsSurfaceRuled(nurbsCurve1, nurbsCurve2).NurbsSurface
    figures["weighted_boundaries"] = nrb.Figure(
        ruledNurbsSurface,
        title="Ruled patch between weighted boundaries",
        display=dict(
            control_points=True,
            control_polygon=True,
            grid=True,
            geometry_color="#0072bd",
        ),
    )
    for value in np.linspace(0.08, 0.92, 5):
        figures["weighted_boundaries"].add(
            ruledNurbsSurface.get_isocurve_u(float(value)),
            name=f"u={value:.2f}",
            display=isocurve_display,
        )
    for value in np.linspace(0.08, 0.92, 5):
        figures["weighted_boundaries"].add(
            ruledNurbsSurface.get_isocurve_v(float(value)),
            name=f"v={value:.2f}",
            display=isocurve_display,
        )
    figures["weighted_boundaries"].add(
        nurbsCurve1,
        name="First boundary",
        display=boundary_display,
    )
    figures["weighted_boundaries"].add(
        nurbsCurve2,
        name="Second boundary",
        display=boundary_display,
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
    figures["crossed_circles"] = nrb.Figure(
        ruledNurbsSurface,
        title="Ruled surface between crossed circles",
        display=dict(
            control_points=True,
            control_polygon=True,
            grid=True,
            geometry_color="#0072bd",
        ),
    )
    for value in np.linspace(0.08, 0.92, 5):
        figures["crossed_circles"].add(
            ruledNurbsSurface.get_isocurve_u(float(value)),
            name=f"u={value:.2f}",
            display=isocurve_display,
        )
    for value in np.linspace(0.08, 0.92, 5):
        figures["crossed_circles"].add(
            ruledNurbsSurface.get_isocurve_v(float(value)),
            name=f"v={value:.2f}",
            display=isocurve_display,
        )
    figures["crossed_circles"].add(
        nurbsCurve1,
        name="First boundary",
        display=boundary_display,
    )
    figures["crossed_circles"].add(
        nurbsCurve2,
        name="Second boundary",
        display=boundary_display,
    )
    if show:
        for figure in figures.values():
            figure.show()
    return figures


if __name__ == "__main__":
    main()
