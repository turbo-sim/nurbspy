"""Span straight lines between two weighted boundary profiles.

Construct and edit one ruled surface with its control net and a reference grid.
Boundary and parameter curves illustrate the initial construction.
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
    figure = nrb.Figure(
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
        figure.add(
            ruledNurbsSurface.get_isocurve_u(float(value)),
            name=f"u={value:.2f}",
            display=isocurve_display,
        )
    for value in np.linspace(0.08, 0.92, 5):
        figure.add(
            ruledNurbsSurface.get_isocurve_v(float(value)),
            name=f"v={value:.2f}",
            display=isocurve_display,
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
