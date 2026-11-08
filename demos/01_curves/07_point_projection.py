"""Find the closest point on a planar NURBS curve to a given point.

Project the point onto the curve and display both points and their connecting
segment together with the control polygon.
"""

import numpy as np
import nurbspy as nrb


def main(show=True):
    figures = {}
    # 2D NURBS curve example

    # Define the control points
    P = [
        nrb.Point(0.2, 0.5),
        nrb.Point(0.4, 0.7),
        nrb.Point(0.8, 0.6),
        nrb.Point(0.6, 0.2),
        nrb.Point(0.4, 0.2),
    ]

    # Create the NURBS curve
    nurbs2D = nrb.NurbsCurve(control_points=P, degree=3)

    # Define point to be projected
    P0 = np.asarray([0.50, 0.50])

    u0 = nurbs2D.project_point_to_curve(P0)
    C0 = nurbs2D.get_value(u0)

    figures["planar"] = nrb.Figure(
        nurbs2D,
        title="Closest point on a NURBS curve",
        display=dict(
            view="xy",
            control_points=True,
            control_polygon=True,
            grid=False,
            geometry_color="#0072bd",
        ),
    )
    # Original and closest points, with their projection segment.
    for fig in figures.values():
        fig.add(
            nrb.NurbsCurve(np.column_stack((P0, C0)), degree=1),
            name="Projection segment",
            display=dict(
                geometry_color="#d97823",
                control_points=False,
                control_polygon=False,
                curve_depth_test=False,
            ),
        )
        fig.scatter(np.column_stack((P0, C0)), color="#d97823", size=7)
    if show:
        for figure in figures.values():
            figure.show()
    return figures


if __name__ == "__main__":
    main()
