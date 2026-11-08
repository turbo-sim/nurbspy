"""Find the closest point on a NURBS surface to a given spatial point.

Compute the surface parameters of the projection and show the query point,
projected point, and connecting segment alongside the control net.
"""

import numpy as np
import nurbspy as nrb


def main(show=True):
    figures = {}
    # NURBS surface example

    # Define the control points
    n_dim, n, m = 3, 5, 4
    P = [
        [
            nrb.Point(0.0, 3.0, 0.0),
            nrb.Point(0.0, 3.0, 1.0),
            nrb.Point(0.0, 3.0, 2.0),
            nrb.Point(0.5, 3.0, 3.0),
        ],
        [
            nrb.Point(1.0, 2.0, 0.0),
            nrb.Point(1.0, 2.0, 1.0),
            nrb.Point(1.0, 2.0, 2.0),
            nrb.Point(1.0, 2.5, 3.0),
        ],
        [
            nrb.Point(2.0, 1.5, 0.0),
            nrb.Point(2.0, 1.5, 1.0),
            nrb.Point(2.0, 1.5, 2.0),
            nrb.Point(2.0, 2.0, 3.0),
        ],
        [
            nrb.Point(3.0, 2.0, 0.0),
            nrb.Point(3.0, 2.0, 1.0),
            nrb.Point(3.0, 2.0, 2.0),
            nrb.Point(3.0, 2.5, 3.0),
        ],
        [
            nrb.Point(4.0, 3.0, 0.0),
            nrb.Point(4.0, 3.0, 1.0),
            nrb.Point(4.0, 3.0, 2.0),
            nrb.Point(3.5, 3.0, 3.0),
        ],
    ]  # points[u_index][v_index]

    nurbsSurface = nrb.NurbsSurface(control_points=P)

    # Define point to be projected
    P0 = np.asarray([0.50, 0.50, 0.50]).reshape((3, 1))

    u0, v0 = nurbsSurface.project_point_to_surface(P0)
    S0 = nurbsSurface.get_value(u0, v0)

    figures["projection"] = nrb.Figure(
        nurbsSurface,
        title="Closest point on a NURBS surface",
        display=dict(
            control_points=True,
            control_polygon=True,
            grid=True,
            geometry_color="#0072bd",
        ),
    )
    # Keep the projected point and connecting segment visible in both views.
    for fig in figures.values():
        fig.add(
            nrb.NurbsCurve(np.column_stack((P0.ravel(), S0.ravel())), degree=1),
            name="Projection segment",
            display=dict(
                geometry_color="#d97823",
                control_points=False,
                control_polygon=False,
                curve_depth_test=False,
            ),
        )
        fig.scatter(np.column_stack((P0.ravel(), S0.ravel())), color="#d97823", size=7)
    if show:
        for figure in figures.values():
            figure.show()
    return figures


if __name__ == "__main__":
    main()
