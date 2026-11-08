"""Illustrate how weights affect a rational Bezier surface.

Assign nonuniform weights to a control net and show the resulting surface
together with the control points that determine its shape.
"""

import numpy as np
import nurbspy as nrb


def main(show=True):
    figures = {}
    # Rational Bezier surface example

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

    # Define the array of control point weights
    W = np.zeros((n, m))
    W[:, 0] = np.asarray([1, 1, 5, 1, 1])
    W[:, 1] = np.asarray([1, 1, 5, 1, 1])
    W[:, 2] = np.asarray([1, 1, 5, 1, 1])
    W[:, 3] = np.asarray([1, 1, 0.5, 1, 1])

    bezierSurface = nrb.NurbsSurface(control_points=P, weights=W)
    figures["surface"] = nrb.Figure(
        bezierSurface,
        title="Rational Bézier surface",
        display=dict(
            control_points=True,
            control_polygon=True,
            grid=True,
            geometry_color="#0072bd",
        ),
    )
    if show:
        for figure in figures.values():
            figure.show()
    return figures


if __name__ == "__main__":
    main()
