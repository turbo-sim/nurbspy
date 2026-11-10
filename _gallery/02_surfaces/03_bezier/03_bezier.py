"""Construct a polynomial Bezier surface from a grid of control points.

Display the patch and control net to illustrate how the control points shape
the surface between its boundaries.

Open one construction in the editor with a reference grid.
"""

import nurbspy as nrb


def main(show=True):
    # Polynomial Bezier surface example

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

    bezierSurface = nrb.NurbsSurface(control_points=P)
    figure = nrb.Figure(
        bezierSurface,
        title="Bézier surface",
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
