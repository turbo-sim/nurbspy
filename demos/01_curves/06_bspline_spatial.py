"""Construct and edit one spatial B-spline curve.

Define its control points and display the control polygon on a reference grid.
Frenet-Serret frames illustrate the local curve directions.
"""

import numpy as np
import nurbspy as nrb


def main(show=True):
    # 3D B-Spline curve example

    # Define the control points
    P = [
        nrb.Point(0.0, 0.0, 0.0),
        nrb.Point(0.0, 0.3, 0.05),
        nrb.Point(0.25, 0.3, 0.3),
        nrb.Point(0.5, 0.3, -0.05),
        nrb.Point(0.5, 0.1, 0.1),
    ]

    # Define the degree of the basis polynomials
    p = 2

    bspline3D = nrb.NurbsCurve(control_points=P, degree=p)
    figure = nrb.Figure(
        bspline3D,
        title="Spatial B-spline curve",
        display=dict(
            control_points=True,
            control_polygon=True,
            grid=True,
            geometry_color="#0072bd",
            frames=True,
            frame_count=6,
            frame_scale=0.06,
        ),
    )
    if show:
        figure.edit()
    return figure


if __name__ == "__main__":
    main()
