"""Construct and edit one spatial NURBS curve.

Define its control points and display the control polygon on a reference grid.
Frenet-Serret frames illustrate the local curve directions.
"""

import numpy as np
import nurbspy as nrb


def main(show=True):
    # 3D NURBS curve example

    # Define the control points
    P = [
        nrb.Point(0.0, 0.0, 0.0),
        nrb.Point(0.0, 0.3, 0.05),
        nrb.Point(0.25, 0.3, 0.3),
        nrb.Point(0.5, 0.3, -0.05),
        nrb.Point(0.5, 0.1, 0.1),
    ]

    # Maximum index of the control points (counting from zero)
    n = len(P) - 1

    # Define the array of control point weights
    W = np.asarray([1, 1, 0.5, 1, 1])

    # Define the degree of the basis polynomials
    # Linear (p = 1), Quadratic (p = 2), Cubic (p = 3), etc.
    # Set p = n (number of control points minus one) to obtain a Bezier
    p = 2

    # Define the knot vector (clamped spline)
    # p+1 zeros, n-p equispaced points between 0 and 1, and p+1 ones. In total r+1 points where r=n+p+1
    U = np.concatenate((np.zeros(p), np.linspace(0, 1, n - p + 2), np.ones(p)))

    # Create the NURBS curve
    nurbs3D = nrb.NurbsCurve(control_points=P, weights=W, degree=p, knots=U)
    figure = nrb.Figure(
        nurbs3D,
        title="Spatial NURBS curve",
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
