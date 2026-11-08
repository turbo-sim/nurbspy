"""Construct planar and spatial NURBS curves with nonuniform weights.

Define the control points, weights, degrees, and knots explicitly, then show
the resulting shapes with their control polygons and Frenet-Serret frames.
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

    # Maximum index of the control points (counting from zero)
    n = len(P) - 1

    # Define the array of control point weights
    W = np.asarray([1, 1, 3, 1, 1])

    # Define the order of the basis polynomials
    # Linear (p = 1), Quadratic (p = 2), Cubic (p = 3), etc.
    # Set p = n (number of control points minus one) to obtain a Bezier
    p = 2

    # Define the knot vector (clamped spline)
    # p+1 zeros, n-p equispaced points between 0 and 1, and p+1 ones. In total r+1 points where r=n+p+1
    U = np.concatenate((np.zeros(p), np.linspace(0, 1, n - p + 2), np.ones(p)))

    nurbs2D = nrb.NurbsCurve(control_points=P, weights=W, degree=p, knots=U)
    figures["planar"] = nrb.Figure(
        nurbs2D,
        title="Planar NURBS curve",
        display=dict(
            view="xy",
            control_points=True,
            control_polygon=True,
            grid=False,
            geometry_color="#0072bd",
            frames=True,
            frame_count=6,
            frame_scale=0.06,
        ),
    )

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

    # Define the order of the basis polynomials
    # Linear (p = 1), Quadratic (p = 2), Cubic (p = 3), etc.
    # Set p = n (number of control points minus one) to obtain a Bezier
    p = 2

    # Define the knot vector (clamped spline)
    # p+1 zeros, n-p equispaced points between 0 and 1, and p+1 ones. In total r+1 points where r=n+p+1
    U = np.concatenate((np.zeros(p), np.linspace(0, 1, n - p + 2), np.ones(p)))

    # Create the NURBS curve
    nurbs3D = nrb.NurbsCurve(control_points=P, weights=W, degree=p, knots=U)
    figures["spatial"] = nrb.Figure(
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
        for figure in figures.values():
            figure.show()
    return figures


if __name__ == "__main__":
    main()
