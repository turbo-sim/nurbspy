"""Construct a B-spline surface using explicit degrees and clamped knots.

Display its control net, exact parameter curves, and surface normals, while
colouring the patch by mean curvature.

Open one construction in the editor with a reference grid.
Isocurves follow edits. Other annotations illustrate the initial construction.
"""

import numpy as np
import nurbspy as nrb


def main(show=True):
    # B-Spline surface example

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

    # Maximum index of the control points (counting from zero)
    n = len(P) - 1
    m = len(P[0]) - 1

    # Define the degree of the basis polynomials
    # Linear (p = 1), Quadratic (p = 2), Cubic (p = 3), etc.
    # Set p = n (number of control points minus one) to obtain a Bezier
    p = 3
    q = 3

    # Define the knot vectors (clamped spline)
    # p+1 zeros, n-p equispaced points between 0 and 1, and p+1 ones.  In total r+1 points where r=n+p+1
    # q+1 zeros, m-p equispaced points between 0 and 1, and q+1 ones. In total s+1 points where s=m+q+1
    U = np.concatenate((np.zeros(p), np.linspace(0, 1, n - p + 2), np.ones(p)))
    V = np.concatenate((np.zeros(q), np.linspace(0, 1, m - q + 2), np.ones(q)))

    bsplineSurface = nrb.NurbsSurface(
        control_points=P, u_degree=p, v_degree=q, u_knots=U, v_knots=V
    )
    figure = nrb.Figure(
        bsplineSurface,
        title="B-spline surface: mean curvature",
        display=dict(
            control_points=True,
            control_polygon=True,
            grid=True,
            isocurves=True,
            geometry_color="#0072bd",
        ),
    )
    figure.update(display=dict(quantity="mean_curvature", colormap="viridis"))
    # These normal segments illustrate the initial surface.
    # Draw short line segments in the surface-normal directions.
    u, v = np.meshgrid(np.linspace(0.15, 0.85, 3), np.linspace(0.15, 0.85, 3))
    points = bsplineSurface.get_value(u.ravel(), v.ravel())
    normals = bsplineSurface.get_normals(u.ravel(), v.ravel())
    for index, (point, normal) in enumerate(zip(points.T, normals.T)):
        segment = nrb.NurbsCurve(
            np.column_stack((point, point + 0.4 * normal)), degree=1
        )
        figure.add(
            segment,
            name=f"Normal {index}",
            display=dict(
                control_points=False,
                control_polygon=False,
                geometry_color="#d97823",
                curve_depth_test=False,
            ),
        )
    if show:
        figure.edit()
    return figure


if __name__ == "__main__":
    main()
