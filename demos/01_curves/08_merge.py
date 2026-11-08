"""Join compatible NURBS curve segments into one curve.

Define three connected segments, merge them, and compare the input segments
and joined curve with their respective control polygons.
"""

import numpy as np
import nurbspy as nrb


def main(show=True):
    figures = {}
    # Create the first NURBS curve

    # Define the control points
    P = [
        nrb.Point(0.0, 0.0),
        nrb.Point(0.1, 0.5),
        nrb.Point(0.2, -0.5),
        nrb.Point(0.3, 0.0),
        nrb.Point(0.4, 0.1),
        nrb.Point(0.5, 0.2),
        nrb.Point(0.6, 0.3),
        nrb.Point(0.75, 0.0),
    ]

    # Define the array of control weights
    W = np.asarray([1, 1, 1, 1, 2, 2, 2, 2])

    # Highest index of the control points (counting from zero)
    n = len(P) - 1

    # Define the order of the basis polynomials
    p = 3

    # Define the knot vector (clamped spline)
    # p+1 zeros, n-p equispaced points between 0 and 1, and p+1 ones.  In total r+1 points where r=n+p+1
    U = np.concatenate((np.zeros(p), np.linspace(0, 1, n - p + 2), np.ones(p)))

    # Create the B-Spline curve
    nurbs1 = nrb.NurbsCurve(control_points=P, weights=W, degree=p, knots=U)

    # Create the second NURBS curve

    # Define the control points
    P = [
        nrb.Point(0.75, 0.0),
        nrb.Point(1.0, 0.5),
        nrb.Point(1.25, -0.5),
        nrb.Point(1.5, 0.0),
    ]

    # Define the array of control weights
    W = np.asarray([5, 2, 2, 4])

    # Highest index of the control points (counting from zero)
    n = len(P) - 1

    # Define the order of the basis polynomials
    p = 3

    # Define the knot vector (clamped spline)
    # p+1 zeros, n-p equispaced points between 0 and 1, and p+1 ones.  In total r+1 points where r=n+p+1
    U = np.concatenate((np.zeros(p), np.linspace(0, 1, n - p + 2), np.ones(p)))

    # Create the B-Spline curve
    nurbs2 = nrb.NurbsCurve(control_points=P, weights=W, degree=p, knots=U)

    # Create the third NURBS curve

    # Define the control points
    P = [
        nrb.Point(1.5, 0.0),
        nrb.Point(2.0, 0.5),
        nrb.Point(1.5, 0.75),
        nrb.Point(1.0, 0.6),
        nrb.Point(0.5, 0.3),
    ]

    # Define the array of control weights
    W = np.asarray([1, 1, 1, 1, 1])

    # Highest index of the control points (counting from zero)
    n = len(P) - 1

    # Define the order of the basis polynomials
    p = 3

    # Define the knot vector (clamped spline)
    # p+1 zeros, n-p equispaced points between 0 and 1, and p+1 ones.  In total r+1 points where r=n+p+1
    U = np.concatenate((np.zeros(p), np.linspace(0, 1, n - p + 2), np.ones(p)))

    # Create the B-Spline curve
    nurbs3 = nrb.NurbsCurve(control_points=P, weights=W, degree=p, knots=U)

    # Merge the NURBS curves

    nurbsMerged = nrb.merge_nurbs_curves([nurbs1, nurbs2, nurbs3])
    figures["segments"] = nrb.Figure(
        nurbs1,
        title="Three compatible curve segments",
        display=dict(
            view="xy",
            control_points=True,
            control_polygon=True,
            grid=False,
            geometry_color="#0072bd",
        ),
    )
    figures["segments"].add(
        nurbs2,
        name="Second segment",
        display=dict(
            geometry_color="#d97823",
            control_points=False,
            control_polygon=False,
            curve_depth_test=False,
        ),
    )
    figures["segments"].add(
        nurbs3,
        name="Third segment",
        display=dict(
            geometry_color="#21856a",
            control_points=False,
            control_polygon=False,
            curve_depth_test=False,
        ),
    )
    for name in figures["segments"].names:
        figures["segments"].update(
            name, display=dict(control_points=True, control_polygon=True)
        )
    figures["merged"] = nrb.Figure(
        nurbsMerged,
        title="Merged NURBS curve",
        display=dict(
            view="xy",
            control_points=True,
            control_polygon=True,
            grid=False,
            geometry_color="#0072bd",
        ),
    )
    if show:
        for figure in figures.values():
            figure.show()
    return figures


if __name__ == "__main__":
    main()
