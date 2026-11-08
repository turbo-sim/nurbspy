"""Join three compatible NURBS segments and edit the resulting curve.

Define the connected segments and merge them into one curve. Display only the
joined curve and its control polygon on a reference grid.
"""

import numpy as np
import nurbspy as nrb


def main(show=True):
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

    # Define the degree of the basis polynomials
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

    # Define the degree of the basis polynomials
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

    # Define the degree of the basis polynomials
    p = 3

    # Define the knot vector (clamped spline)
    # p+1 zeros, n-p equispaced points between 0 and 1, and p+1 ones.  In total r+1 points where r=n+p+1
    U = np.concatenate((np.zeros(p), np.linspace(0, 1, n - p + 2), np.ones(p)))

    # Create the B-Spline curve
    nurbs3 = nrb.NurbsCurve(control_points=P, weights=W, degree=p, knots=U)

    # Merge the NURBS curves

    nurbsMerged = nrb.merge_nurbs_curves([nurbs1, nurbs2, nurbs3])
    figure = nrb.Figure(
        nurbsMerged,
        title="Merged NURBS curve",
        display=dict(
            view="xy",
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
