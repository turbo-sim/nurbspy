"""Check analytical NURBS curve derivatives against finite differences.

Compute first and second derivatives of a weighted spatial curve, report their
errors, and display the curve with its control polygon and Frenet-Serret frames.
"""

import numpy as np
import nurbspy as nrb


def main(show=True):
    figures = {}
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
    W = np.asarray([1, 1, 3, 1, 1])

    # Define the order of the basis polynomials
    p = 3

    # Define the knot vector (clamped spline)
    # p+1 zeros, n-p equispaced points between 0 and 1, and p+1 ones. In total r+1 points where r=n+p+1
    U = np.concatenate((np.zeros(p), np.linspace(0, 1, n - p + 2), np.ones(p)))

    my_nurbs = nrb.NurbsCurve(control_points=P, weights=W, degree=p, knots=U)
    figures["curve"] = nrb.Figure(
        my_nurbs,
        title="Native derivative verification",
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

    # Check the analytic derivatives against a finite difference aproximation

    # Define a u-parametrization suitable for finite differences
    h = 1e-4
    hh = 2 * (h + h**2)
    Nu = 1000
    u = np.linspace(0.00 + hh, 1.00 - hh, Nu)

    # Compute the NURBS derivatives analytically
    dC_exact = my_nurbs.get_derivative(u, order=1)
    ddC_exact = my_nurbs.get_derivative(u, order=2)
    dddC_exact = my_nurbs.get_derivative(u, order=3)

    # Approximate the NURBS derivatives by central finite differences
    dC_fd = (my_nurbs.get_value(u + h) - my_nurbs.get_value(u - h)) / (2 * h)
    ddC_fd = (
        my_nurbs.get_value(u + h)
        - 2 * my_nurbs.get_value(u)
        + my_nurbs.get_value(u - h)
    ) / (h**2)
    dddC_fd = (
        -1 / 2 * my_nurbs.get_value(u - 2 * h)
        + my_nurbs.get_value(u - h)
        - my_nurbs.get_value(u + h)
        + 1 / 2 * my_nurbs.get_value(u + 2 * h)
    ) / (h**3)

    # Print the results
    print(
        "The two-norm of the first derivative error is   :  ",
        np.sum((dC_fd - dC_exact) ** 2) ** (1 / 2) / len(u),
    )
    print(
        "The two-norm of the second derivative error is  :  ",
        np.sum((ddC_fd - ddC_exact) ** 2) ** (1 / 2) / len(u),
    )
    print(
        "The two-norm of the third derivative error is   :  ",
        np.sum((dddC_fd - dddC_exact) ** 2) ** (1 / 2) / len(u),
    )
    if show:
        for figure in figures.values():
            figure.show()
    return figures


if __name__ == "__main__":
    main()
