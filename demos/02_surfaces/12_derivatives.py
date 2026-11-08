"""Check analytical NURBS surface derivatives against finite differences.

Compare first, second, and mixed partial derivatives on a weighted surface
and display the patch with its control net.
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

    # Define the array of control point weights
    W = np.zeros((n, m))
    W[:, 0] = np.asarray([1, 1, 5, 1, 1])
    W[:, 1] = np.asarray([1, 1, 10, 1, 1])
    W[:, 2] = np.asarray([1, 1, 10, 1, 1])
    W[:, 3] = np.asarray([1, 1, 5, 1, 1])

    # Maximum index of the control points (counting from zero)
    n = len(P) - 1
    m = len(P[0]) - 1

    # Define the order of the basis polynomials
    # Linear (p = 1), Quadratic (p = 2), Cubic (p = 3), etc.
    # Set p = n (number of control points minus one) to obtain a Bezier
    p = 3
    q = 3

    # Define the knot vectors (clamped spline)
    # p+1 zeros, n-p equispaced points between 0 and 1, and p+1 ones.  In total r+1 points where r=n+p+1
    # q+1 zeros, m-p equispaced points between 0 and 1, and q+1 ones. In total s+1 points where s=m+q+1
    U = np.concatenate((np.zeros(p), np.linspace(0, 1, n - p + 2), np.ones(p)))
    V = np.concatenate((np.zeros(q), np.linspace(0, 1, m - q + 2), np.ones(q)))

    nurbsSurface = nrb.NurbsSurface(
        control_points=P, weights=W, u_degree=p, v_degree=q, u_knots=U, v_knots=V
    )
    figures["surface"] = nrb.Figure(
        nurbsSurface,
        title="Native derivative verification",
        display=dict(
            control_points=True,
            control_polygon=True,
            grid=True,
            geometry_color="#0072bd",
        ),
    )
    S_func = nurbsSurface.get_value

    # Check the analytic derivatives against a finite difference aproximation

    # Define a (u,v) parametrization suitable for finite differences. 1D arrays of (u,v) query points
    h = 1e-4
    hh = h + h**2
    Nu, Nv = 50, 50
    u = np.linspace(0.00 + h, 1.00 - h, Nu)
    v = np.linspace(0.00 + h, 1.00 - h, Nv)
    [u, v] = np.meshgrid(u, v, indexing="xy")
    u = u.flatten()
    v = v.flatten()

    # Derivative (0,0)
    S = S_func(u, v)
    dS_exact = nurbsSurface.get_derivative(u, v, order_u=0, order_v=0)
    print(
        "Derivative (0,0) two-norm error is: ",
        np.real(np.sum((dS_exact - S) ** 2) ** (1 / 2) / (Nu * Nv)),
    )

    # Derivative (1,0)
    dS_exact = nurbsSurface.get_derivative(u, v, order_u=1, order_v=0)
    dS_CFD = (S_func(u + h, v) - S_func(u - h, v)) / (2 * h)
    print(
        "Derivative (1,0) two-norm error is: ",
        np.real(np.sum((dS_exact - dS_CFD) ** 2) ** (1 / 2) / (Nu * Nv)),
    )

    # Derivative (0,1)
    dS_exact = nurbsSurface.get_derivative(u, v, order_u=0, order_v=1)
    dS_CFD = (S_func(u, v + h) - S_func(u, v - h)) / (2 * h)
    print(
        "Derivative (0,1) two-norm error is: ",
        np.real(np.sum((dS_exact - dS_CFD) ** 2) ** (1 / 2) / (Nu * Nv)),
    )

    # Derivative (1,1)
    dS_exact = nurbsSurface.get_derivative(u, v, order_u=1, order_v=1)
    dS_CFD = (
        S_func(u + h, v + h)
        - S_func(u - h, v + h)
        - S_func(u + h, v - h)
        + S_func(u - h, v - h)
    ) / (4 * h**2)
    print(
        "Derivative (1,1) two-norm error is: ",
        np.real(np.sum((dS_exact - dS_CFD) ** 2) ** (1 / 2) / (Nu * Nv)),
    )

    # Derivative (2,0)
    dS_exact = nurbsSurface.get_derivative(u, v, order_u=2, order_v=0)
    dS_CFD = (S_func(u + h, v) - 2 * S_func(u, v) + S_func(u - h, v)) / (h**2)
    print(
        "Derivative (2,0) two-norm error is: ",
        np.real(np.sum((dS_exact - dS_CFD) ** 2) ** (1 / 2) / (Nu * Nv)),
    )

    # Derivative (0,2)
    dS_exact = nurbsSurface.get_derivative(u, v, order_u=0, order_v=2)
    dS_CFD = (S_func(u, v + h) - 2 * S_func(u, v) + S_func(u, v - h)) / (h**2)
    print(
        "Derivative (0,2) two-norm error is: ",
        np.real(np.sum((dS_exact - dS_CFD) ** 2) ** (1 / 2) / (Nu * Nv)),
    )
    if show:
        for figure in figures.values():
            figure.show()
    return figures


if __name__ == "__main__":
    main()
