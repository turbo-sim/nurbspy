"""Small, deterministic geometry inputs shared by numerical tests."""

import numpy as np


def rational_curve_data():
    """Return fresh cubic curve data for parameter and shape-sensitivity checks.

    Nonuniform knots and unequal weights exercise the rational quotient and
    several polynomial spans. Each call returns new arrays that tests may perturb.
    """
    # The three-dimensional control polygon gives nontrivial derivatives.
    points = np.array(
        [
            [0.0, 0.3, 0.8, 1.2, 1.7, 2.0],
            [0.0, 0.5, -0.2, 0.8, 0.3, 0.1],
            [0.0, 0.1, 0.4, -0.1, 0.6, 0.2],
        ]
    )
    weights = np.array([1.0, 0.8, 1.4, 0.9, 1.2, 1.0])
    knots = np.array([0.0, 0.0, 0.0, 0.0, 0.3, 0.7, 1.0, 1.0, 1.0, 1.0])
    return dict(control_points=points, weights=weights, degree=3, knots=knots)


def rational_surface_data():
    """Return fresh data for a nonplanar rational surface with unequal weights."""

    # Define the array of control points
    n_dim, n, m = 3, 5, 4
    P = np.zeros((n_dim, n, m))

    # First row
    P[:, 0, 0] = [0.00, 3.00, 0.00]
    P[:, 1, 0] = [1.00, 2.00, 0.00]
    P[:, 2, 0] = [2.00, 1.50, 0.00]
    P[:, 3, 0] = [3.00, 2.00, 0.00]
    P[:, 4, 0] = [4.00, 3.00, 0.00]

    # Second row
    P[:, 0, 1] = [0.00, 3.00, 1.00]
    P[:, 1, 1] = [1.00, 2.00, 1.00]
    P[:, 2, 1] = [2.00, 1.50, 1.00]
    P[:, 3, 1] = [3.00, 2.00, 1.00]
    P[:, 4, 1] = [4.00, 3.00, 1.00]

    # Third row
    P[:, 0, 2] = [0.00, 3.00, 2.00]
    P[:, 1, 2] = [1.00, 2.00, 2.00]
    P[:, 2, 2] = [2.00, 1.50, 2.00]
    P[:, 3, 2] = [3.00, 2.00, 2.00]
    P[:, 4, 2] = [4.00, 3.00, 2.00]

    # Fourth row
    P[:, 0, 3] = [0.50, 3.00, 3.00]
    P[:, 1, 3] = [1.00, 2.50, 3.00]
    P[:, 2, 3] = [2.00, 2.00, 3.00]
    P[:, 3, 3] = [3.00, 2.50, 3.00]
    P[:, 4, 3] = [3.50, 3.00, 3.00]

    # Define the array of control point weights
    W = np.zeros((n, m))
    W[:, 0] = np.asarray([1, 1, 5, 1, 1])
    W[:, 1] = np.asarray([1, 1, 10, 1, 1])
    W[:, 2] = np.asarray([1, 1, 10, 1, 1])
    W[:, 3] = np.asarray([1, 1, 5, 1, 1])

    # Maximum index of the control points (counting from zero)
    n = np.shape(P)[1] - 1
    m = np.shape(P)[2] - 1

    # Define the order of the basis polynomials
    # Linear (p = 1), Quadratic (p = 2), Cubic (p = 3), etc.
    # Set p = n (number of control points minus one) to obtain a Bezier
    p = 3
    q = 3

    # Define the knot vectors (clamped spline)
    # p+1 zeros, n-p equispaced points between 0 and 1, and p+1 ones.  In total r+1 points where r=n+p+1
    # q+1 zeros, m-q equispaced points between 0 and 1, and q+1 ones. In total s+1 points where s=m+q+1
    U = np.concatenate((np.zeros(p), np.linspace(0, 1, n - p + 2), np.ones(p)))
    V = np.concatenate((np.zeros(q), np.linspace(0, 1, m - q + 2), np.ones(q)))

    # Return constructor arguments so each test builds its own surface.
    return dict(
        control_points=P, weights=W, u_degree=p, v_degree=q, u_knots=U, v_knots=V
    )
