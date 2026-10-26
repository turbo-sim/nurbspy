"""Unit tests for the basis polynomial computations"""

# -------------------------------------------------------------------------------------------------------------------- #
# Importing packages
# -------------------------------------------------------------------------------------------------------------------- #
import numpy as np
from numpy.testing import assert_allclose


# -------------------------------------------------------------------------------------------------------------------- #
# Prepare the test suite
# -------------------------------------------------------------------------------------------------------------------- #
def test_basis_function_example_1(basis_backend):
    """Test the basis function value against a known example (Ex2.2 from the NURBS book)"""

    # Compute the basis polynomials of a known problem
    def get_analytic_polynomials(u):
        """Evaluate the piecewise quadratic polynomials given in Example 2.2."""

        N = np.zeros((8, u.size), dtype=u.dtype)
        for i, u in enumerate(u):
            N02 = (1 - u) ** 2 * (0 <= u < 1)
            N12 = (2 * u - 3 / 2 * u**2) * (0 <= u < 1) + (1 / 2 * (2 - u) ** 2) * (
                1 <= u < 2
            )
            N22 = (
                (1 / 2 * u**2) * (0 <= u < 1)
                + (-3 / 2 + 3 * u - u**2) * (1 <= u < 2)
                + (1 / 2 * (3 - u) ** 2) * (2 <= u < 3)
            )
            N32 = (
                (1 / 2 * (u - 1) ** 2) * (1 <= u < 2)
                + (-11 / 2 + 5 * u - u**2) * (2 <= u < 3)
                + (1 / 2 * (4 - u) ** 2) * (3 <= u < 4)
            )
            N42 = (1 / 2 * (u - 2) ** 2) * (2 <= u < 3) + (
                -16 + 10 * u - 3 / 2 * u**2
            ) * (3 <= u < 4)
            N52 = (u - 3) ** 2 * (3 <= u < 4) + (5 - u) ** 2 * (4 <= u < 5)
            N62 = (2 * (u - 4) * (5 - u)) * (4 <= u < 5)
            N72 = (u - 4) ** 2 * (4 <= u <= 5)
            N[:, i] = np.asarray([N02, N12, N22, N32, N42, N52, N62, N72])

        return N

    # Maximum index of the basis polynomials (counting from zero)
    n = 7

    # Define the order of the basis polynomials
    p = 2

    # Define the knot vector (clamped spline)
    # p+1 zeros, n-p equispaced points between 0 and 1, and p+1 ones. In total r+1 points where r=n+p+1
    #                 u0    u1    u2    u3    u4    u5    u6    u7    u8    u9   u10
    U = np.asarray([0.00, 0.00, 0.00, 1.00, 2.00, 3.00, 4.00, 4.00, 5.00, 5.00, 5.00])

    # Define the u-parametrization
    uu = np.linspace(0, 5, 21)

    # Evaluate the polynomials numerically
    N_basis = basis_backend.compute_basis_polynomials(n, p, U, uu)

    # Evaluate the polynomials numerically
    N_analytic = get_analytic_polynomials(uu)

    # Check the error
    error = np.sum((N_analytic - N_basis) ** 2) ** (1 / 2)
    print("The two-norm of the evaluation error is         :  ", error)
    assert_allclose(N_basis, N_analytic, rtol=1e-12, atol=1e-12)


def test_partition_of_unity_property(basis_backend):
    """Test the partition of unity property of the basis polynomials"""

    # Maximum index of the basis polynomials (counting from zero)
    n = 4

    # Define the order of the basis polynomials
    p = 3

    # Define the knot vector (clamped spline)
    # p+1 zeros, n-p equispaced points between 0 and 1, and p+1 ones. In total r+1 points where r=n+p+1
    U = np.concatenate((np.zeros(p), np.linspace(0, 1, n - p + 2), np.ones(p)))

    # Define the u-parametrization
    u = np.linspace(0, 1, 101)

    # Compute the basis polynomials derivatives analytically
    N_basis = basis_backend.compute_basis_polynomials(n, p, U, u)

    # Check the error
    error = np.sum((np.sum(N_basis, axis=0) - 1) ** 2) ** (1 / 2)
    print("The two-norm of the partition of unity error is :  ", error)
    assert_allclose(np.sum(N_basis, axis=0), 1.0, rtol=0, atol=1e-12)


def test_basis_function_zeroth_derivative(basis_backend):
    """Test that the zero-th derivative agrees with the function evaluation"""

    # Maximum index of the basis polynomials (counting from zero)
    n = 4

    # Define the order of the basis polynomials
    p = 3

    # Define the knot vector (clamped spline)
    # p+1 zeros, n-p equispaced points between 0 and 1, and p+1 ones. In total r+1 points where r=n+p+1
    U = np.concatenate((np.zeros(p), np.linspace(0, 1, n - p + 2), np.ones(p)))

    # Define the u-parametrization
    u = np.linspace(0, 1, 101)

    # Compute the basis polynomials derivatives analytically
    N_basis = basis_backend.compute_basis_polynomials(n, p, U, u)
    dN_basis = basis_backend.compute_basis_polynomials_derivatives(
        n, p, U, u, derivative_order=0
    )

    # Check the error
    error = np.sum((dN_basis - N_basis) ** 2) ** (1 / 2)
    print("The two-norm of the zeroth derivative error is  :  ", error)
    assert_allclose(dN_basis, N_basis, rtol=1e-12, atol=1e-12)


def test_basis_function_first_derivative_cfd(basis_backend):
    """Test the first derivative of the basis polynomials against central finite differences"""

    # Maximum index of the basis polynomials (counting from zero)
    n = 4

    # Define the order of the basis polynomials
    p = 3

    # Define the knot vector (clamped spline)
    # p+1 zeros, n-p equispaced points between 0 and 1, and p+1 ones. In total r+1 points where r=n+p+1
    U = np.concatenate((np.zeros(p), np.linspace(0, 1, n - p + 2), np.ones(p)))

    # Define a new u-parametrization suitable for finite differences
    h = 1e-5
    hh = h + h**2
    Nu = 1000
    u = np.linspace(
        0.00 + hh, 1.00 - hh, Nu
    )  # Make sure that the limits [0, 1] also work when making changes

    # Compute the basis polynomials derivatives analytically
    dN_basis = basis_backend.compute_basis_polynomials_derivatives(
        n, p, U, u, derivative_order=1
    )

    # Compute the basis polynomials derivatives by central finite differences
    a = -1 / 2 * basis_backend.compute_basis_polynomials(n, p, U, u - h)
    b = +1 / 2 * basis_backend.compute_basis_polynomials(n, p, U, u + h)
    dN_fd = (a + b) / h

    # Check the error
    error = np.sum((dN_basis - dN_fd) ** 2) ** (1 / 2)
    print("The two-norm of the first derivative error is   :  ", error)
    assert_allclose(dN_basis, dN_fd, rtol=1e-7, atol=1e-8)


def test_basis_function_second_derivative_cfd(basis_backend):
    """Test the second derivative of the basis polynomials against central finite differences"""

    # Maximum index of the basis polynomials (counting from zero)
    n = 4

    # Define the order of the basis polynomials
    p = 3

    # Define the knot vector (clamped spline)
    # p+1 zeros, n-p equispaced points between 0 and 1, and p+1 ones. In total r+1 points where r=n+p+1
    U = np.concatenate((np.zeros(p), np.linspace(0, 1, n - p + 2), np.ones(p)))

    # Define a new u-parametrization suitable for finite differences
    h = 1e-4
    hh = h + h**2
    Nu = 1000
    u = np.linspace(
        0.00 + hh, 1.00 - hh, Nu
    )  # Make sure that the limits [0, 1] also work when making changes

    # Compute the basis polynomials derivatives
    ddN_basis = basis_backend.compute_basis_polynomials_derivatives(
        n, p, U, u, derivative_order=2
    )

    # Check the second derivative against central finite differences
    a = +1 * basis_backend.compute_basis_polynomials(n, p, U, u - h)
    b = -2 * basis_backend.compute_basis_polynomials(n, p, U, u)
    c = +1 * basis_backend.compute_basis_polynomials(n, p, U, u + h)
    ddN_fd = (a + b + c) / h**2

    # Check the error
    error = np.sum((ddN_basis - ddN_fd) ** 2) ** (1 / 2)
    print("The two-norm of the second derivative error is  :  ", error)
    assert_allclose(ddN_basis, ddN_fd, rtol=1e-6, atol=1e-6)
