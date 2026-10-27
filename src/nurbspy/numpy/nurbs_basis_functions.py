# -------------------------------------------------------------------------------------------------------------------- #
# Import packages
# -------------------------------------------------------------------------------------------------------------------- #
import numpy as np

try:
    from numba import jit as _jit
except ImportError as exc:
    # Numba is optional, including when an installed version cannot import
    # because it does not support the active NumPy version.
    if exc.name != "numba":
        import warnings as _warnings

        _warnings.warn(
            f"Numba could not be imported; using the NumPy implementation. {exc}",
            RuntimeWarning,
            stacklevel=2,
        )

    def _jit(**kwargs):
        def decorator(function):
            return function

        return decorator


# -------------------------------------------------------------------------------------------------------------------- #
# Compute the basis polynomials
# -------------------------------------------------------------------------------------------------------------------- #
@_jit(nopython=True, cache=True)
def compute_basis_polynomials(n, p, U, u, return_degree=None):

    """ Evaluate the n-th B-Spline basis polynomials of degree ´p´ for the input u-parametrization

    The basis polynomials are computed from their definition by implementing equation 2.5 directly

    Parameters
    ----------
    n : integer
        Highest index of the basis polynomials (n+1 basis polynomials)

    p : integer
        Degree of the basis polynomials

    U : ndarray with shape (r+1=n+p+2,)
        Knot vector of the basis polynomials
        Set the multiplicity of the first and last entries equal to ´p+1´ to obtain a clamped spline

    u : scalar or ndarray with shape (Nu,)
        Parameter used to evaluate the basis polynomials (real or complex!)

    return_degree : int
        Degree of the returned basis polynomials

    Returns
    -------
    N : ndarray with shape (n+1, Nu)
        Array containing the basis polynomials of order ´p´ evaluated at ´u´
        The first dimension of ´N´ spans the n-th polynomials
        The second dimension of ´N´ spans the ´u´ parametrization sample points

    """

    # # Check the degree of the basis basis polynomials (not possible to compile with Numba yet)
    # if p < 0: raise Exception('The degree of the basis polynomials cannot be negative')
    #
    # # Check the number of basis basis polynomials (not possible to compile with Numba yet)
    # if p > n: raise Exception('The degree of the basis polynomials must be equal or lower than the number of basis polynomials')

    # Number of points where the polynomials are evaluated (vectorized computations)
    u = np.asarray(u * 1.0)     # Convert to array of floats
    Nu = u.size

    # Number of basis polynomials at the current step of the recursion
    m = n + p + 1

    # Initialize the array of basis polynomials
    N = np.zeros((p + 1, m, Nu), dtype=u.dtype)

    # First step of the recursion formula (p = 0)
    # The case when point_index==n and u==1 is an special case. See the NURBS book section 2.5 and algorithm A2.1
    # The np.real() operator is used such that the function extends to complex u-parameter as well
    for i in range(m):
        # N[0, i, :] = 0.0 + 1.0 * (u >= U[i]) * (u < U[i + 1])
        # N[0, i, :] = 0.0 + 1.0 * (u >= U[i]) * (u < U[i + 1]) + 1.00 * (np.logical_and(u == 1, i == n))
        N[0, i, :] = 0.0 + 1.0 * (np.real(u) >= U[i]) * (np.real(u) < U[i + 1]) + 1.00 * (np.logical_and(np.real(u) == U[-1], i == n))

    # Second and next steps of the recursion formula (p = 1, 2, ...)
    for k in range(1, p + 1):

        # Update the number of basis polynomials
        m = m - 1

        # Compute the basis polynomials using the de Boor recursion formula
        for i in range(m):

            # Compute first factor (avoid division by zero by convention)
            if (U[i + k] - U[i]) == 0:
                n1 = np.zeros((Nu,), dtype=u.dtype)
            else:
                n1 = (u - U[i]) / (U[i + k] - U[i]) * N[k - 1, i, :]

            # Compute second factor (avoid division by zero by convention)
            if (U[i + k + 1] - U[i + 1]) == 0:
                n2 = np.zeros((Nu,), dtype=u.dtype)
            else:
                n2 = (U[i + k + 1] - u) / (U[i + k + 1] - U[i + 1]) * N[k - 1, i + 1, :]

            # Compute basis polynomial (recursion formula 2.5)
            N[k, i, ...] = n1 + n2

    # Get the n+1 basis polynomials of the desired degree
    N = N[p, 0:n+1, :] if return_degree is None else N[return_degree, 0:n+1, :]


    # # Numba assertions
    # assert p >= 0    # The degree of the basis polynomials cannot be negative
    # assert p <= n    # The degree of the basis polynomials must be equal or lower than the number of basis polynomials

    return N



# -------------------------------------------------------------------------------------------------------------------- #
# Compute the derivatives of the basis polynomials
# -------------------------------------------------------------------------------------------------------------------- #
@_jit(nopython=True, cache=True)
def compute_basis_polynomials_derivatives(n, p, U, u, derivative_order):

    """ Evaluate the derivative of the n-th B-Spline basis polynomials of degree ´p´ for the input u-parametrization

    The basis polynomial derivatives use the degree recurrence in equations
    2.7 and 2.9, evaluated iteratively to avoid cached recursive Numba calls.

    Parameters
    ----------
    n : integer
        Highest index of the basis polynomials (n+1 basis polynomials)

    p : integer
        Degree of the original basis polynomials

    U : ndarray with shape (r+1=n+p+2,)
        Knot vector of the basis polynomials
        Set the multiplicity of the first and last entries equal to ´p+1´ to obtain a clamped spline

    u : scalar or ndarray with shape (Nu,)
        Parameter used to evaluate the basis polynomials

    derivative_order : scalar
        Order of the basis polynomial derivatives

    Returns
    -------
    N_ders : ndarray with shape (n+1, Nu)
        Array containing the basis spline polynomials derivatives evaluated at ´u´
        The first dimension of ´N´ spans the n-th polynomials
        The second dimension of ´N´ spans the ´u´ parametrization sample points

    """

    # Number of points where the polynomials are evaluated (vectorized computations)
    u = np.asarray(u * 1.0)     # Convert to array of floats
    Nu = u.size

    # A polynomial of degree p has zero derivatives above p on every span.
    # Reject a negative starting degree: the former recursive implementation
    # used invalid indices in Python and unchecked native memory in Numba.
    # Retain the usual sample axis and dtype, including complex-step inputs.
    if derivative_order > p:
        return np.zeros((n + 1, Nu), dtype=u.dtype)

    if derivative_order <= 0:
        return compute_basis_polynomials(n, p, U, u)

    # Begin with every degree-zero interval. Lower-degree bases have more
    # entries than the final n+1 bases; retain those right-hand neighbors for
    # non-clamped knots instead of indexing past a truncated array.
    count = n + p + 1
    N = np.zeros((count, Nu), dtype=u.dtype)
    for i in range(count):
        N[i, :] = (
            1.0 * (np.real(u) >= U[i]) * (np.real(u) < U[i + 1])
            + 1.0 * np.logical_and(np.real(u) == U[-1], i == n)
        )

    # First build the ordinary basis of degree p-k, then apply k derivative
    # recurrences while raising the degree back to p. No Python/Numba function
    # calls itself, so compiled cache entries have no recursive relocations.
    for degree in range(1, p + 1):
        count -= 1
        next_values = np.zeros((count, Nu), dtype=u.dtype)
        differentiating = degree > p - derivative_order
        for i in range(count):
            left = U[i + degree] - U[i]
            right = U[i + degree + 1] - U[i + 1]
            if left != 0:
                if differentiating:
                    next_values[i, :] += degree * N[i, :] / left
                else:
                    next_values[i, :] += (u - U[i]) * N[i, :] / left
            if right != 0:
                if differentiating:
                    next_values[i, :] -= degree * N[i + 1, :] / right
                else:
                    next_values[i, :] += (U[i + degree + 1] - u) * N[i + 1, :] / right
        N = next_values
    return N
