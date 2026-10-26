from functools import partial as _partial

import jax
import jax.lax as lax
import jax.numpy as jnp


@_partial(jax.jit, static_argnames=('n', 'p', 'max_order'))
def _compute_basis_derivatives(n, p, U, u, max_order):
    """Build values and requested derivative rows in one degree recurrence.

    Basis indices and parameter samples are array axes; only the small degree
    and derivative-order loops are unrolled. Keep every lower-degree span
    until the final degree, including entries beyond the final basis count.
    """
    U = jnp.asarray(U)
    u = jnp.atleast_1d(u)
    interior = (u[None, :] >= U[:-1, None]) & (u[None, :] < U[1:, None])
    last = (u[None, :] == U[-1]) & (U[1:, None] == U[-1])
    # Match the original default-float storage even when U and u are float32.
    dtype = jnp.zeros(()).dtype
    previous = jnp.where(interior | last, 1.0, 0.0).astype(dtype)[None, :, :]
    max_order = min(p, max_order)

    for degree in range(1, p + 1):
        count = U.shape[0] - degree - 1
        left = U[:count, None]
        right = U[degree + 1:degree + count + 1, None]
        denom1 = U[degree:degree + count, None] - left
        denom2 = right - U[1:count + 1, None]
        # Mask the denominator before division as well as the coefficient:
        # an inactive division by zero can otherwise poison reverse-mode AD.
        safe1 = jnp.where(denom1 == 0.0, 1.0, denom1)
        safe2 = jnp.where(denom2 == 0.0, 1.0, denom2)
        factor1 = jnp.where(denom1 == 0.0, 0.0, 1.0 / safe1)
        factor2 = jnp.where(denom2 == 0.0, 0.0, 1.0 / safe2)
        rows = [
            (u[None, :] - left) * factor1 * previous[0, :-1]
            + (right - u[None, :]) * factor2 * previous[0, 1:]
        ]
        for order in range(1, min(degree, max_order) + 1):
            rows.append(degree * (
                factor1 * previous[order - 1, :-1]
                - factor2 * previous[order - 1, 1:]
            ))
        previous = jnp.stack(rows)

    return previous[:, :n + 1, :]


# -------------------------------------------------------------------------------------------------------------------- #
# Standalone function to compute basis function values
# -------------------------------------------------------------------------------------------------------------------- #
def compute_basis_polynomials(n, p, U, u):
    """
    Evaluate B-spline basis functions of degree `p` for a set of parameter values `u`.

    The function implements the Cox-de Boor recursion formula using JAX array
    operations across both basis indices and parameter samples.

    Parameters
    ----------
    n : int
        Highest index of the basis functions (number of functions = n+1).
    p : int
        Degree of the basis polynomials.
    U : array_like
        Knot vector of length n+p+2 defining the B-spline basis.
        The first and last knots should typically have multiplicity p+1 for clamped splines.
    u : float or array_like
        Scalar or array of parameter values where the basis functions are evaluated.

    Returns
    -------
    N : ndarray of shape (n+1, Nu)
        Array containing all basis functions evaluated at each u value.
        The first axis spans the basis index i, the second spans the parameter samples.
    """

    return _compute_basis_derivatives(n, p, U, u, max_order=0)[0]

# Apply JIT compilation
compute_basis_polynomials = jax.jit(
    compute_basis_polynomials,
    static_argnames=('n', 'p'),
)

def _basis_single_u(n, p, U, u):
    """Evaluate all (n+1) B-spline basis functions of degree p at a single scalar u."""
    return _compute_basis_derivatives(n, p, U, u, max_order=0)[0, :, 0]


# -------------------------------------------------------------------------------------------------------------------- #
# Standalone function to compute basis function derivatives
# -------------------------------------------------------------------------------------------------------------------- #
def compute_all_basis_polynomials_derivatives(n, p, U, u):
    """
    Evaluate the analytic derivatives of B-spline basis functions of degree `p`.

    The function implements the recursive derivative relations for B-spline basis
    functions, reusing each lower-degree table across derivative orders. Array
    operations cover all basis indices and parameter samples together.

    Parameters
    ----------
    n : int
        Highest index of the basis functions (number of functions = n+1).
    p : int
        Degree of the basis polynomials.
    U : array_like
        Knot vector of length n+p+2 defining the B-spline basis.
    u : float or array_like
        Scalar or array of parameter values where the derivatives are evaluated.

    Returns
    -------
    N_ders : ndarray of shape (p+1, n+1, Nu)
        Array containing derivatives of orders zero through p at all u-values.
    """

    return _compute_basis_derivatives(n, p, U, u, max_order=p)

# Apply JIT compilation
compute_all_basis_polynomials_derivatives = jax.jit(
    compute_all_basis_polynomials_derivatives,
    static_argnames=('n', 'p'),
)

def _basis_all_derivatives_single_u(n, p, U, u):
    """
    Compute all derivatives d^r/du^r N_{i,p}(u) for r = (0,...,p).

    Returns
    -------
    N_ders : ndarray, shape (p+1, n+1)
        N_ders[r, i] = r-th derivative of basis function i of degree p at u.
    """

    return _compute_basis_derivatives(n, p, U, u, max_order=p)[:, :, 0]


def compute_basis_polynomials_derivatives(n, p, U, u, derivative_order):
    """
    Return only the specified derivative order of B-spline basis functions.
    """

    u = jnp.atleast_1d(u)

    # Compute derivatives up to order "p"
    all_derivatives = compute_all_basis_polynomials_derivatives(n, p, U, u)

    # Define selector helper functions
    def in_range(order):
        return all_derivatives[order, :, :]

    def out_of_range(_):
        return jnp.zeros((n + 1, u.shape[0]))

    # Return the derivative of the desired order
    return jax.lax.cond(
        derivative_order <= p,
        in_range,
        out_of_range,
        operand=derivative_order,
    )

# Apply JIT compilation
compute_basis_polynomials_derivatives = jax.jit(
    compute_basis_polynomials_derivatives,
    static_argnames=('n', 'p'),
)



# -------------------------------------------------------------------------------------------------------------------- #
# Verification of analytic formulas against automatic differentiation
# -------------------------------------------------------------------------------------------------------------------- #
def compute_basis_polynomials_derivatives_jax(n, p, U, u, derivative_order):
    """
    Evaluate derivatives of B-spline basis functions using JAX automatic differentiation.

    This method uses nested applications of `jax.jacfwd` on the scalar evaluation function
    `_basis_single_u()` to compute derivatives of arbitrary order. It is mathematically
    exact within floating-point precision and provides a useful verification of the analytic
    recursive formulation.

    Parameters
    ----------
    n : int
        Highest index of the basis functions (number of functions = n+1).
    p : int
        Degree of the basis polynomials.
    U : array_like
        Knot vector of length n+p+2 defining the B-spline basis.
    u : float or array_like
        Scalar or array of parameter values where the derivatives are evaluated.
    derivative_order : int
        Order of the derivative to compute (0 ≤ derivative_order ≤ p).

    Returns
    -------
    N_ders : ndarray of shape (n+1, Nu)
        Array containing the derivatives of each basis function evaluated at all u-values.
    """

    # Convert to JAX arrays
    U = jnp.asarray(U)
    u = jnp.atleast_1d(u)

    # Define scalar basis function
    def basis_scalar(uu):
        return _basis_single_u(n, p, U, uu)

    # Precompute all derivatives up to order p
    def all_derivatives_fn(uu):
        f = basis_scalar
        results = [f(uu)]
        for _ in range(p):
            f = jax.jacfwd(f)
            results.append(f(uu))
        return jnp.stack(results)  # shape (p+1, n+1)

    # Vectorize over u-values
    all_derivatives = jax.vmap(all_derivatives_fn)(u)

    # Reorder shape (Nu, p+1, n+1) to (p+1, n+1, Nu)
    all_derivatives = jnp.transpose(all_derivatives, (1, 2, 0))

    # Dynamically select derivative order
    def in_range(k):
        return all_derivatives[k, :, :]

    def out_of_range(_):
        return jnp.zeros((n + 1, u.shape[0]))

    N_ders = lax.cond(
        derivative_order <= p,
        in_range,
        out_of_range,
        operand=derivative_order,
    )

    return N_ders  # Shape (n+1, Nu)


# JIT compile with only n and p static
compute_basis_polynomials_derivatives_jax = jax.jit(
    compute_basis_polynomials_derivatives_jax,
    static_argnames=('n', 'p'),
)

