
import jax
import jax.lax as lax
import jax.numpy as jnp

# -------------------------------------------------------------------------------------------------------------------- #
# Standalone function to compute basis function values
# -------------------------------------------------------------------------------------------------------------------- #
def compute_basis_polynomials(n, p, U, u):
    """
    Evaluate B-spline basis functions of degree `p` for a set of parameter values `u`.

    The function implements the Cox-de Boor recursion formula using JAX and vectorizes
    the scalar evaluation over multiple u-values using `jax.vmap`.

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

    # Convert to JAX arrays
    U = jnp.asarray(U)
    u = jnp.atleast_1d(u)

    # Define scalar evaluation function
    basis_fn = lambda uu: _basis_single_u(n, p, U, uu)

    # Vectorize over all u-values
    N_all = jax.vmap(basis_fn)(u)

    # Transpose so that shape = (n+1, Nu)
    return jnp.transpose(N_all)

# Apply JIT compilation
compute_basis_polynomials = jax.jit(
    compute_basis_polynomials,
    static_argnames=('n', 'p'),
)

def _basis_single_u(n, p, U, u):
    """Evaluate all (n+1) B-spline basis functions of degree p at a single scalar u."""

    # Number of knot spans
    m = len(U) - 1

    # Initialize zeroth-degree basis functions
    # All intervals are [U[i], U[i+1]) except the last one, which is closed
    is_interior = (u >= U[:-1]) & (u < U[1:])
    is_last = (u == U[-1]) & (U[1:] == U[-1])
    N0 = jnp.where(is_interior | is_last, 1.0, 0.0)

    # Initialize storage
    N = jnp.zeros((p + 1, m))
    N = N.at[0, :].set(N0)

    # The recursive computation is implemented via JAX-controlled loops
    # using lax.fori_loop for XLA fusion (faster than Python loops).
    def outer_body(k, N):
        """Compute all basis functions of degree k from degree k-1."""
        m_k = m - k  # number of active basis entries shrinks with degree

        def inner_body(i, N):
            """Compute N[k,i] from N[k-1,i] and N[k-1,i+1] (Cox-de Boor recursion)."""
            denom1 = U[i + k] - U[i]
            term1 = jnp.where(denom1 == 0.0, 0.0, (u - U[i]) / denom1 * N[k - 1, i])

            denom2 = U[i + k + 1] - U[i + 1]
            term2 = jnp.where(denom2 == 0.0, 0.0, (U[i + k + 1] - u) / denom2 * N[k - 1, i + 1])

            # Update entry N[k, i] (immutably, handled efficiently by JAX)
            return N.at[k, i].set(term1 + term2)

        # Apply inner loop over all i indices for degree k
        N = lax.fori_loop(0, m_k, inner_body, N)
        return N

    # Outer loop over polynomial degree (1 → p)
    N = lax.fori_loop(1, p + 1, outer_body, N)

    # Return only the (n+1) basis functions of degree p
    return N[p, :n + 1]


# -------------------------------------------------------------------------------------------------------------------- #
# Standalone function to compute basis function derivatives
# -------------------------------------------------------------------------------------------------------------------- #
def compute_all_basis_polynomials_derivatives(n, p, U, u):
    """
    Evaluate the analytic derivatives of B-spline basis functions of degree `p`.

    The function implements the recursive derivative relations for B-spline basis functions
    (following the same recursion pattern as the basis definition). Computations are
    vectorized across multiple u-values using `jax.vmap`.

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
        Array containing the analytic derivative of each basis function at all u-values.
    """

    # Vectorize across u-values
    all_derivatives = jax.vmap(lambda uu: _basis_all_derivatives_single_u(n, p, U, uu))(u)

    # Transpose shape (Nu, p+1, n+1) to (p+1, n+1, Nu)
    all_derivatives = jnp.transpose(all_derivatives, (1, 2, 0))

    return all_derivatives

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

    # Degree-0 basis and its derivatives (only the function itself)
    N0 = _basis_single_u(n, 0, U, u)
    N_ders = jnp.zeros((1, n + 1))
    N_ders = N_ders.at[0, :].set(N0)

    # Helper: Build all derivatives for a given degree d
    def build_degree_block(d, N_prev):
        # Allocate derivatives for this degree
        N_block = jnp.zeros((d + 1, n + 1))

        # Zeroth derivative → degree-d basis
        N_d = _basis_single_u(n, d, U, u)
        N_block = N_block.at[0, :].set(N_d)

        # Higher derivatives follow recursive formula
        for r in range(1, d + 1):
            N_prev_r = N_prev[r - 1, :]
            N_r = jnp.zeros((n + 1,))

            for i in range(n + 1):
                denom1 = U[i + d] - U[i]
                denom2 = U[i + d + 1] - U[i + 1]
                term1 = jnp.where(denom1 == 0.0, 0.0, d * N_prev_r[i] / denom1)
                term2 = jnp.where(denom2 == 0.0, 0.0, d * N_prev_r[i + 1] / denom2)
                N_r = N_r.at[i].set(term1 - term2)

            N_block = N_block.at[r, :].set(N_r)

        return N_block

    # Build successively from degree 1 to p
    for d in range(1, p + 1):
        N_ders = build_degree_block(d, N_ders)

    # Final N_ders contains all derivatives for degree p
    return N_ders  # Shape (p+1, n+1)


def compute_basis_polynomials_derivatives(n, p, U, u, derivative_order):
    """
    Return only the specified derivative order of B-spline basis functions.
    """

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
    u = jnp.asarray(u)

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

