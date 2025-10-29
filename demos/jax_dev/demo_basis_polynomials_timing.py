
""" Example showing how to compute a family of basis polynomials """


# -------------------------------------------------------------------------------------------------------------------- #
# Importing packages
# -------------------------------------------------------------------------------------------------------------------- #
import time
import jax.numpy as jnp
import nurbspy.jax as nrb
import matplotlib.pyplot as plt


# -------------------------------------------------------------------------------------------------------------------- #
# Basis polynomials and derivatives example
# -------------------------------------------------------------------------------------------------------------------- #
# Maximum index of the basis polynomials (counting from zero)
n = 4

# Define the order of the basis polynomials
p = 3

# Define the knot vector (clamped spline)
# p+1 zeros, n-p equispaced points between 0 and 1, and p+1 ones. In total r+1 points where r=n+p+1
U = jnp.concatenate((jnp.zeros(p), jnp.linspace(0, 1, n - p + 2), jnp.ones(p)))

# Define a new u-parametrization suitable for finite differences
Nu = 1000
u = jnp.linspace(0.00, 1.00, Nu)       # Make sure that the limits [0, 1] also work when making changes

# --------------------------------------------------------------------------------
# Timing: 20 steady-state runs with per-iteration L2 errors
# --------------------------------------------------------------------------------

# --- Header ---
print("Timing 20 steady-state runs (in milliseconds):")
print(" idx |   Basis   | dN(analytic) | ddN(analytic) |  dN(JAX)  | ddN(JAX) | dN-dN_JAX | ddN-ddN_JAX ")
print("-----|------------|--------------|---------------|-----------|----------|-------------------|---------------------")

# --- Timing loop ---
for k in range(20):
    t0 = time.perf_counter()
    N_basis = nrb.compute_basis_polynomials(n, p, U, u)
    t1 = time.perf_counter()
    dN_basis = nrb.compute_basis_polynomials_derivatives(n, p, U, u, 1)
    t2 = time.perf_counter()
    ddN_basis = nrb.compute_basis_polynomials_derivatives(n, p, U, u, 2)
    t3 = time.perf_counter()
    dN_basis_jax = nrb.compute_basis_polynomials_derivatives_jax(n, p, U, u, 1)
    t4 = time.perf_counter()
    ddN_basis_jax = nrb.compute_basis_polynomials_derivatives_jax(n, p, U, u, 2)
    t5 = time.perf_counter()

    # Compute 2-norm relative errors for this iteration
    rel_err_dN  = float(jnp.max(dN_basis - dN_basis_jax))
    rel_err_ddN = float(jnp.max(ddN_basis - ddN_basis_jax))

    # Print nicely formatted line
    print(f"{k:4d} | { (t1-t0)*1e3:10.3f} | { (t2-t1)*1e3:12.3f} | { (t3-t2)*1e3:13.3f} | "
          f"{ (t4-t3)*1e3:9.3f} | { (t5-t4)*1e3:8.3f} | "
          f"{rel_err_dN:17.3e} | {rel_err_ddN:19.3e}")
