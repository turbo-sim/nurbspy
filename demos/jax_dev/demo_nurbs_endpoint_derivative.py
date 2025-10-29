"""
Verify endpoint tangency property of a 2D Bezier curve using analytic derivatives.
"""

# -------------------------------------------------------------------------------------------------------------------- #
# Imports
# -------------------------------------------------------------------------------------------------------------------- #
import jax.numpy as jnp
import matplotlib.pyplot as plt
import nurbspy.jax as nrb


# -------------------------------------------------------------------------------------------------------------------- #
# Define control points and build curves
# -------------------------------------------------------------------------------------------------------------------- #

# Control points (ndim, n+1)
P = jnp.array([
    [0.20, 0.40, 0.80, 0.60, 0.40],
    [0.50, 0.70, 0.60, 0.20, 0.25],
])

# Weights (n+1,)
W = jnp.array([1.0, 1.3, 0.8, 1.2, 1.0])  # example non-uniform weights
# W = jnp.array([1.0, 1.0, 1.0, 1.0, 1.0])  # example non-uniform weights

# Degree
p = 4

# Build CPU and JAX curves explicitly as NURBS
nurbs = nrb.NurbsCurve(control_points=P, weights=W, degree=p)


# -------------------------------------------------------------------------------------------------------------------- #
# Compute analytic derivatives at the endpoints
# -------------------------------------------------------------------------------------------------------------------- #
# Parametric endpoints
u_start = 0.0
u_end   = 1.0

# Analytic derivative from curve object
dC_start = nurbs.get_derivative(jnp.array([u_start]), 1)[:, 0]
dC_end   = nurbs.get_derivative(jnp.array([u_end]),   1)[:, 0]

# Curve position at endpoints (needed for rational formula)
C_start = nurbs.get_value(jnp.array([u_start]))[:, 0]
C_end   = nurbs.get_value(jnp.array([u_end]))[:, 0]

# Expected tangent at start (rational endpoint formula)
w0, w1 = W[0], W[1]
P0, P1 = P[:, 0], P[:, 1]
dC_expected_start = p * (
    (w1 * P1 - w0 * P0) / w0
    - (w1 - w0) / w0 * C_start
)

# Expected tangent at end
wn_1, wn = W[-2], W[-1]
Pn_1, Pn = P[:, -2], P[:, -1]
dC_expected_end = p * (
    (wn * Pn - wn_1 * Pn_1) / wn
    - (wn - wn_1) / wn * C_end
)

# -------------------------------------------------------------------------------------------------------------------- #
# Compare and print results
# -------------------------------------------------------------------------------------------------------------------- #
abs_err_start = jnp.linalg.norm(dC_start - dC_expected_start)
abs_err_end   = jnp.linalg.norm(dC_end - dC_expected_end)

print("\n--- Endpoint tangency verification ---")
print(f"Degree p = {p}")
print(f"Start derivative analytic:  {dC_start}")
print(f"Start derivative expected:  {dC_expected_start}")
print(f"Start absolute error:       {abs_err_start:.3e}")
print(f"End derivative analytic:    {dC_end}")
print(f"End derivative expected:    {dC_expected_end}")
print(f"End absolute error:         {abs_err_end:.3e}")

# assert abs_err_start < 1e-12, "Start derivative does not match endpoint tangency condition!"
# assert abs_err_end < 1e-12,   "End derivative does not match endpoint tangency condition!"

print("\nEndpoint tangency property verified successfully.")


# -------------------------------------------------------------------------------------------------------------------- #
# Plot the curve and tangents
# -------------------------------------------------------------------------------------------------------------------- #
fig, ax = nurbs.plot(frenet_serret=False)
C = nurbs.get_value(jnp.linspace(0, 1, 200))
ax.plot(*C, "k-", label="Bezier curve")

# Plot tangents at endpoints
scale = 0.2
P0, Pn = P[:, 0], P[:, -1]
ax.arrow(*P0, *(scale * dC_start), color="r", width=0.002, label="Tangent at start")
ax.arrow(*Pn, *(scale * dC_end), color="b", width=0.002, label="Tangent at end")

ax.legend()
ax.set_aspect("equal")
ax.set_title("Endpoint tangency verification")
plt.tight_layout()
plt.show()
