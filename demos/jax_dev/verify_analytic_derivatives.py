"""
Verification of analytic vs automatic differentiation for Bezier curve derivatives
"""

# -------------------------------------------------------------------------------------------------------------------- #
# Imports
# -------------------------------------------------------------------------------------------------------------------- #
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import nurbspy.jax as nrb


# -------------------------------------------------------------------------------------------------------------------- #
# Define control points and build curves
# -------------------------------------------------------------------------------------------------------------------- #
P = jnp.array([
    [0.20, 0.40, 0.80, 0.60, 0.40],  # x-coordinates
    [0.50, 0.70, 0.60, 0.20, 0.20],  # y-coordinates
])

# Classical and JAX-compatible Bézier curves
nurbsCurve = nrb.NurbsCurve(control_points=P)


# -------------------------------------------------------------------------------------------------------------------- #
# Function for comparing analytic vs autodiff derivatives
# -------------------------------------------------------------------------------------------------------------------- #
def verify_derivative_order(nurbs, u, order, tol=1e-10, plot=True):
    """
    Compare analytic and JAX autodiff derivatives for the specified order.
    """

    # Analytic curve values and derivatives
    C = nurbs.get_value(u)  # shape (ndim, N)
    dC_analytic = nurbs.get_derivative(u, order)

    # Define scalar curve evaluation
    def curve_eval_single(u_scalar):
        """Evaluate curve coordinates for a single scalar u."""
        val = nurbs.get_value(jnp.array([u_scalar]))
        return val[:, 0]  # flatten to (ndim,)

    # Nested jacfwd for arbitrary derivative order
    f = curve_eval_single
    for _ in range(order):
        f = jax.jacfwd(f)

    # Vectorize across u
    dC_autodiff = jax.vmap(f)(u).T  # shape (ndim, N)

    # Compute RMS absolute and relative errors
    abs_err = jnp.sqrt(jnp.mean((dC_analytic - dC_autodiff) ** 2, axis=1))

    print(f"\n--- Derivative order {order} ---")
    print("RMS absolute error:", abs_err)

    # Assertions to ensure analytic and AD results match closely
    assert jnp.all(abs_err < tol), (
        f"Mismatch detected for order {order}.\n"
        f"Absolute RMS error: {abs_err}\n"
    )

    # Plot comparison if requested
    if plot:
        fig, ax = plt.subplots(figsize=(5, 4))
        colors = ["r", "b", "g"]
        labels = ["x", "y", "z"]
        for dim in range(C.shape[0]):
            c = colors[dim % len(colors)]
            label = labels[dim % len(labels)]
            ax.plot(u, dC_analytic[dim], f"{c}-", markersize=3.5,
                    label=rf"$\frac{{d^{order}}}{{du^{order}}}{label}_{{\mathrm{{analytic}}}}$")
            ax.plot(u, dC_autodiff[dim], f"{c}o", markersize=3.5,
                    label=rf"$\frac{{d^{order}}}{{du^{order}}}{label}_{{\mathrm{{autodiff}}}}$")

        ax.legend(ncols=2, fontsize=11, loc="lower right")
        fig.tight_layout(pad=1)


# -------------------------------------------------------------------------------------------------------------------- #
# Run verification for multiple derivative orders
# -------------------------------------------------------------------------------------------------------------------- #
u = jnp.linspace(0.0, 1.0, 51)

for order in range(1, 6):  # test 1st, 2nd, 3rd derivatives
    verify_derivative_order(nurbsCurve, u, order, tol=1e-10, plot=True)

print("\nAll derivative orders verified successfully!")


plt.show()