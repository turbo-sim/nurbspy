"""
Gradient-based NURBS curve fitting with the JAX backend
=========================================================

This example reconstructs a NURBS curve from a noisy point cloud by
optimizing its control point positions and weights with gradient-based
optimization, using ``nurbspy.jax``. This is the shape-optimization
workflow ``nurbspy`` was built for: a scattered set of measured points
(e.g. a digitized blade profile) is fitted by minimizing the distance
from every point to its closest point on the curve.

Two nested problems are involved:

1.  Inner problem (point inversion): for the *current* curve, find the
    parameter u*_k that minimizes the distance from target point Q_k to
    the curve, i.e. u*_k = argmin_u ||C(u) - Q_k||. This is solved with
    ``curve.project_points``, a vectorized, bounded Newton solve on the
    orthogonality condition (C(u) - Q) . C'(u) = 0.

2.  Outer problem (shape fitting): minimize the total squared distance
    over the curve's control points P and weights W:

        L(P, W) = sum_k || C(u*_k; P, W) - Q_k ||^2

    with respect to P and W, using reverse-mode automatic
    differentiation (``jax.grad``): one scalar output (the loss), many
    inputs (every control point coordinate and every weight).

Differentiating through point inversion
----------------------------------------
A natural first instinct is to let ``jax.grad`` differentiate straight
through the inner Newton solve. This is unnecessary, and empirically
this is currently NOT robust for ``project_points``: the same call that
evaluates cleanly in the forward pass raises a runtime error from the
backward pass ("A linear solver received non-finite (NaN or inf)
input"), for ordinary, well-conditioned interior points -- this appears
to be an issue in how the current point-inversion solve composes with
Optimistix's implicit-function-theorem differentiation, not a property
of NURBS curves specifically.

Fortunately, differentiating through the inner solve is also
unnecessary. ``project_points`` finds u* by solving exactly the
first-order stationarity condition of the squared-distance objective,
so this is a textbook application of the envelope theorem (Danskin's
theorem): at the optimum,

    d/dP [ min_u ||C(u; P) - Q||^2 ] = d/dP ||C(u*; P) - Q||^2

holding u* fixed, because d/du ||C(u; P) - Q||^2 = 0 at u = u*. So this
example freezes u* with ``jax.lax.stop_gradient`` before differentiating,
which sidesteps the unstable path above entirely and is also cheaper,
since it avoids invoking implicit differentiation through an iterative
solver at every outer optimization step. This is verified against
central finite differences (which re-solve the point inversion at every
perturbation) at the bottom of this script.

Weight positivity
-------------------
NURBS weights must stay strictly positive for the curve to remain a
valid rational parametrization. Rather than constraining the outer
optimizer, this example optimizes the log-weights and exponentiates
them inside the loss, which guarantees positivity by construction with
an unconstrained optimizer.
"""
import jax
import jax.numpy as jnp
import optimistix as optx
import matplotlib.pyplot as plt
import nurbspy.jax as nrb

jax.config.update("jax_enable_x64", True)


# -------------------------------------------------------------------------------------------------------------------- #
# Generate a synthetic target point cloud from a hidden "ground truth" curve
# -------------------------------------------------------------------------------------------------------------------- #
P_true = jnp.array([
    [0.00, 0.30, 0.70, 1.00, 1.30],
    [0.00, 0.80, -0.30, 0.80, 0.00],
])
degree = 3
true_curve = nrb.NurbsCurve(control_points=P_true, degree=degree)

n_points = 16
u_sample = jnp.linspace(0.04, 0.96, n_points)
Q_target = true_curve.get_value(u_sample)

noise_key = jax.random.PRNGKey(0)
Q_target = Q_target + 0.015 * jax.random.normal(noise_key, Q_target.shape)

# Initial guess: a perturbed control polygon with unit weights, representing
# an approximate initial CAD curve to be refined against the measured points.
P_init = P_true + jnp.array([
    [0.15, -0.12, 0.10, -0.15, 0.12],
    [0.12, -0.20, 0.18, -0.12, 0.08],
])
logW_init = jnp.zeros(P_init.shape[1])
params_init = {"P": P_init, "logW": logW_init}


# -------------------------------------------------------------------------------------------------------------------- #
# Differentiable fitting objective
# -------------------------------------------------------------------------------------------------------------------- #
def fitting_loss(params, Q):
    """Mean squared distance from the target points to the curve.

    Reverse-mode AD w.r.t. `params` gives one scalar output and as many
    inputs as there are control point coordinates and weights.
    """
    P, W = params["P"], jnp.exp(params["logW"])
    curve = nrb.NurbsCurve(control_points=P, weights=W, degree=degree)

    # Inner problem: point inversion, frozen before differentiating (envelope theorem)
    u_star = jax.lax.stop_gradient(curve.project_points(Q))

    # Outer objective: squared distance at the (fixed) closest points
    C = curve.get_value(u_star)
    return jnp.mean(jnp.sum((C - Q) ** 2, axis=0))


# -------------------------------------------------------------------------------------------------------------------- #
# Demonstrate reverse-mode AD explicitly: one scalar loss, many inputs
# -------------------------------------------------------------------------------------------------------------------- #
loss_value, grad = jax.value_and_grad(fitting_loss)(params_init, Q_target)
n_inputs = params_init["P"].size + params_init["logW"].size
print(f"Initial loss:            {loss_value:.6e}")
print(f"Reverse-mode AD:         1 scalar output, {n_inputs} scalar inputs")
print(f"|grad P|:                {jnp.linalg.norm(grad['P']):.4e}")
print(f"|grad logW|:              {jnp.linalg.norm(grad['logW']):.4e}\n")


# -------------------------------------------------------------------------------------------------------------------- #
# Outer optimization: fit control points and weights with BFGS
# -------------------------------------------------------------------------------------------------------------------- #
solver = optx.BFGS(rtol=1e-8, atol=1e-8)
solution = optx.minimise(
    fitting_loss, solver, params_init, args=Q_target, throw=False, max_steps=500
)
params_fit = solution.value
loss_fit = fitting_loss(params_fit, Q_target)

print(f"Optimizer status:        {solution.result}")
print(f"Final loss:               {loss_fit:.6e}")
print(f"Loss reduction:           {loss_value / loss_fit:.1f}x\n")


# -------------------------------------------------------------------------------------------------------------------- #
# Verify the envelope-theorem gradient against central finite differences
# (each perturbed evaluation re-solves the inner point-inversion problem)
# -------------------------------------------------------------------------------------------------------------------- #
eps = 1e-6


def finite_difference_gradient(field):
    base = params_init[field]
    flat = base.reshape(-1)
    entries = []
    for i in range(flat.size):
        perturbation = jnp.zeros_like(flat).at[i].set(eps)
        params_plus = {**params_init, field: (flat + perturbation).reshape(base.shape)}
        params_minus = {**params_init, field: (flat - perturbation).reshape(base.shape)}
        entries.append(
            (fitting_loss(params_plus, Q_target) - fitting_loss(params_minus, Q_target)) / (2 * eps)
        )
    return jnp.array(entries).reshape(base.shape)


fd_grad_P = finite_difference_gradient("P")
max_error = jnp.max(jnp.abs(grad["P"] - fd_grad_P))
print(f"Max |reverse-mode AD grad P - finite-difference grad P|: {max_error:.3e}")
print("(should be within finite-difference truncation error, ~1e-6 to 1e-7)\n")


# -------------------------------------------------------------------------------------------------------------------- #
# Plot the target points, initial guess, and fitted curve
# -------------------------------------------------------------------------------------------------------------------- #
fig, ax = plt.subplots(figsize=(7, 5), layout="constrained")

initial_curve = nrb.NurbsCurve(control_points=P_init, weights=jnp.exp(logW_init), degree=degree)
fitted_curve = nrb.NurbsCurve(control_points=params_fit["P"], weights=jnp.exp(params_fit["logW"]), degree=degree)

u_plot = jnp.linspace(0.0, 1.0, 501)
C_init = initial_curve.get_value(u_plot)
C_fit = fitted_curve.get_value(u_plot)

ax.plot(C_init[0], C_init[1], "--", color="gray", linewidth=1.5, label="Initial guess")
ax.plot(C_fit[0], C_fit[1], "-", color="tab:blue", linewidth=2, label="Fitted curve")
ax.plot(Q_target[0], Q_target[1], "o", markerfacecolor="w", markeredgecolor="tab:red",
        markersize=6, linestyle="", label="Target points")

ax.set(xlabel="x", ylabel="y", title="Gradient-based NURBS curve fitting (nurbspy.jax)")
ax.set_aspect("equal", adjustable="box")
ax.grid(alpha=0.2)
ax.legend()
plt.show()
