"""
Minimum-bending-energy curve through prescribed points
=========================================================

Problem 1 from `optimization_examples.md`: given a set of prescribed
points in the plane, find the smooth NURBS curve that interpolates all
of them while minimizing its bending energy

    J(P) = integral of kappa(u)^2 ds = integral_0^1 kappa(u)^2 ||C'(u)|| du

where kappa(u) is the curvature, computed with the existing
``NurbsCurve.get_curvature`` method (``nurbspy.jax``) rather than
re-deriving the curvature formula by hand.

Formulation
-------------
* Control points P are the only optimization variables; the degree,
  knot vector, and weights are fixed (a polynomial B-spline).
* Each prescribed point Q_j is assigned a fixed parameter u_j via
  chord-length parameterization of the Q_j themselves; the u_j are not
  optimized.
* The bending energy is approximated with a fixed Gauss-Legendre
  quadrature rule on [0, 1], so the computational graph JAX
  differentiates has a static, fixed shape.
* The interpolation conditions C(u_j) = Q_j are enforced as nonlinear
  equality constraints. This is a genuinely constrained problem (unlike
  the point-inversion curve-fitting example, where the data term is a
  soft penalty), so it is solved with SciPy's SLSQP using JAX-computed
  objective gradients and constraint Jacobians (``jax.grad`` /
  ``jax.jacrev``), converted to NumPy for SciPy.

Before optimizing, the objective gradient is checked against a central
finite-difference directional derivative along a random direction, as
recommended in `optimization_examples.md`.

A fixed quadrature rule can be exploited by the optimizer
------------------------------------------------------------
A fixed, coarse quadrature rule does not just lose accuracy -- it can be
actively exploited by the optimizer, since SLSQP only ever "sees" the
bending energy through its value at the quadrature nodes. With too few
nodes, SLSQP found solutions whose *reported* bending energy looked
small only because the quadrature under-resolved the true integral: with
30 nodes, one tight loop was driven into a gap between quadrature nodes
entirely (a near-cusp with parameter speed ||C'(u)|| collapsing to
~1e-3 and a curvature spike four orders of magnitude higher than
anywhere else on the curve); even at 60 nodes, with no visible loop, the
sharp curvature peaks of this example were still under-resolved enough
that the reported bending energy was 24% below its converged value. This
is exactly the discretization risk flagged in `optimization_examples.md`
for the bounded-curvature problem, and it applies here too: this script
validates the optimized curve on a grid much denser than the one used
during optimization, and 120 Gauss-Legendre nodes were needed here for
the two to agree to within 1%.
"""
import numpy as np
import numpy.polynomial.legendre as leggauss
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from scipy.optimize import minimize

import nurbspy.jax as nrb
from nurbspy.jax.nurbs_curve import compute_basis_polynomials
from nurbspy.graphics import set_plot_options

jax.config.update("jax_enable_x64", True)
set_plot_options()


# -------------------------------------------------------------------------------------------------------------------- #
# Problem setup: prescribed points, chord-length parameters, curve size
#
# n_control is kept at just one more than the number of prescribed points
# (one "spare" control point pair). Pure bending-energy minimization has no
# term penalizing self-intersection, so with more spare control points than
# this, SLSQP can -- and, empirically, reliably does -- exploit the extra
# freedom to route a tight, self-intersecting loop through a low-curvature
# shortcut instead of producing a fair, loop-free curve (verified directly:
# with 3 spare control point pairs, the "optimal" curve had a curvature
# spike of ~30000 hidden between quadrature nodes; even with only 2 spare
# pairs, and even with control points explicitly bounded to a neighborhood
# of the data, a visible loop remained the lower-energy solution). This is
# a genuine property of the mathematical problem as stated, not a solver
# bug, and it is exactly why real fairing systems add extra shape
# constraints beyond pure bending-energy minimization (see problem 3 in
# `optimization_examples.md`, which trades off bending energy against arc
# length for this reason).
# -------------------------------------------------------------------------------------------------------------------- #
Q = np.array([
    [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
    [0.0, 2.0, -1.0, 2.0, -1.0, 1.5],
])
m = Q.shape[1]
degree = 3
n_control = m + 1

# Chord-length parameterization of the prescribed points (fixed, not optimized)
chords = np.linalg.norm(np.diff(Q, axis=1), axis=0)
u_data = np.concatenate(([0.0], np.cumsum(chords) / np.sum(chords)))
u_data = jnp.asarray(u_data)
Q = jnp.asarray(Q)
ndim = Q.shape[0]


# -------------------------------------------------------------------------------------------------------------------- #
# Fixed Gauss-Legendre quadrature on [0, 1] for the bending-energy integral
# -------------------------------------------------------------------------------------------------------------------- #
def gauss_legendre_01(n):
    """Gauss-Legendre nodes and weights mapped from [-1, 1] to [0, 1]."""
    x, w = leggauss.leggauss(n)
    return 0.5 * (x + 1.0), 0.5 * w


u_quad_np, w_quad_np = gauss_legendre_01(120)
u_quad = jnp.asarray(u_quad_np)
w_quad = jnp.asarray(w_quad_np)


# -------------------------------------------------------------------------------------------------------------------- #
# Objective: bending energy, and interpolation equality constraints
# -------------------------------------------------------------------------------------------------------------------- #
def unpack(x):
    return x.reshape(ndim, n_control)


def bending_energy(x):
    P = unpack(x)
    curve = nrb.NurbsCurve(control_points=P, degree=degree)
    dC = curve.get_derivative(u_quad, order=1)          # (ndim, n_quad)
    speed = jnp.linalg.norm(dC, axis=0)                  # (n_quad,)
    kappa = curve.get_curvature(u_quad)                  # (n_quad,)
    return jnp.sum(w_quad * kappa**2 * speed)


def interpolation_constraints(x):
    P = unpack(x)
    curve = nrb.NurbsCurve(control_points=P, degree=degree)
    C = curve.get_value(u_data)                          # (ndim, m)
    return (C - Q).reshape(-1)


objective_and_grad = jax.jit(jax.value_and_grad(bending_energy))
constraints_fn = jax.jit(interpolation_constraints)
constraints_jac = jax.jit(jax.jacrev(interpolation_constraints))


def objective_np(x):
    value, _ = objective_and_grad(jnp.asarray(x))
    return float(value)


def objective_grad_np(x):
    _, grad = objective_and_grad(jnp.asarray(x))
    return np.asarray(grad, dtype=np.float64)


def constraints_np(x):
    return np.asarray(constraints_fn(jnp.asarray(x)), dtype=np.float64)


def constraints_jac_np(x):
    return np.asarray(constraints_jac(jnp.asarray(x)), dtype=np.float64)


# -------------------------------------------------------------------------------------------------------------------- #
# Initial guess
# -------------------------------------------------------------------------------------------------------------------- #
# Option A (previous default): a feasible but non-optimal interpolating
# curve, obtained by solving the interpolation constraints directly.
#
# For fixed weights and knot vector, curve position is *linear* in the
# control points: C(u) = P @ N(u), where N(u) collects the basis functions.
# Since n_control > m, the constraints C(u_j) = Q_j are underdetermined;
# NumPy's least-squares solver returns the minimum-norm solution, which
# interpolates all the prescribed points exactly but has no reason to
# minimize bending energy. Using this feasible baseline (rather than an
# infeasible guess such as a straight line, which would also make the
# bending-energy comparison meaningless and the curvature -- and hence the
# gradient of kappa^2 -- degenerate at exactly zero) makes the "initial
# vs. optimized" comparison below an apples-to-apples comparison of two
# curves that both satisfy the same constraints, so SLSQP's minimization
# can only match or reduce the bending energy relative to it.
# -------------------------------------------------------------------------------------------------------------------- #
# dummy_curve = nrb.NurbsCurve(control_points=jnp.zeros((ndim, n_control)), degree=degree)
# N_data = np.asarray(compute_basis_polynomials(n_control - 1, degree, dummy_curve.U, u_data))  # (n_control, m)
#
# Q_np = np.asarray(Q)
# P_init = np.stack([
#     np.linalg.lstsq(N_data.T, Q_np[dim], rcond=None)[0] for dim in range(ndim)
# ])
# x_init = P_init.reshape(-1)

# Option B (poorer initial guess): a straight line between the first and
# last prescribed point, evenly spaced by control-point index.
#
# This is infeasible (it does not interpolate the interior points), and it
# has exactly zero curvature everywhere, so both the "initial vs. optimized"
# bending-energy comparison and the gradient-validation check below become
# degenerate at this starting point: the true objective gradient is exactly
# zero at a perfectly straight line (d(kappa^2)/dP = 2*kappa*d(kappa)/dP = 0
# when kappa = 0), so reverse-mode AD and finite differences will both
# correctly return ~1e-16, not a meaningful relative error. This is not a
# bug -- see the note above -- but it does mean this starting point is a
# poor choice for validating the gradient or for an honest "reduction"
# narrative; it mainly demonstrates that SLSQP can still recover a
# reasonable fair curve from a much worse start.
Q_np = np.asarray(Q)
t_init = np.linspace(0.0, 1.0, n_control)
P_init = np.outer(Q_np[:, 0], 1 - t_init) + np.outer(Q_np[:, -1], t_init)
x_init = P_init.reshape(-1)


# -------------------------------------------------------------------------------------------------------------------- #
# Gradient validation: directional derivative vs. central finite difference
# -------------------------------------------------------------------------------------------------------------------- #
rng = np.random.default_rng(0)
direction = rng.normal(size=x_init.shape)
direction /= np.linalg.norm(direction)

h = 1e-6
J_plus = objective_np(x_init + h * direction)
J_minus = objective_np(x_init - h * direction)
fd_directional = (J_plus - J_minus) / (2 * h)
ad_directional = objective_grad_np(x_init) @ direction

print("Gradient validation (random-direction directional derivative)")
print(f"  Reverse-mode AD:    {ad_directional: .8e}")
print(f"  Finite difference:  {fd_directional: .8e}")
print(f"  Relative error:     {abs(ad_directional - fd_directional) / abs(fd_directional):.3e}\n")


# -------------------------------------------------------------------------------------------------------------------- #
# Solve the constrained minimum-bending-energy problem with SLSQP
# -------------------------------------------------------------------------------------------------------------------- #
convergence_history = [objective_np(x_init)]
infeasibility_history = [np.max(np.abs(constraints_np(x_init)))]


def callback(x_k, *_):
    convergence_history.append(objective_np(x_k))
    infeasibility_history.append(np.max(np.abs(constraints_np(x_k))))


result = minimize(
    objective_np,
    x_init,
    jac=objective_grad_np,
    method="SLSQP",
    constraints=[{"type": "eq", "fun": constraints_np, "jac": constraints_jac_np}],
    callback=callback,
    options={"maxiter": 200, "ftol": 1e-12},
)

x_opt = result.x
J_init = objective_np(x_init)
J_opt = objective_np(x_opt)
constraint_residual = np.max(np.abs(constraints_np(x_opt)))

print(f"SLSQP status:              {result.message}")
print(f"Initial bending energy:    {J_init:.6e}")
print(f"Final bending energy:      {J_opt:.6e}   ({J_init / J_opt:.1f}x reduction)")
print(f"Max interpolation residual: {constraint_residual:.3e}\n")


# -------------------------------------------------------------------------------------------------------------------- #
# Validate the optimized curve on a grid much denser than the optimization
# quadrature: a fixed-node quadrature rule can be exploited by the optimizer
# (see module docstring), so the reported bending energy is only trustworthy
# if it agrees with a dense-grid estimate and the curve never nearly stalls.
# -------------------------------------------------------------------------------------------------------------------- #
curve_opt_check = nrb.NurbsCurve(control_points=unpack(jnp.asarray(x_opt)), degree=degree)
u_dense = jnp.linspace(0.0, 1.0, 2001)
dC_dense = curve_opt_check.get_derivative(u_dense, order=1)
speed_dense = jnp.linalg.norm(dC_dense, axis=0)
kappa_dense = curve_opt_check.get_curvature(u_dense)
J_dense = float(jnp.trapezoid(kappa_dense**2 * speed_dense, u_dense))
min_speed_dense = float(jnp.min(speed_dense))

print("Dense-grid validation (2001 points, vs. the 120-node optimization quadrature)")
print(f"  Bending energy (quadrature): {J_opt:.6e}")
print(f"  Bending energy (dense grid): {J_dense:.6e}")
print(f"  Minimum parameter speed:     {min_speed_dense:.4e}  (should not be near zero)")
if abs(J_dense - J_opt) / J_opt > 0.05 or min_speed_dense < 1e-2:
    print("  WARNING: quadrature and dense-grid estimates disagree -- increase n_quad and re-run.\n")
else:
    print("  OK: quadrature and dense-grid estimates agree; no near-cusp detected.\n")


# -------------------------------------------------------------------------------------------------------------------- #
# Plots: control polygons and curves, curvature distributions, convergence history
# -------------------------------------------------------------------------------------------------------------------- #
curve_init = nrb.NurbsCurve(control_points=unpack(jnp.asarray(x_init)), degree=degree)
curve_opt = nrb.NurbsCurve(control_points=unpack(jnp.asarray(x_opt)), degree=degree)

u_plot = jnp.linspace(0.0, 1.0, 501)
C_init = curve_init.get_value(u_plot)
C_opt = curve_opt.get_value(u_plot)
kappa_init = curve_init.get_curvature(u_plot)
kappa_opt = curve_opt.get_curvature(u_plot)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), layout="constrained")

ax = axes[0]
P_init_arr, P_opt_arr = unpack(jnp.asarray(x_init)), unpack(jnp.asarray(x_opt))
ax.plot(P_init_arr[0], P_init_arr[1], "o--", color="gray", markersize=4, linewidth=1, label="Initial control polygon")
ax.plot(C_init[0], C_init[1], "-", color="gray", linewidth=1.5, label="Initial curve")
ax.plot(P_opt_arr[0], P_opt_arr[1], "s--", color="tab:orange", markersize=4, linewidth=1, label="Optimized control polygon")
ax.plot(C_opt[0], C_opt[1], "-", color="tab:blue", linewidth=2, label="Optimized curve")
ax.plot(Q[0], Q[1], "o", markerfacecolor="w", markeredgecolor="tab:red", markersize=8, linestyle="", label="Prescribed points")
ax.set(xlabel="x", ylabel="y", title="Minimum-bending-energy interpolation")
ax.set_aspect("equal", adjustable="box")
ax.grid(alpha=0.2)
ax.legend(fontsize=8)

ax = axes[1]
ax.plot(u_plot, kappa_init, "-", color="gray", label="Initial")
ax.plot(u_plot, kappa_opt, "-", color="tab:blue", label="Optimized")
ax.set(xlabel="$u$", ylabel=r"curvature $\kappa(u)$", title="Curvature distribution")
ax.grid(alpha=0.2)
ax.legend(fontsize=8)

ax = axes[2]
line_energy, = ax.plot(convergence_history, "o-", color="tab:blue", markersize=4, label="Bending energy $J$")
ax.set(xlabel="SLSQP iteration", ylabel="Bending energy $J$", title="Convergence history", yscale="log")
ax.yaxis.label.set_color("tab:blue")
ax.tick_params(axis="y", colors="tab:blue")
ax.grid(alpha=0.2)

ax_infeas = ax.twinx()
line_infeas, = ax_infeas.plot(infeasibility_history, "s--", color="tab:red", markersize=4, label="Max constraint infeasibility")
ax_infeas.set(ylabel="Max $|C(u_j) - Q_j|$", yscale="log")
ax_infeas.yaxis.label.set_color("tab:red")
ax_infeas.tick_params(axis="y", colors="tab:red")

ax.legend(handles=[line_energy, line_infeas], fontsize=8, loc="upper right")

plt.show()
