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
* The interpolation conditions C(u_j) = Q_j are affine in P for fixed
  weights and knots, and are passed to SLSQP as equality constraints.
  This is a constrained problem (unlike
  the point-inversion curve-fitting example, where the data term is a
  soft penalty), so it is solved with SciPy's SLSQP using JAX-computed
  objective gradients and constraint Jacobians (``jax.grad`` /
  ``jax.jacrev``), converted to NumPy for SciPy.

Before optimizing, the objective gradient is checked against a central
finite-difference directional derivative along a random direction, as
recommended in `optimization_examples.md`.

The initial curve is a feasible interpolating B-spline, obtained by a
linear least-squares solve. The script prints a convergence row after
each SLSQP iteration. JAX first-call and synchronized warm-call timings
are reported separately from the solver time; the latter includes the
iteration reporting but excludes JAX warmup and gradient validation.
Timings exclude module imports and time spent in the interactive plot window.

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
from time import perf_counter

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

demo_start = perf_counter()
print("Minimum-bending-energy interpolation: building a feasible initial curve...", flush=True)


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


def objective_and_grad_np(x):
    """Return both together so SciPy can reuse one JAX evaluation per design."""
    value, grad = objective_and_grad(jnp.asarray(x))
    return float(value), np.asarray(grad, dtype=np.float64)


def objective_np(x):
    return objective_and_grad_np(x)[0]


def objective_grad_np(x):
    return objective_and_grad_np(x)[1]


def constraints_np(x):
    return np.asarray(constraints_fn(jnp.asarray(x)), dtype=np.float64)


def constraints_jac_np(x):
    return np.asarray(constraints_jac(jnp.asarray(x)), dtype=np.float64)


# -------------------------------------------------------------------------------------------------------------------- #
# Feasible initial curve: solve the linear interpolation equations
# -------------------------------------------------------------------------------------------------------------------- #
# For fixed weights and knot vector, curve position is *linear* in the
# control points: C(u) = P @ N(u), where N(u) collects the basis functions.
# Since n_control > m, the constraints C(u_j) = Q_j are underdetermined;
# NumPy's least-squares solver returns the minimum-norm solution, which
# interpolates all the prescribed points exactly but has no reason to
# minimize bending energy. This is a direct interpolation baseline, not
# a preliminary bending-energy optimization. Both the initial and final
# curves therefore satisfy the same prescribed-point requirements.
# -------------------------------------------------------------------------------------------------------------------- #
dummy_curve = nrb.NurbsCurve(control_points=jnp.zeros((ndim, n_control)), degree=degree)
N_data = np.asarray(compute_basis_polynomials(n_control - 1, degree, dummy_curve.U, u_data))
Q_np = np.asarray(Q)
coefficients, _, rank, _ = np.linalg.lstsq(N_data.T, Q_np.T, rcond=None)
P_init = coefficients.T
x_init = P_init.reshape(-1)
initial_residual = np.max(np.abs(P_init @ N_data - Q_np))
if rank != m or not np.isfinite(initial_residual) or initial_residual > 1e-10:
    raise RuntimeError("Could not construct a feasible interpolating initial curve.")
setup_time = perf_counter() - demo_start

print(f"  Cubic interpolating B-spline: {n_control} control points, {m} prescribed points")
print(f"  Initial max interpolation residual: {initial_residual:.3e}")
print(f"  Setup and initial interpolation:    {setup_time:.3f} s\n", flush=True)


# -------------------------------------------------------------------------------------------------------------------- #
# Warm the exact kernels used by SciPy, keeping their first-call costs out
# of the solver timing. Synchronization matters because JAX dispatch is async.
# First call includes tracing/compilation and execution; it is not compile-only.
# A single warm call gives an indicative latency, not a benchmark average.
# -------------------------------------------------------------------------------------------------------------------- #
print("JAX kernel timings (first call includes tracing/compilation and execution)", flush=True)
print(f"  {'Kernel':<28} {'First call [s]':>15} {'Warm call [ms]':>16}", flush=True)
warmup_start = perf_counter()
x_init_jax = jnp.asarray(x_init)
for name, kernel in (
    ("Objective + gradient", objective_and_grad),
    ("Interpolation residuals", constraints_fn),
    ("Constraint Jacobian", constraints_jac),
):
    call_start = perf_counter()
    jax.block_until_ready(kernel(x_init_jax))
    first_call_time = perf_counter() - call_start
    call_start = perf_counter()
    jax.block_until_ready(kernel(x_init_jax))
    warm_call_time = perf_counter() - call_start
    print(f"  {name:<28} {first_call_time:15.6f} {1e3 * warm_call_time:16.3f}", flush=True)
warmup_time = perf_counter() - warmup_start
print()


# -------------------------------------------------------------------------------------------------------------------- #
# Gradient validation: directional derivative vs. central finite difference
# -------------------------------------------------------------------------------------------------------------------- #
gradient_check_start = perf_counter()
rng = np.random.default_rng(0)
direction = rng.normal(size=x_init.shape)
direction /= np.linalg.norm(direction)

h = 1e-6
J_plus = objective_np(x_init + h * direction)
J_minus = objective_np(x_init - h * direction)
fd_directional = (J_plus - J_minus) / (2 * h)
ad_directional = objective_grad_np(x_init) @ direction
gradient_check_time = perf_counter() - gradient_check_start
relative_gradient_error = abs(ad_directional - fd_directional) / max(
    abs(ad_directional), abs(fd_directional), np.finfo(float).eps
)

print("Gradient validation (random-direction directional derivative)")
print(f"  Reverse-mode AD:    {ad_directional: .8e}")
print(f"  Finite difference:  {fd_directional: .8e}")
print(f"  Relative error:     {relative_gradient_error:.3e}\n")


# -------------------------------------------------------------------------------------------------------------------- #
# Solve the constrained minimum-bending-energy problem with SLSQP
# -------------------------------------------------------------------------------------------------------------------- #
J_init = objective_np(x_init)
convergence_history = [J_init]
infeasibility_history = [np.max(np.abs(constraints_np(x_init)))]


def print_iteration(iteration, energy, residual, elapsed):
    print(f"{iteration:6d} {energy:18.9e} {residual:18.3e} {elapsed:13.3f}", flush=True)


def callback(x_k):
    convergence_history.append(objective_np(x_k))
    infeasibility_history.append(np.max(np.abs(constraints_np(x_k))))
    print_iteration(
        len(convergence_history) - 1, convergence_history[-1],
        infeasibility_history[-1], perf_counter() - solver_start,
    )


print("SLSQP convergence (iteration 0 is the feasible initial curve)")
print("Iteration energies need not decrease monotonically; the final result is reported below.")
print(f"{'Iter':>6} {'Bending energy':>18} {'Max |C-Q|':>18} {'Elapsed [s]':>13}")
print_iteration(0, J_init, infeasibility_history[0], 0.0)
solver_start = perf_counter()
result = minimize(
    objective_and_grad_np,
    x_init,
    jac=True,
    method="SLSQP",
    constraints=[{"type": "eq", "fun": constraints_np, "jac": constraints_jac_np}],
    callback=callback,
    options={"maxiter": 200, "ftol": 1e-12},
)
solver_time = perf_counter() - solver_start

x_opt = result.x
J_opt = float(result.fun)
constraint_residual = np.max(np.abs(constraints_np(x_opt)))

print(f"\nSLSQP success:             {result.success}")
print(f"SLSQP status:              {result.message}")
print(f"Iterations:                {result.nit}")
print(f"SciPy objective/gradient calls: {result.nfev} / {result.njev}")
print(f"Initial bending energy:    {J_init:.6e}")
print(f"Final bending energy:      {J_opt:.6e}   ({J_init / J_opt:.1f}x reduction)")
print(f"Max interpolation residual: {constraint_residual:.3e}\n")


# -------------------------------------------------------------------------------------------------------------------- #
# Validate the optimized curve on a grid much denser than the optimization
# quadrature: a fixed-node quadrature rule can be exploited by the optimizer
# (see module docstring), so the reported bending energy is only trustworthy
# if it agrees with a dense-grid estimate and the curve never nearly stalls.
# -------------------------------------------------------------------------------------------------------------------- #
validation_start = perf_counter()
curve_opt_check = nrb.NurbsCurve(control_points=unpack(jnp.asarray(x_opt)), degree=degree)
u_dense = jnp.linspace(0.0, 1.0, 2001)
dC_dense = curve_opt_check.get_derivative(u_dense, order=1)
speed_dense = jnp.linalg.norm(dC_dense, axis=0)
kappa_dense = curve_opt_check.get_curvature(u_dense)
J_dense = float(jnp.trapezoid(kappa_dense**2 * speed_dense, u_dense))
min_speed_dense = float(jnp.min(speed_dense))
validation_time = perf_counter() - validation_start
quadrature_relative_difference = abs(J_dense - J_opt) / J_opt

print("Dense-grid validation (2001 points, vs. the 120-node optimization quadrature)")
print(f"  Bending energy (quadrature): {J_opt:.6e}")
print(f"  Bending energy (dense grid): {J_dense:.6e}")
print(f"  Relative difference:        {quadrature_relative_difference:.2%}")
print(f"  Minimum parameter speed:     {min_speed_dense:.4e}  (should not be near zero)")
if quadrature_relative_difference > 0.05 or min_speed_dense < 1e-2:
    print("  WARNING: quadrature and dense-grid estimates disagree -- increase n_quad and re-run.\n")
else:
    print("  OK: quadrature and dense-grid estimates agree; no near-cusp detected.\n")


# -------------------------------------------------------------------------------------------------------------------- #
# Plots: control polygons and curves, curvature distributions, convergence history
# -------------------------------------------------------------------------------------------------------------------- #
plot_start = perf_counter()
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
# Clip only the plotted values: a feasible iterate can have exactly zero residual.
plot_residuals = np.maximum(infeasibility_history, np.finfo(float).eps)
line_infeas, = ax_infeas.plot(plot_residuals, "s--", color="tab:red", markersize=4, label="Max constraint infeasibility")
ax_infeas.set(ylabel="Max $|C(u_j) - Q_j|$", yscale="log")
ax_infeas.yaxis.label.set_color("tab:red")
ax_infeas.tick_params(axis="y", colors="tab:red")

ax.legend(handles=[line_energy, line_infeas], fontsize=8, loc="upper right")

print("Timing summary")
for name, elapsed in (
    ("Setup + feasible initial curve", setup_time),
    ("JAX first calls + warm calls", warmup_time),
    ("Gradient validation", gradient_check_time),
    ("SLSQP including iteration logging", solver_time),
    ("Dense-grid validation", validation_time),
    ("Plot preparation", perf_counter() - plot_start),
):
    print(f"  {name:<36} {elapsed:9.3f} s")
print(f"  {'Total before displaying figures':<36} {perf_counter() - demo_start:9.3f} s", flush=True)

plt.show()
