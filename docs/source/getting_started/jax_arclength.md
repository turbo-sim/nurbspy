# JAX arc length and distance parameterization

`nurbspy.jax.NurbsCurve.get_arclength()` uses knot-aware adaptive Gauss–Kronrod
quadrature by default. It integrates the analytic speed $v(u)=\|C'(u)\|$ over
nonzero knot spans, including partial spans at the requested limits. The
explicit ODE alternative integrates $s'(u)=v(u)$ with forced boundaries at knots.

```python
import jax.numpy as jnp
import nurbspy.jax as nrb

curve = nrb.NurbsCurve(jnp.array([[0., 0.5, 1.], [0., 0., 1.]]))
length = curve.get_arclength()                         # adaptive, tol=1e-6
length = curve.get_arclength(atol=1e-10, rtol=1e-9)      # separate controls
length = curve.get_arclength_ode(tol=1e-9)              # explicit ODE
info = curve.get_arclength(return_info=True)
```

`tol` supplies each unspecified `atol` and `rtol`. Quadrature requests the
combined target $\max(\mathrm{atol},\mathrm{rtol}|L|)$. ODE tolerances control
local integration errors and are not a global length-error guarantee. Both
methods return a signed integral when the limits are reversed. Equal limits
give zero while retaining the endpoint derivatives of the signed integral.

The existing `get_arclength_quad(n_points=41)` API retains its fixed global
Clenshaw–Curtis rule for compatibility. It is not adaptive. Prefer
`get_arclength()` when accuracy must be controlled by tolerances; a fixed rule
can miss narrow knot or rational-weight features.

## Failure and differentiation

The adaptive and ODE methods require real finite geometry, positive rational
weights and continuous curves. A scalar call raises if the solver fails or the
inputs are invalid, including under JIT. With `return_info=True`, an
`ArcLengthInfo` contains `value`, `error_estimate`, `evaluations`, `status` and
`intervals`. Status zero means successful termination. Quadrature statuses
1–8 are the public Quadax statuses; additional codes are 16 for invalid inputs,
32 for nonfinite sampled speed/value and 64 for unsuccessful ODE termination.
For ODE, `error_estimate` is NaN and `evaluations` is -1; `intervals` counts
attempted steps. Unknown diagnostics are not invented from local tolerances.

Quadrature uses public Quadax APIs. Its default `adjoint="direct"` differentiates
the accepted discretization. `adjoint="leibniz"` performs a separate adaptive
derivative integration and can be useful for repeated optimization evaluations.
Neither a primal error estimate nor a successful status certifies gradient
accuracy. Tighter controls and independent sensitivity checks remain useful.
See the [Quadax differentiation documentation](https://quadax.readthedocs.io/en/latest/differentiation.html).

Length may be zero or have isolated zero speed. At exactly zero velocity the
implementation selects a zero norm subgradient for AD. This convention does
not make length differentiable at a wholly collapsed curve.

Large positive weight ratios trigger endpoint grading; normalization and
reflection improve resolution of endpoint layers. These protections and
finite-speed probes cannot prove that every narrow feature was sampled.
Embedded error estimates are not certificates, and an extreme case can report
failure rather than a reliable length. Increase `max_intervals` when a workspace
limit is the cause; more workspace cannot cure floating-point resolution.

## Repeated changing-design objectives

```python
table = nrb.prepare_arclength(curve, order=16, subdivisions=1)
length = table.value(new_control_points, new_positive_weights)
```

The table stores Gauss nodes and polynomial basis derivatives. Control points
and weights are supplied on each evaluation and remain differentiable. Omitted
weights mean unit weights. Rebuild for changed knots, degree, integration
limits, order or subdivisions. Keep preparation inside the differentiated
function if those inputs are optimization variables.

Prepared integration is optional and has no tolerance guarantee. Check it
against adaptive integration across the intended design family; doubling the
Gauss order is useful evidence but both orders can miss the same sharp layer.

## Dense arc-length map

```python
u_of_s, total_length = curve.reparametrize_by_arclength(tol=1e-6)
parameters = u_of_s(jnp.linspace(0, total_length, 100))
```

This retains the ODE approach for fast repeated queries. Adaptive quadrature
first computes each knot-span length. The reciprocal ODE $du/ds=1/v(u)$ is
then integrated independently on each span, using normalized coordinates and
dense output. Span endpoints are anchored exactly, preventing drift from one
knot interval into the next. Queries select an interpolant; they perform no
quadrature or root finding. Scalar/array inputs retain their shape and queries
outside $[0,L]$ raise an error.

The map requires positive speed. Sampled stationary points and flat spans are
rejected instead of hiding a reciprocal singularity with a speed floor.
Sampled regularity and dense-output monotonicity checks are not global proofs.
Isolated zero speed can have a unique continuous inverse, but that inverse may
have an infinite derivative; this regular ODE API does not support it.

`tol` is not a global inverse-map or unit-speed guarantee. Validate applications
through independent $s(u(s))\approx s$ and actual unit-speed checks, especially
for sensitivities. `steps_per_span` sets a minimum dense-output resolution;
`max_steps` limits dense storage per span (defaults 16 and 512 respectively).
Tighter tolerances and larger budgets are available.

The returned callable is a geometry snapshot. Rebuild it when geometry changes,
and keep construction inside a JAX transform for geometry sensitivities. On
regular curves both construction and dense evaluation support AD. A small
arc-length residual gives weaker parameter accuracy when speed is small:

$$
|\Delta u|\lesssim\frac{|\Delta s|+\text{integration error}}{v_{\min}}.
$$

The standalone `development/arclength/` benchmark imports these production APIs
and records construction, compilation, first and warm execution, independent
accuracy and changing-design workflows. Ordinary tests contain correctness
regressions and no machine-dependent timing thresholds. The NumPy backend's
existing integration API is unchanged by these JAX improvements.
