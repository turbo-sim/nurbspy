# Production fix discovered by this study

## Above-degree NumPy/Numba basis derivatives

For a degree-$p$ polynomial basis, the ordinary piecewise derivative of order
$k>p$ is zero. This does not describe distributional derivatives at knot jumps.
Rational NURBS curve derivatives can remain nonzero above the degree: the guard
applies to the polynomial basis, not to derivatives of the rational quotient.

`compute_basis_polynomials_derivatives` printed a warning but kept recursively
decreasing the degree. It reached negative-degree basis calculations and
accessed invalid indices. Pure NumPy raised `IndexError`; initial isolated Numba
probes terminated with native process exit code `3221226356` on this machine.
This was a correctness and native-memory-safety defect, not rounding.

The fix returns a zero array of shape `(n+1, number_of_samples)` immediately
after converting the samples. It preserves float/complex sample dtype and
scalar sample axes. No dependency, solver or integration default was changed.

## Cached recursion and non-clamped boundaries

The first guarded version passed fresh compilation tests, but a subsequent
process loading Numba's cached recursive function produced a Windows access
violation. The final implementation evaluates the same mathematical degree
recurrence iteratively, retaining `cache=True` without recursive call addresses.
It first constructs the basis of degree $p-k$, then applies $k$ derivative
recurrences while raising the degree back to $p$.

Lower-degree bases also need more than the final `n+1` entries. The old recursion
discarded right-hand neighbors, then indexed `N[i+1]` when forming derivatives.
Clamped terminal knots masked that problem by making the corresponding
denominator zero; non-clamped knots did not. The iterative recurrence retains
all lower-degree intervals and shrinks the array at each degree. Independent
SciPy cases now verify derivatives on a non-clamped active domain as well.

Highest-order derivatives can jump at non-clamped active-domain boundaries.
The non-clamped reference comparisons use the left limit at the upper boundary,
making the one-sided convention explicit rather than treating it as roundoff.

Focused regression cases in
`tests/geometry/test_basis_derivative_helpers.py` exercise degrees 0, 2 and 3,
two excess orders, scalar/vector samples, repeated knots/endpoints and NumPy
complex-step dtype. The shared fixture also checks JAX. Running with Numba
installed tests its compiled decorator; a separate `NUMBA_DISABLE_JIT=1` run
checks the pure Python/NumPy path.
An additional backend test evaluates the same compiled function in two
processes sharing a fresh disk cache and requires both to exit successfully.

`outputs/results/basis_bug_before_fix.json` preserves original failing records and the
original NumPy basis source hash. Main results and HTML are regenerated after
the fix. Numba cubic fourth-derivative requests remain isolated as a precaution
so a future regression cannot corrupt the whole study.

## Findings deliberately left visible

- NumPy's Frenet normal and JAX's rotated 2D tangent use different orientation
  conventions. The equivalent oriented-normal adapter agrees; deciding which
  convention `get_normal` should expose needs an API decision.
- Fixed-rule arc length can under-resolve a curve. NumPy's global Gauss-10 and
  JAX's global Clenshaw-Curtis-41 do not promise adaptive tolerance control.
  Replacing them changes numerical/performance behavior; this study does not
  silently turn a fixed rule into an adaptive method.
- Collapsed-curve frames and curvature/torsion are undefined. Normalizing NaN
  and zero policies across backends is an explicit contract decision.
- The larger curve projection batch exposes a NumPy/Numba local-minimum miss.
  Choosing better sampled or multiple initial guesses is an algorithm change;
  merely tightening local stopping tolerances does not fix its basin choice.
- Dense inverse-map parameter accuracy and derivative accuracy differ. Default
  local ODE controls do not guarantee unit speed to the same tolerance. The
  benchmark compares tighter tolerance/denser output without changing defaults.

The local counterexample in `outputs/results/projection_counterexample.json`
checks both candidate distances with the production NumPy evaluator itself:
parameter `0.4065919974` gives squared distance `0.02761046384`, while the
feasible parameter `0.2558535225` gives `0.02637660995`. A strictly better
feasible candidate proves the returned result is not a global minimum; no
global-optimality certificate for the independent search is needed for that
counterexample.
