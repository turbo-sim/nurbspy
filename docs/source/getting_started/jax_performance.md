# JAX performance and prepared evaluations

Use `import nurbspy.jax as nrb` for the optional JAX backend. The default
`import nurbspy` interface uses NumPy. The JAX evaluator vectorizes the basis
and sample axes, leaving only degree and derivative-order recurrences as static
loops. Curve and surface derivatives compute only the polynomial rows needed
for the requested order. This reduces the computation graph JAX must compile
while preserving the general API, dynamic knots and sample parameters.

The rational quotient-rule formulas and geometric definitions are unchanged.
Reordered floating-point operations can give small rounding differences.
Derivative orders above the polynomial degree still matter for rational
geometry; they are not discarded.

For repeated evaluations at **fixed knots and sample locations**, explicitly
prepare polynomial basis tables. This avoids rebuilding their recurrence as
control points or weights change. Preparation adds no dependencies beyond the
existing JAX extra, and there is no hidden global cache or automatic invalidation.
The general evaluators remain useful when sample parameters move, including
adaptive integration and point-projection refinement.

## Curves

```python
import jax
import jax.numpy as jnp
import nurbspy.jax as nrb

P = jnp.array([[0., 0.5, 1.], [0., 1., 0.]])
W = jnp.array([1., 0.8, 1.])
curve = nrb.NurbsCurve(P, W)
u = jnp.linspace(0., 1., 101)
table = nrb.prepare_curve(curve, u, max_order=2)

points = table.get_value(curve.P, curve.W)
derivatives = table.get_derivatives(curve.P, curve.W)
tangent = table.get_tangent(curve.P, curve.W)
curvature = table.get_curvature(curve.P, curve.W)
```

`get_derivative(P, W, order=1)` selects one derivative. Values have shape
`(ndim, N)`; all derivatives have shape `(max_order+1, ndim, N)`.
Scalars are accepted and produce a sample axis of length one.

**Omitting `W` means unit weights**, not the weights of the original curve.
Prepared basis objects retain neither control points nor weights. Pass both
explicitly for rational geometry. New P/W arrays must match the prepared
control count; their values may change without rebuilding the table.

## Surfaces and tensor grids

```python
# surface is an existing nrb.NurbsSurface.
u_axis = jnp.linspace(0., 1., 64)
v_axis = jnp.linspace(0., 1., 48)
grid = nrb.prepare_surface(
    surface, u_axis, v_axis, max_order_u=2, max_order_v=2, grid=True
)
points = grid.get_value(surface.P, surface.W)       # (ndim, 64, 48)
normal = grid.get_normals(surface.P, surface.W)    # (3, 64, 48)
mean, gaussian = grid.get_curvature(surface.P, surface.W)
du = grid.get_derivative(surface.P, surface.W, order_u=1, order_v=0)
```

`grid=True` takes independent 1D axes in `ij` order: `points[:, i, j]` is
$S(u_i,v_j)$. The default `grid=False` takes scalar or equal-length paired
parameters, matching `NurbsSurface`:

```python
paired = nrb.prepare_surface(surface, u_samples, v_samples)
points = paired.get_value(surface.P, surface.W)  # (ndim, N)
```

`get_derivatives(P, W)` returns all partials through the prepared maxima:

| Layout | Full derivative shape |
| --- | --- |
| Paired | `(max_order_u+1, max_order_v+1, ndim, N)` |
| Grid | `(max_order_u+1, max_order_v+1, ndim, Nu_samples, Nv_samples)` |

Surface basis objects also accept changing P/W, with the original control-net
shape. Omitting W again means unit weights. Normals and surface curvatures
require a regular 3D patch; curve curvature supports regular 2D/3D curves.

## Lifetime, JIT and automatic differentiation

Rebuild a basis table after changing knots, degree, control count, sample
locations, required derivative maxima, or paired/grid layout. Evaluation
methods intentionally accept only P/W; they cannot detect changes to the
geometry from which a table was originally prepared.

Orders must be static nonnegative integers. A requested order may exceed the
degree, but cannot exceed the prepared maximum. Homogeneous polynomial rows
above degree are zero; the rational quotient recurrence still computes every
requested rational derivative. It follows the identities $wC=A$ and $wS=A$.
Static binomial coefficients use `math.comb` to express this formula directly.

Prepared objects are Equinox/JAX PyTrees with array tables and static shape
metadata. Their evaluation methods use `eqx.filter_jit`, like the geometry
classes, and support JAX batching and forward/reverse differentiation in P/W.
Pass changing data as array arguments to a reused compiled function:

```python
# targets has shape (ndim, N); it can change without rebuilding the table.
@jax.jit
def fitting_loss(P, W, targets):
    residual = table.get_value(P, W) - targets
    return jnp.mean(residual**2)

value_and_grad = jax.jit(jax.value_and_grad(fitting_loss, argnums=(0, 1)))
value, (grad_P, grad_W) = value_and_grad(curve.P, curve.W, targets)
```

A table prepared outside a differentiated function fixes knots and sample
locations. For their sensitivities, put preparation **inside** that function:

```python
def loss(P, W, U, u):
    changing_curve = nrb.NurbsCurve(P, W, degree=2, knots=U)
    changing_table = nrb.prepare_curve(changing_curve, u, max_order=1)
    derivative = changing_table.get_derivative(P, W, order=1)
    return jnp.sum(derivative**2)

gradients = jax.grad(loss, argnums=(0, 1, 2, 3))(P, W, curve.U, u)
```

There is no `stop_gradient` in preparation. This second pattern recomputes
basis work as knots/parameters change, so the general evaluator is often
simpler. Knot and parameter derivatives require a smooth region or the basis
evaluator's one-sided span convention. Independently splitting repeated knots
is not covered by a smooth-sensitivity guarantee.

## Repeated surface point projection

```python
projection = nrb.prepare_surface_projection(surface)
# Query arrays have shape (ndim, N); returned parameters have shape (2, N).
uv = projection.project_points(queries, ftol=1e-12, gtol=1e-10)
other_uv = projection.project_points(other_queries, ftol=1e-12, gtol=1e-10)
single_uv = projection.project_point(queries[:, 0])  # (2,)
```

This object has a different lifetime: it snapshots the **complete surface**
and its coarse spatial initialization grid. Rebuild it after changing P, W,
knots or degrees. Replacing just its geometry leaves with `eqx.tree_at` would
leave stale grid points. The snapshot uses the general solver's knot-midpoint
and uniform grid and four-start selection. Newton and line-search parameters
keep moving, so refinement uses the general evaluator and the same solver
settings and implicit derivative equations. It does not replace projection
with nearest-grid lookup.

Query differentiation is supported. For geometry differentiation, construct
the snapshot inside the differentiated function. Implicit sensitivities require
a locally unique regular minimum, a nonsingular reduced Hessian and a stable
active set. Seed ties, basin changes, singularities and active-set transitions
have no guaranteed derivative. Multistart local projection does not guarantee
the global minimum on an arbitrary folded surface. The bounded search retains
the existing normalized parameter domain $[0,1]^2$.

## Cost, memory and numerical limits

Preparation has its own evaluation, tracing/compilation and storage cost. It
can cost more for a single call; reuse is what can make it worthwhile. It does
not enable a persistent compilation cache. Different shapes, degrees, orders
or static arguments can still require new JAX specializations. A table can be
passed as a PyTree argument or closed over in a compiled function; constant
folding can differ between those choices.

A curve table stores `(min(p,max_order)+1) * n_control * n_samples` elements.
A surface stores the **sum** of the two 1D table sizes rather than their tensor
product. Grid outputs and derivative arrays still grow with both sample axes.
Choose the smallest order needed: zero for values, one for length/area/normals,
and two for curvature. High rational orders add quotient-rule arithmetic.
Projection preparation only saves initialization work; XLA may already hoist
some of it in a general batched call, so warm gains can be modest.

Basis tables use the backend's default real dtype, currently float64. Curve
derivative outputs preserve P's dtype; surfaces retain their usual promotion
rules. Prefer float64 P/W when accuracy matters. Near-zero above-degree
rational derivatives for unit weights can contain cancellation residuals;
the polynomial helper provides exact padded zeros when needed.

Preparation changes neither geometry regularity nor integration accuracy.
Zero rational denominators, zero-speed curves and degenerate surface metrics
remain singular. Fixed-node quadrature needs a convergence check for each
problem, particularly across knot spans or with extreme rational weights.
Freezing quadrature nodes also fixes their knot dependence. Adaptive routines
and moving-parameter workflows should continue to use the general evaluator.
