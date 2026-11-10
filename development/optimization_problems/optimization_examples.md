# Geometry optimization examples for nurbspy + JAX

This note proposes five numerical geometry-optimization examples that can showcase the automatic-differentiation capabilities of `nurbspy.jax`.

The main idea is that the geometry is itself part of the differentiable model. The optimization variables are NURBS control-point coordinates, and quantities such as curve coordinates, derivatives, arc length, and curvature are evaluated through `nurbspy.jax`. JAX then differentiates the complete objective and constraint functions with respect to the design variables.

The curve examples below can be implemented with the existing JAX curve backend. The minimum-area surface example should be treated as a follow-on example once `NurbsSurface` has a JAX-compatible implementation.

## General implementation strategy

For all curve problems, use a planar cubic NURBS or B-spline curve

$$
\mathbf C(u; \mathbf P), \qquad u\in[0,1],
$$

where the control-point coordinates

$$
\mathbf P = [\mathbf P_0,\ldots,\mathbf P_{n-1}]
$$

are the optimization variables. For the first implementations, keep the degree, knot vector, and weights fixed. Optimizing weights can be explored later, but varying only the control points makes the examples easier to interpret.

Use

```python
import jax
import jax.numpy as jnp
import nurbspy.jax as nrb
```

and define the objective as a pure function of a flattened design vector:

```python
def objective(x):
    P = unpack_control_points(x)
    curve = nrb.NurbsCurve(control_points=P, degree=3)
    ...
    return J
```

Then obtain exact derivatives of the discretized problem with

```python
value_and_grad = jax.jit(jax.value_and_grad(objective))
```

or

```python
grad_objective = jax.jit(jax.grad(objective))
```

For constrained optimization, a practical first choice is `scipy.optimize.minimize` with `method="SLSQP"` or `"trust-constr"`. Wrap the JAX objective and constraint Jacobians so that SciPy receives NumPy arrays.

For integrals, prefer a fixed quadrature rule during optimization. Gauss-Legendre quadrature is a good choice because the number and location of evaluation points remain fixed, giving JAX a static and reproducible computation graph. A simple dense trapezoidal rule is acceptable for an initial demonstration, but Gauss-Legendre quadrature is preferable for the final documented examples.

Define quadrature nodes \(u_q\) and weights \(w_q\) once outside the objective.

For a planar curve,

$$
ds = \|\mathbf C'(u)\|\,du.
$$

The curvature is

$$
\kappa(u)
=
\frac{x'(u)y''(u)-y'(u)x''(u)}
{\left(x'(u)^2+y'(u)^2\right)^{3/2}}.
$$

When only \(\kappa^2\) is required, compute it directly without taking an absolute value:

$$
\kappa^2
=
\frac{\left(x'y''-y'x''\right)^2}
{\|\mathbf C'\|^6}.
$$

This is smooth and convenient for automatic differentiation.

Add a small numerical regularization only if needed when evaluating quantities containing powers of \(\|\mathbf C'\|\). A valid curve should normally have nonzero parameter speed everywhere.

For every example, verify the JAX gradient at the initial design with a finite-difference or complex-step directional derivative before running the optimization.

---

## 1. Minimum-bending-energy curve through prescribed points

### Motivation

Given several points in the plane, find a smooth curve that passes through them while minimizing its bending energy.

This is a classical fairness problem. The objective penalizes curvature everywhere along the curve, so the optimizer searches for the least-bent curve compatible with the interpolation constraints.

The problem gives a very clear demonstration of differentiable geometry because the objective depends on first and second curve derivatives, curvature, numerical integration, and finally the NURBS control points.

### Mathematical problem

Let the prescribed points be

$$
\mathbf Q_j,\qquad j=0,\ldots,m-1,
$$

and assign each point a fixed parameter value \(u_j\).

Minimize

$$
J(\mathbf P)
=
\int_C \kappa^2\,ds.
$$

Using the NURBS parameter,

$$
J(\mathbf P)
=
\int_0^1
\kappa(u)^2
\|\mathbf C'(u)\|
\,du.
$$

For a planar curve this can be evaluated as

$$
J(\mathbf P)
=
\int_0^1
\frac{
\left(x'(u)y''(u)-y'(u)x''(u)\right)^2
}{
\|\mathbf C'(u)\|^5
}
\,du.
$$

Subject to the interpolation constraints

$$
\mathbf C(u_j)=\mathbf Q_j.
$$

If the first and last prescribed points correspond to \(u=0\) and \(u=1\), the endpoints are automatically fixed by these constraints.

### Recommended demonstration

Use 5 or 6 prescribed points forming a visibly nontrivial path. Use perhaps 8 or 10 control points so that the curve has enough freedom to satisfy the interpolation constraints while changing its shape.

Construct an intentionally poor but feasible initial curve, then optimize the internal control points.

Show:

- prescribed points,
- initial control polygon and curve,
- optimized control polygon and curve,
- curvature \(\kappa(s)\) or \(\kappa(u)\) before and after optimization,
- initial and final bending energy,
- optimizer convergence history.

### Implementation notes

Use fixed interpolation parameters. Chord-length parameterization of the prescribed points is a sensible choice:

$$
u_j =
\frac{
\sum_{k=1}^{j}\|\mathbf Q_k-\mathbf Q_{k-1}\|
}{
\sum_{k=1}^{m-1}\|\mathbf Q_k-\mathbf Q_{k-1}\|
}.
$$

Do not optimize the \(u_j\) values in the first version. That creates an additional parameterization problem that distracts from the main demonstration.

Define an equality-constraint function that returns all interpolation residuals flattened into one vector:

```python
def interpolation_constraints(x):
    P = unpack_control_points(x)
    curve = nrb.NurbsCurve(control_points=P, degree=degree)
    C = curve.get_value(u_data)
    return (C - Q).reshape(-1)
```

Provide the constraint Jacobian to SciPy using `jax.jacrev`.

For the objective, evaluate first and second derivatives at the quadrature nodes and integrate the bending-energy density.

A useful validation is to check that the final curve satisfies the interpolation constraints to approximately optimizer tolerance and that the bending energy decreases substantially.

### Possible extension

Add prescribed endpoint tangent directions,

$$
\mathbf C'(0)\parallel \mathbf t_0,
\qquad
\mathbf C'(1)\parallel \mathbf t_1,
$$

to obtain a clamped fairing problem.

---

## 2. Shortest curve with bounded curvature

### Motivation

Find the shortest path between two configurations while respecting a maximum allowable curvature.

This is closely related to bounded-curvature path-planning problems such as the Dubins-path problem. It has a direct physical interpretation: a vehicle, robot, tool path, or streamline-like trajectory cannot turn with an arbitrarily small radius.

The problem is particularly good for illustrating the difference between unconstrained and constrained shape optimization. Without a curvature bound, the shortest path is simply a straight segment. With prescribed endpoint directions and a maximum curvature, a nontrivial optimized path emerges.

### Mathematical problem

Minimize the arc length

$$
L(\mathbf P)
=
\int_C ds
=
\int_0^1
\|\mathbf C'(u)\|\,du.
$$

Subject to fixed endpoints,

$$
\mathbf C(0)=\mathbf Q_0,
\qquad
\mathbf C(1)=\mathbf Q_1,
$$

prescribed endpoint tangent directions,

$$
\mathbf C'(0)\parallel \mathbf t_0,
\qquad
\mathbf C'(1)\parallel \mathbf t_1,
$$

and the curvature bound

$$
|\kappa(u)| \le \kappa_{\max}
\qquad
\forall u\in[0,1].
$$

Equivalently,

$$
\kappa(u)^2 \le \kappa_{\max}^2.
$$

The squared form is preferable numerically because it avoids the nondifferentiability of an absolute value.

### Recommended demonstration

Choose start and end points whose prescribed tangent directions make the straight-line connection impossible.

For example:

- start at \((0,0)\) with tangent pointing to the right,
- end at \((1,1)\) with tangent pointing to the left or upward,
- choose a moderate value of \(\kappa_{\max}\).

Plot:

- initial curve,
- optimized curve,
- start and end tangent arrows,
- curvature distribution,
- horizontal lines at \(\pm\kappa_{\max}\),
- optionally the local minimum turning radius \(R_{\min}=1/\kappa_{\max}\).

The optimized curve should visibly exploit the curvature bound where shortening the path is advantageous.

### Implementation notes

For a clamped B-spline/NURBS curve, the endpoints can often be fixed conveniently by removing the endpoint control-point coordinates from the design vector. This is preferable to using equality constraints when possible.

Endpoint tangent-direction constraints can also be enforced geometrically through the first two and last two control points for a clamped curve. An especially robust implementation is to parameterize those neighboring control points along the prescribed tangent rays, leaving only their scalar distances from the endpoints as design variables.

If tangent constraints are imposed numerically instead, in two dimensions require the cross product to vanish:

$$
C'_x t_y-C'_y t_x=0.
$$

An additional dot-product inequality can enforce the correct direction rather than the opposite direction:

$$
\mathbf C'\cdot\mathbf t > 0.
$$

The continuous curvature inequality must be discretized. Evaluate it at a reasonably dense set of fixed collocation points:

$$
g_q(\mathbf P)
=
\kappa_{\max}^2-\kappa(u_q)^2
\ge 0.
$$

After optimization, validate the solution on a much denser parameter grid than the optimization grid. If the validation grid reveals violations between collocation points, increase the number of curvature-constraint points and re-optimize.

This distinction should be stated explicitly in the example: the numerical problem enforces the curvature bound at discrete points, then verifies it on a denser grid.

### Possible extension

Add circular or polygonal obstacles and minimum-clearance constraints. The resulting problem becomes a smooth path-planning example with both curvature and collision constraints.

---

## 3. Short and smooth curve through waypoints

### Motivation

Pure minimum-length optimization tends to produce aggressive turning, while pure minimum-bending-energy optimization may produce unnecessarily long paths.

A natural compromise is to optimize both length and smoothness. This creates a simple multi-objective geometry problem in which one parameter controls the tradeoff between a short curve and a gently turning curve.

This is useful for trajectory design, tool paths, road or rail alignment, centerline generation, and many engineering preprocessing tasks.

### Mathematical problem

Require the curve to pass through prescribed waypoints

$$
\mathbf C(u_j)=\mathbf Q_j.
$$

Define the arc length

$$
L
=
\int_C ds
$$

and bending energy

$$
E_b
=
\int_C \kappa^2\,ds.
$$

Because \(L\) has dimensions of length and \(E_b\) has dimensions of inverse length, use a dimensionless objective rather than adding them directly.

Let \(L_{\mathrm{ref}}\) be a reference length, for example the polyline length through the waypoints. Minimize

$$
J
=
\frac{L}{L_{\mathrm{ref}}}
+
\lambda L_{\mathrm{ref}} E_b,
$$

where \(\lambda\) is dimensionless.

### Recommended demonstration

Use the same waypoint set for several optimizations, for example

$$
\lambda = 0,\quad 10^{-2},\quad 10^{-1},\quad 1.
$$

The precise values should be adjusted once the problem is implemented.

Plot all optimized curves together. The expected trend is:

- small \(\lambda\): shorter path with stronger turns,
- large \(\lambda\): smoother path with larger turning radii and potentially greater length.

Also show a small table containing:

- \(\lambda\),
- arc length,
- bending energy,
- maximum absolute curvature.

This makes the optimization tradeoff immediately visible.

### Implementation notes

Reuse the same curve representation, quadrature, and interpolation-constraint machinery as problem 1.

Implement separate reusable functions:

```python
def arc_length(P):
    ...

def bending_energy(P):
    ...

def objective(P, lam, L_ref):
    return arc_length(P) / L_ref + lam * L_ref * bending_energy(P)
```

JIT-compile the objective with \(\lambda\) passed as a numerical argument or construct one objective closure per optimization.

The same initial control points should be used for every value of \(\lambda\) if the purpose is a clean comparison.

A continuation strategy can also be useful: solve first for one \(\lambda\), then use the converged solution as the initial condition for a nearby value.

### Possible extension

Replace bending energy with a penalty on curvature variation,

$$
\int_C \left(\frac{d\kappa}{ds}\right)^2 ds,
$$

for an even stronger fairness criterion. This requires higher curve derivatives and is better treated as an advanced example.

---

## 4. Regularized NURBS fitting to noisy data

### Motivation

Given noisy measurements of an underlying curve, fit a NURBS curve that follows the data without reproducing every fluctuation.

A least-squares fit alone can overfit noisy data when many control points are used. Adding a curvature penalty produces a smooth approximation.

This is a useful example because it connects differentiable geometry to a standard inverse problem and makes the role of regularization very clear.

### Mathematical problem

Given measured points

$$
\mathbf Q_j,\qquad j=0,\ldots,m-1,
$$

with corresponding fixed curve parameters \(u_j\), define the data-misfit term

$$
E_{\mathrm{data}}
=
\frac{1}{m}
\sum_{j=0}^{m-1}
\|\mathbf C(u_j)-\mathbf Q_j\|^2.
$$

Define the bending-energy regularization

$$
E_b
=
\int_C \kappa^2\,ds.
$$

A dimensionless objective can be written as

$$
J
=
\frac{E_{\mathrm{data}}}{L_{\mathrm{ref}}^2}
+
\lambda L_{\mathrm{ref}} E_b.
$$

Here \(L_{\mathrm{ref}}\) can again be the data-polyline length or the bounding-box diagonal.

Unlike problems 1 and 3, the data points are not equality constraints. The curve is free to deviate slightly from them to achieve a smoother fit.

### Recommended demonstration

Generate synthetic data from a known smooth curve and add random noise.

Fit the same data using several values of the regularization coefficient:

- \(\lambda=0\): nearly pure least squares,
- moderate \(\lambda\): smooth fit,
- large \(\lambda\): deliberately over-smoothed fit.

Show:

- noisy data,
- true curve if synthetic data are used,
- fitted curves,
- control polygons.

Optionally report the RMS fitting error and bending energy for each solution.

This gives an intuitive visual explanation of regularization.

### Implementation notes

Assign \(u_j\) using chord-length parameterization of the noisy data. Keep these parameters fixed for the first version.

The fitting objective is unconstrained if the endpoints are not prescribed, which makes this the simplest optimization example numerically. It may therefore be the best first example to implement as a basic JAX optimization tutorial.

A practical design is:

```python
def loss(P_flat):
    P = P_flat.reshape(...)
    curve = nrb.NurbsCurve(control_points=P, degree=degree)

    C_data = curve.get_value(u_data)
    data_error = jnp.mean(jnp.sum((C_data - Q) ** 2, axis=0))

    Eb = bending_energy(curve)

    return data_error / L_ref**2 + lam * L_ref * Eb
```

Then use

```python
loss_and_grad = jax.jit(jax.value_and_grad(loss))
```

and pass the resulting gradient to an optimizer such as L-BFGS-B or BFGS.

If the problem has too many control points relative to data points, the regularization becomes especially useful.

### Optional advanced formulation

The parameter values \(u_j\) can also be optimized:

$$
\min_{\mathbf P,u_j}
\sum_j
\|\mathbf C(u_j)-\mathbf Q_j\|^2
+
\text{regularization}.
$$

This is a more faithful geometric fitting problem because the closest location on the curve is not known in advance.

However, the parameters must remain ordered,

$$
0=u_0 < u_1 < \cdots < u_{m-1}=1,
$$

which complicates the optimization. This should be treated as an extension rather than the initial example.

Another alternative is iterative reparameterization: optimize the control points, project each data point onto the current curve, update the \(u_j\), and repeat.

---

## 5. Minimum-area NURBS surface with fixed boundary

### Motivation

Given a closed boundary in three-dimensional space, find a surface spanning that boundary with minimum area.

This is the numerical analogue of a minimal-surface or soap-film problem. A physical soap film supported by a wire frame tends toward a surface that locally minimizes area.

This would be an excellent surface counterpart to the curve examples because the objective depends on both parametric surface derivatives and the complete interior control net.

### Current nurbspy consideration

At present, the JAX backend should be regarded as curve-focused. The NumPy API contains `NurbsSurface`, but a JAX-compatible NURBS surface implementation is needed before this example can demonstrate automatic differentiation through the surface geometry.

Therefore this example naturally serves two purposes:

1. implement and validate a JAX-compatible `NurbsSurface`,
2. demonstrate its usefulness through a nontrivial gradient-based surface optimization.

### Mathematical problem

Let

$$
\mathbf S(u,v;\mathbf P),
\qquad
(u,v)\in[0,1]^2
$$

be a tensor-product NURBS surface.

Its differential area element is

$$
dA
=
\left\|
\mathbf S_u\times\mathbf S_v
\right\|
\,du\,dv.
$$

The total area is

$$
A(\mathbf P)
=
\int_0^1\int_0^1
\left\|
\mathbf S_u(u,v)\times\mathbf S_v(u,v)
\right\|
\,du\,dv.
$$

Minimize

$$
A(\mathbf P)
$$

with the boundary of the NURBS patch fixed.

For a clamped tensor-product surface, this can be implemented by holding the boundary rows and columns of the control net fixed and optimizing only the interior control points.

### Recommended demonstration

Use a nonplanar closed boundary for which the solution is visually interesting.

A simple setup is a square or rectangular parameter-domain boundary whose four boundary curves lie in three dimensions, for example with alternating corners displaced upward and downward.

Construct an initial surface from the same boundary, perhaps using a Coons patch or a deliberately perturbed interior control net.

Then minimize the area by moving only the interior control points.

Show:

- boundary curves,
- initial surface and control net,
- optimized surface and control net,
- initial and final areas,
- convergence history.

A side-by-side or before/after 3D plot would make an excellent documentation figure.

### Implementation notes

A JAX surface implementation needs at least:

- tensor-product NURBS surface evaluation,
- first derivatives \(\mathbf S_u\) and \(\mathbf S_v\),
- compatibility with `jax.jit`,
- differentiation with respect to the control-point array.

For a control net of shape

```text
(3, n_u, n_v)
```

construct the optimization vector only from interior points:

```text
P[:, 1:-1, 1:-1]
```

while the four boundary strips remain fixed.

Use tensor-product Gauss-Legendre quadrature:

$$
A
\approx
\sum_i\sum_j
w_iw_j
\left\|
\mathbf S_u(u_i,v_j)
\times
\mathbf S_v(u_i,v_j)
\right\|.
$$

Because the boundary is fixed by construction, the problem is unconstrained in the interior control-point coordinates. This makes the optimization itself relatively simple once JAX surface differentiation exists.

Use `jax.value_and_grad` for the area and solve with BFGS or L-BFGS-B.

### Important mathematical caveat

Minimizing area over the finite-dimensional space of NURBS surfaces with a fixed degree, knot vectors, and control-net resolution gives the minimum-area surface available within that chosen NURBS approximation space. It is not necessarily the exact continuous Plateau solution.

Increasing the control-net resolution should improve the approximation. This could itself become a useful convergence study.

### Validation

Validate the surface implementation independently before optimization:

1. Compare JAX surface coordinates against the NumPy `NurbsSurface` for identical control points, weights, knots, and parameters.
2. Compare \(\mathbf S_u\) and \(\mathbf S_v\) against the existing NumPy implementation.
3. Verify JAX derivatives of total area with respect to selected control-point coordinates using finite differences or complex step where applicable.
4. Verify that the boundary remains unchanged after optimization.
5. Verify that the optimized surface area is lower than the initial area.
6. Repeat with a finer quadrature rule to confirm that the reported area is converged.

---

## Suggested implementation order

The examples should be implemented in increasing numerical complexity.

### Stage 1: regularized curve fitting

Implement problem 4 first.

It is unconstrained, visually clear, and provides the simplest demonstration that JAX can differentiate a meaningful geometry objective with respect to all control points.

### Stage 2: minimum bending energy

Implement problem 1 next.

This introduces exact interpolation constraints and demonstrates differentiation through second derivatives and curvature.

### Stage 3: length-smoothness tradeoff

Implement problem 3 by reusing the arc-length, bending-energy, and waypoint-constraint utilities from the previous examples.

This can become a particularly good documentation example because several solutions can be compared in one figure.

### Stage 4: bounded-curvature shortest path

Implement problem 2 after the basic constrained-optimization infrastructure is stable.

This introduces inequality constraints and requires careful post-optimization verification between the curvature collocation points.

### Stage 5: minimum-area surface

Implement the JAX NURBS surface backend and then solve problem 5.

This should be the main surface showcase.

---

## Suggested shared utilities

Avoid duplicating numerical code between examples. Create small reusable utilities for the demonstrations, unless equivalent functionality already belongs naturally in the package API.

Useful helpers include:

```python
gauss_legendre_1d(n)
curve_arclength(curve, nodes, weights)
curve_bending_energy(curve, nodes, weights)
curve_curvature(curve, u)
chord_length_parameterization(points)
pack_free_control_points(P)
unpack_free_control_points(x, P_fixed)
```

For the surface example:

```python
gauss_legendre_2d(n_u, n_v)
surface_area(surface, nodes_u, weights_u, nodes_v, weights_v)
pack_interior_control_points(P)
unpack_interior_control_points(x, P_boundary)
```

Keep plotting outside JIT-compiled functions. Optimization functions should contain only JAX-compatible numerical operations.

---

## Gradient validation

Every example should explicitly verify at least one gradient before optimization.

For a scalar objective \(J(\mathbf x)\), choose a random normalized direction \(\mathbf d\) and compare

$$
\nabla J(\mathbf x)^T\mathbf d
$$

against a central finite-difference directional derivative

$$
\frac{
J(\mathbf x+h\mathbf d)
-
J(\mathbf x-h\mathbf d)
}{
2h
}.
$$

Testing a directional derivative is more compact than checking every design variable and gives a strong end-to-end check of the complete objective.

For the documentation, report the relative discrepancy once, then proceed to optimization.

---

## What the examples should demonstrate about nurbspy

The main message should not simply be that JAX can optimize control points. The examples should make clear that `nurbspy.jax` allows the complete geometry calculation to remain differentiable.

A representative computational chain is

```text
control points
    ↓
NURBS curve
    ↓
curve coordinates and derivatives
    ↓
arc length / curvature / fitting error
    ↓
numerical quadrature
    ↓
objective and constraints
    ↓
JAX automatic differentiation
    ↓
gradient-based optimizer
```

The user therefore writes the geometric objective itself rather than deriving and maintaining separate analytical sensitivities with respect to every control-point coordinate.

That is the capability these examples should showcase.
