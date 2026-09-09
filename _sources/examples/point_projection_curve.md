# Point projection onto a curve

## The point-inversion problem

Given a curve $\mathbf C(u)$ and an external point $\mathbf P$, point
projection (also called point inversion) finds the parameter $u^*$ that
minimizes the distance between them:

```{math}
u^* = \operatorname*{argmin}_{u\,\in\,[0,1]} \left\lVert \mathbf C(u) - \mathbf P \right\rVert.
```

At an interior minimum, $\mathbf C(u^*)$ is the closest point on the curve
to $\mathbf P$, and the vector between them is orthogonal to the tangent
there:

```{math}
:label: point-projection-orthogonality-curve
\big(\mathbf C(u^*) - \mathbf P\big)\cdot \mathbf C'(u^*) = 0.
```

If the closest point lies outside the parameter domain, the minimizer sits
at an endpoint ($u^*=0$ or $u^*=1$) instead, where this condition need not
hold: the projection is limited by the domain boundary rather than by the
curve bending away from $\mathbf P$.

## How `nurbspy` solves it

`curve.project_point_to_curve(P)` minimizes $\lVert\mathbf C(u)-\mathbf
P\rVert$ directly with SciPy's L-BFGS-B, using the analytic gradient of
that objective, which is exactly the orthogonality residual
{eq}`point-projection-orthogonality-curve` normalized by the distance. To
reduce the chance of converging to the wrong local minimum, it first
evaluates the objective at the midpoint of every knot span and starts the
local optimization from whichever gives the smallest distance.

Multi-start from knot-span midpoints is not a global-optimality guarantee:
a curve with several closely spaced local minima can still trap the
optimizer at the wrong one. Check contentious cases against
{eq}`point-projection-orthogonality-curve`, or against an independent grid
search, as done below.

`maxiter`, `ftol`, and `gtol` control the underlying L-BFGS-B solve and
default to `100`, `1e-6`, and `1e-6` respectively; pass tighter tolerances
or a larger `maxiter` for hard-to-converge projections, or looser ones
when only an approximate foot point is needed.

## Script

Run `python demos/documentation/point_projection_curve.py`, or
{download}`download the script <../../../demos/documentation/point_projection_curve.py>`.

```{literalinclude} ../../../demos/documentation/point_projection_curve.py
:language: python
```

## Output

| Target | $u^*$ | Foot point | Orthogonality |
| --- | --- | --- | --- |
| $(0.5,\ 2.5)$ | $0.1722$ | $(0.8758,\ 1.0252)$ | $-3.8\times10^{-7}$ |
| $(3.5,\ 2.5)$ | $0.8278$ | $(3.1242,\ 1.0252)$ | $3.8\times10^{-7}$ |
| $(1.0,\ -1.0)$ | $0.0000$ | $(0.0,\ 0.0)$ | $6.0$ |

```{figure} images/point_projection_curve.png
:alt: A NURBS curve with three external points connected by dashed lines to their closest points on the curve.

Two projections land in the curve's interior; the third is closest to its clamped start point.
```

The first two targets project to interior points, where the orthogonality
residual is at solver tolerance. The third target, $(1.0,-1.0)$, is closest
to the curve's clamped start point $\mathbf C(0)=(0,0)$: the minimizer sits
at the domain boundary $u^*=0$, so {eq}`point-projection-orthogonality-curve`
does not hold there, and the nonzero residual reflects that, not a solver
failure.
