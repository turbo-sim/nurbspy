# Bézier, B-spline, and NURBS surfaces

## Control nets and paired parameters

A surface uses two independent parameters. Its control-point array has
shape `(d, n+1, m+1)`: coordinate, $u$ index, then $v$ index. Its weights
have shape `(n+1, m+1)`.

After this comparison of constructor inputs, the surface walkthroughs
progress from a [bilinear patch](bilinear_surface.md) to a
[ruled surface](ruled_surface.md), a [full cylinder by extrusion](extruded_surface.md),
a [torus by revolution](revolution_surface.md), and a
[Coons patch](coons_patch.md).

| Geometry | Constructor arguments |
| --- | --- |
| Polynomial Bézier | `control_points=P` |
| Rational Bézier | `control_points=P, weights=W` |
| Polynomial B-spline | `control_points=P, u_degree=p, v_degree=q, u_knots=U, v_knots=V` |
| NURBS | The B-spline arguments plus `weights=W` |

A Bézier net with shape `(3, 4, 3)` has degrees $(3,2)$.
Supplying degrees $(2,2)$ instead gives a B-spline with an interior
$u$ knot. If both knot vectors are omitted, the constructor generates
normalized clamped vectors. For explicit knots, supply both vectors and
both degrees, with `len(U)=n+p+2` and `len(V)=m+q+2`.

**`get_value(u, v)` evaluates paired parameter samples.** It does not
form their Cartesian product. To sample a grid, use `meshgrid`, flatten
both grids, evaluate, and reshape the returned coordinates.

## Complete comparison script

Run `python demos/documentation/surface_comparison.py`, or
{download}`download the script <../../../demos/documentation/surface_comparison.py>`.

```{literalinclude} ../../../demos/documentation/surface_comparison.py
:language: python
```

## Output and interpretation

The `41 × 31` grid has 1,271 parameter pairs, so `S.shape` is
`(3, 1271)`. If needed, `S.reshape(3, 41, 31)` recovers the grid layout.
Here `surface.plot()` performs its own sampling and draws the surface,
control net, boundaries, and isoparametric curves on each subplot.

| Surface | Degrees | Position at $(u,v)=(0.5,0.5)$ |
| --- | --- | --- |
| Bézier | $(3,2)$ | $(1.5,\ 1.0,\ 1.25)$ |
| B-spline | $(2,2)$ | $(1.5,\ 1.0,\ 1.5)$ |
| NURBS | $(2,2)$ | $(1.5,\ 1.0,\ 1.75)$ |

```{figure} images/surface_comparison.png
:alt: Three surfaces on an identical control net, showing the effect of lower degree and larger interior weights.

The same net defines a Bézier patch, a B-spline surface, or a NURBS surface.
```

The NURBS weights pull the surface toward the two elevated interior
control points. All three surfaces interpolate the four corners because
the knot vectors are clamped. Interior control points generally are not
interpolation points.

Use `surface.get_derivative(u, v, order_u=1, order_v=0)` for
$\mathbf S_u$ and switch the orders for $\mathbf S_v$.
`surface.get_normals(u, v)` returns unit normals at regular points;
`surface.get_curvature(u, v)` returns `(mean_curvature, gaussian_curvature)`.
The [NURBS theory](../theory/nurbs.md) explains these quantities.
