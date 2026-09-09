# Bézier, B-spline, and NURBS curves

## One interface, different inputs

For a curve in $d$ dimensions with $n+1$ control points, `P` has shape
`(d, n+1)`. Each column is a control point. Use floating-point arrays for
both control points and weights.

| Geometry | Constructor arguments | Degree and knots |
| --- | --- | --- |
| Polynomial Bézier | `control_points=P` | Degree $n$; one clamped span |
| Rational Bézier | `control_points=P, weights=W` | Degree $n$; one clamped span |
| Polynomial B-spline | `control_points=P, degree=p, knots=U` | Supplied degree and knot vector; unit weights |
| NURBS | `control_points=P, weights=W, degree=p, knots=U` | Supplied degree, knot vector, and weights |

When a degree is supplied and knots are omitted, the constructor creates
a normalized clamped vector with evenly spaced interior knots. Explicit
knots, as used here, make the spans easy to inspect. `len(U)` must equal
`P.shape[1] + p + 1`.

## Complete comparison script

This script uses five shared control points. The Bézier has degree four;
the B-spline and NURBS have degree three, with a knot at $u=0.5$.
The NURBS increases the middle control point's weight from one to three.

Run `python demos/documentation/curve_comparison.py`, or
{download}`download the script <../../../demos/documentation/curve_comparison.py>`.

```{literalinclude} ../../../demos/documentation/curve_comparison.py
:language: python
```

## Output and interpretation

Each evaluation returns `C.shape == (2, 301)`, with coordinates in rows
and parameter samples in columns. At $u=0.5$:

| Curve | Degree | Position |
| --- | --- | --- |
| Bézier | 4 | $(1.5,\ 1.3125)$ |
| B-spline | 3 | $(1.5,\ 1.5)$ |
| NURBS | 3 | $(1.5,\ 1.75)$ |

```{figure} images/curve_comparison.png
:alt: Three curves sharing a control polygon, with the rational curve pulled toward the upper middle point.

Degrees and weights change the shape while clamping preserves the endpoints.
```

The Bézier uses all five Bernstein basis functions over one span. The
cubic B-spline has two spans and local basis support. The NURBS uses those
same cubic bases, but the larger central weight pulls it toward the
central control point. Equal weights recover the B-spline.

For derivatives use `curve.get_derivative(u, order=1)`; it returns the same
coordinate/sample layout. `curve.get_curvature(u)` returns one curvature
magnitude per sample. The parameter `u` is generally neither arc length
nor a Cartesian coordinate.

The next example shows why [rational weights are useful for circles](circular_arc.md).
