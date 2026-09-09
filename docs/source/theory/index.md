# Theory

These pages present the mathematical foundations of `nurbspy`, covering Bézier, B-spline, and NURBS curves and surfaces. The material progresses from Bézier to B-spline and then to NURBS representations, treating curves before surfaces and developing the relevant definitions, basis-function properties, and derivative formulas along the way. Most of the material follows *The NURBS Book* {cite:t}`NURBS_book`.


```{toctree}
:maxdepth: 1

bezier
bspline
nurbs
g2_continuity
../references/bibliography
```

## Notation and assumptions

| Symbol | Meaning |
| --- | --- |
| $\mathbf P_i$, $\mathbf P_{i,j}$ | Curve control points or a surface control net |
| $n+1$, $m+1$ | Numbers of control points in the two parameter directions |
| $p$, $q$ | Polynomial **degrees**; the corresponding orders are $p+1$, $q+1$ |
| $U$, $V$ | Nondecreasing knot vectors, including repeated knots |
| $w_i$, $w_{i,j}$ | Control-point weights |
| $u$, $v$ | Parameters, distinct from spatial coordinates and arc length |

Unless stated otherwise, knots are normalized and clamped to $[0,1]$,
weights are strictly positive, and geometry is real. Derivatives at the
ends of a curve or surface are one-sided limits. Curvature requires a
nonzero first derivative.
