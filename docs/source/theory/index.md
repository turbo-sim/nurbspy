# Theory

These pages explain the mathematics behind `nurbspy`, from curve and surface
representations to geometric calculations. Equations are paired with practical
interpretations and figures evaluated using the package. The representation
and derivative material follows *The NURBS Book* {cite:t}`NURBS_book`.

## Representations and continuity

```{toctree}
:maxdepth: 1

bezier
bspline
nurbs
g2_continuity
```

## Curve and surface operations

The operation pages connect analytic derivatives to lengths, local frames,
surface measures and inverse problems. Each page distinguishes mathematical
assumptions from numerical safeguards, identifies the relevant backend APIs,
and links to reproducible figure-generation code.

```{toctree}
:maxdepth: 1

curve_arclength
curve_parameterization
frenet_serret
surface_isocurves
surface_integrals
surface_differential_geometry
point_projection
```

## References

```{toctree}
:maxdepth: 1

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

## Regenerate the illustrations

From the repository root, with the development dependencies and JAX extra
installed:

```bash
python docs/generate_theory_images.py
python docs/generate_theory_images.py --only frames surface_geometry
```

The scripts call `set_plot_options()` and save each numerical illustration as
a 300 dpi PNG, an SVG embedded in these pages and a vector PDF for publication.
They also check the illustrated calculations against exact formulas or
independent numerical references. The
{download}`generator <../../generate_theory_images.py>` and
{download}`figure-code notes <../../theory_figures/README.md>` describe the
workflow. The scripts share helper geometries and styling, so run them from
the repository using the commands above.
