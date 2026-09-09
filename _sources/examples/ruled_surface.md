# Ruled surface

A ruled surface connects corresponding points on two boundary curves by
straight lines. For the polynomial curves in this example,

```{math}
\mathbf S(u,v)=(1-v)\mathbf C_1(u)+v\mathbf C_2(u).
```

The $u$ parameter follows each generating curve; increasing $v$ moves
along a straight ruling between them.

## Inputs

{class}`nurbspy.nurbs_surface_ruled.NurbsSurfaceRuled` takes two curves
with matching control-point array shapes, degrees, and knot vectors.
The example uses two quadratic Bézier curves with unit weights, so those
requirements hold automatically. Its surface has degrees $(2,1)$.

For rational input curves with different weight functions, this helper
blends their homogeneous representations. Each ruling remains straight,
but `v` is generally a rational parametrization along it; the linear
blend above applies when both weight functions agree, as they do here.

## Complete script

Run `python demos/documentation/ruled_surface.py`, or
{download}`download the script <../../../demos/documentation/ruled_surface.py>`.

```{literalinclude} ../../../demos/documentation/ruled_surface.py
:language: python
```

## Output

The control net has shape `(3, 3, 2)`. The midpoint is
`[1.5    1.     0.7375]`. The script checks the linear-blend formula at
61 values of $u$ with $v=0.35$; the error is at floating-point roundoff.

```{figure} images/ruled_surface.png
:alt: Two highlighted curved boundaries connected by straight rulings.

The orange curves generate the patch; the straight black lines hold $u$ fixed.
```

`surface.plot()` draws the patch and its isoparametric curves.
Each boundary's `plot_curve()` method highlights the generating geometry.
