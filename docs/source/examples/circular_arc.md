# An exact circular arc

Polynomial Bézier curves cannot represent a nonconstant circular arc
exactly. A rational quadratic can. For the unit quarter circle, choose

```{math}
\mathbf P_0=(1,0),\quad \mathbf P_1=(1,1),\quad \mathbf P_2=(0,1),
\qquad (w_0,w_1,w_2)=(1,1/\sqrt2,1).
```

The first and last points lie on the circle; the middle point is the
intersection of their tangent lines. Providing `P` and `W` alone creates
a rational Bézier curve of degree two.

Run `python demos/documentation/circular_arc.py`, or
{download}`download the script <../../../demos/documentation/circular_arc.py>`.

```{literalinclude} ../../../demos/documentation/circular_arc.py
:language: python
```

## Output

The midpoint is `[0.70710678 0.70710678]`. The radius and curvature are
both one at every sample, with numerical errors close to machine
precision. Removing the weights produces a polynomial quadratic whose
midpoint is $(0.75,0.75)$, outside the unit circle.

```{figure} images/circular_arc.png
:alt: Exact rational quarter circle compared with a polynomial quadratic using the same three control points.

Weights make the rational quadratic follow the circle exactly.
```

For other radii, planes, or angle spans, use
{class}`nurbspy.nurbs_curve_circular_arc.CircularArc`. The constructor
returns a helper object; its `.NurbsCurve` attribute holds the curve.
Angles are in radians, while the resulting curve is evaluated with a
normalized parameter $u\in[0,1]$.
