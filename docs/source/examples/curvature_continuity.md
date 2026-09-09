# Curvature continuity

The [continuity theory](../theory/g2_continuity.md) permits different
parameter speeds at a smooth join. Here two cubic Bézier segments meet
at the origin. The second segment is constructed to satisfy

```{math}
\mathbf C_B'(0)=0.6\,\mathbf C_A'(1),\qquad
\mathbf C_B''(0)=0.6^2\,\mathbf C_A''(1)+0.4\,\mathbf C_A'(1).
```

The first equation fixes the first interior control point of segment B.
The second fixes its second interior point. Its last point is free.

Run `python demos/documentation/curvature_continuity.py`, or
{download}`download the script <../../../demos/documentation/curvature_continuity.py>`.

```{literalinclude} ../../../demos/documentation/curvature_continuity.py
:language: python
```

## Output

The endpoint speeds are `3.000000` and `1.800000`. Both unit tangents
are $(1,0)$ and both curvature vectors are $(0,2/3)$, up to roundoff.
Thus the join is $G^2$, while the unequal first derivatives mean it is
not parametrically $C^1$.

```{figure} images/curvature_continuity.png
:alt: Two cubic Bezier curves meeting with the same tangent and curvature but different control-polygon edge lengths.

Different control-polygon spacing can preserve the same geometric continuity.
```
