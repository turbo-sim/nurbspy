# Coons patch

A Coons patch fills the region between four boundary curves. It blends
the ruled surfaces between opposite boundaries and subtracts their
shared bilinear corner contribution:

```{math}
\begin{aligned}
\mathbf S(u,v)
&=(1-v)\mathbf C_{\rm south}(u)+v\mathbf C_{\rm north}(u)\\
&\quad +(1-u)\mathbf C_{\rm west}(v)+u\mathbf C_{\rm east}(v)
-\mathbf B(u,v).
\end{aligned}
```

Here $\mathbf B$ is the bilinear interpolant of the four corners.
This expression describes the polynomial cubic Bézier example below.

## Boundary orientation and compatibility

South and north run **west to east**. West and east run **south to north**.
Their endpoints must agree:

| Corner | Matching endpoints |
| --- | --- |
| Southwest | `south(0) = west(0)` |
| Southeast | `south(1) = east(0)` |
| Northwest | `north(0) = west(1)` |
| Northeast | `north(1) = east(1)` |

{class}`nurbspy.nurbs_surface_coons.NurbsSurfaceCoons` requires matching
degrees, knot vectors, and control-point counts for opposite boundaries,
as well as matching corner positions and weights. All four curves here
are polynomial cubic Bézier curves with unit weights. This makes the
class's control-net blending agree with the classical Coons formula.

## Complete script

Run `python demos/documentation/coons_patch.py`, or
{download}`download the script <../../../demos/documentation/coons_patch.py>`.

```{literalinclude} ../../../demos/documentation/coons_patch.py
:language: python
```

## Output

The resulting control net has shape `(3, 4, 4)` and degrees $(3,3)$.
The script compares all four surface boundaries against the input curves,
at 61 samples per boundary. The discrepancies are at floating-point
roundoff.

```{figure} images/coons_patch.png
:alt: A curved Coons patch interpolating four highlighted nonplanar boundary curves.

The patch interpolates all four orange boundary curves.
```

`surface.plot()` draws the patch and its parameter lines, while
`curve.plot_curve()` highlights each boundary. Boundary interpolation
specifies positions along the edges; matching tangent planes or curvature
to adjacent surfaces requires additional constraints.
