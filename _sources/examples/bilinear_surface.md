# Bilinear surface

A bilinear patch is the simplest tensor-product surface: four corners
define a net of shape `(3, 2, 2)` and degrees $(1,1)$. Its coordinate
lines are straight, although four noncoplanar corners produce a curved
surface.

## Inputs and corner convention

{class}`nurbspy.nurbs_surface_bilinear.NurbsSurfaceBilinear` takes four
three-dimensional corner arrays. Its argument names follow this mapping:

| Argument | Surface parameter | Control-net entry |
| --- | --- | --- |
| `P00` | $(0,0)$ | `P[:, 0, 0]` |
| `P01` | $(1,0)$ | `P[:, 1, 0]` |
| `P10` | $(0,1)$ | `P[:, 0, 1]` |
| `P11` | $(1,1)$ | `P[:, 1, 1]` |

In this helper's notation,

```{math}
\mathbf S(u,v)
=(1-v)[(1-u)\mathbf P_{00}+u\mathbf P_{01}]
+v[(1-u)\mathbf P_{10}+u\mathbf P_{11}].
```

The helper's `.NurbsSurface` attribute exposes the usual evaluation and
plotting interface.

## Complete script

Run `python demos/documentation/bilinear_surface.py`, or
{download}`download the script <../../../demos/documentation/bilinear_surface.py>`.

```{literalinclude} ../../../demos/documentation/bilinear_surface.py
:language: python
```

## Output

The midpoint is `[1.    1.    0.575]`, the average of the four corners.
The script checks that each corner is interpolated exactly to numerical
precision. The red control net and black isoparametric curves are drawn by
`surface.plot(control_points=True, isocurves_u=7, isocurves_v=7)`.

```{figure} images/bilinear_surface.png
:alt: A nonplanar bilinear patch with four corner control points and straight coordinate lines.

Four corners define a surface that is linear in each parameter separately.
```
