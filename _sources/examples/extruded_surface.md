# Extruded surface: a full cylinder

Extruding a full unit circle along the $z$ axis produces the cylindrical
surface

```{math}
\mathbf S(u,v)=\mathbf C(u)+2v(0,0,1),\qquad (u,v)\in[0,1]^2.
```

The cylinder has radius one and height two. This example constructs the
entire lateral surface, with coincident boundaries at $u=0$ and $u=1$.
Its two circular ends remain open.

## Inputs

`CircularArc` represents the full $2\pi$ circle exactly with four
rational quadratic spans. Use its `.NurbsCurve` attribute as the input
to {class}`nurbspy.nurbs_surface_extruded.NurbsSurfaceExtruded`.
The circle has three coordinate rows because the extrusion is spatial,
even though the initial profile lies in the $xy$ plane.

The extrusion helper normalizes `D` and uses `d` as the extrusion
distance. Its `.NurbsSurface` attribute exposes the resulting surface.
Here $u$ follows the circle, while $v$ follows the vertical extrusion.

## Complete script

Run `python demos/documentation/extruded_surface.py`, or
{download}`download the script <../../../demos/documentation/extruded_surface.py>`.

```{literalinclude} ../../../demos/documentation/extruded_surface.py
:language: python
```

## Output

The control net has shape `(3, 9, 2)` and degrees $(2,1)$.
Both rows of weights are inherited from the circle. At every sample
$x^2+y^2=1$ and $z=2v$, up to floating-point roundoff. The script also
checks that the two sides of the circumferential seam coincide.

A unit cylinder has Gaussian curvature $K=0$ and mean-curvature
magnitude $|H|=1/2$. The sign of $H$ depends on the surface normal.
The script prints the errors against these geometric properties.

```{figure} images/extruded_surface.png
:alt: A complete cylindrical lateral surface with circular and vertical isoparametric curves.

An exact circular profile extruded in a straight line produces a full cylinder.
```

The package's `surface.plot()` method draws the cylinder and its
isoparametric curves. Setting `control_points=True` also displays the
two rings of control points.
