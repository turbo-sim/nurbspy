# Surface of revolution: a torus

A torus is formed by revolving an offset circle about an axis in its plane.
Here the generating circle has radius $r=0.6$, its center is at $(R,0,0)$
with $R=2$, and the axis is the $z$ axis. A full $2\pi$ rotation creates
a ring torus, since $R>r$.

## Inputs and parameters

First create the exact circular profile with `CircularArc`. Then pass its
`.NurbsCurve` to
{class}`nurbspy.nurbs_surface_revolution.NurbsSurfaceRevolution`, together
with the axis point, axis direction, and angular limits.
The constructor keywords are `angle_start` and `angle_end`, in radians.

For this surface, **$u$ follows the revolution and $v$ follows the profile**.
Both parameters are normalized to $[0,1]$; they are not the angles themselves.
The familiar angular parametrization explains the geometry:

```{math}
\begin{aligned}
x&=(R+r\cos\phi)\cos\theta,\\
y&=(R+r\cos\phi)\sin\theta,\\
z&=r\sin\phi.
\end{aligned}
```

Its parameter-independent identity is

```{math}
\left(\sqrt{x^2+y^2}-R\right)^2+z^2=r^2.
```

## Complete script

Run `python demos/documentation/revolution_surface.py`, or
{download}`download the script <../../../demos/documentation/revolution_surface.py>`.

```{literalinclude} ../../../demos/documentation/revolution_surface.py
:language: python
```

## Output

The control net has shape `(3, 9, 9)` and degrees $(2,2)$.
Four rational quadratic spans close each circular direction.
The script verifies the tube radius $r$ and the coincidence of both
pairs of parameter boundaries, with errors near machine precision.

```{figure} images/revolution_surface.png
:alt: A full rational torus with its circular generating profile highlighted in orange.

Revolving the orange circle produces the torus; black lines show the two parameter directions.
```

The plot uses `surface.plot()` and `profile.plot_curve()`. Reducing the
revolution angle creates an open portion of the torus; other generating
curves produce other surfaces of revolution.
