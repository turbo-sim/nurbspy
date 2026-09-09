# G² continuity of NURBS curves

This page defines geometric continuity between curve segments and
derives the endpoint curvature formulas needed to join Bézier,
B-spline, and NURBS curves smoothly, then applies them to airfoil
parametrization at the leading and trailing edges, closing with a
comparison to {cite:t}`Mykhaskiv2018`. The basis-function and
endpoint-derivative definitions are covered on the
[Bézier](bezier.md), [B-spline](bspline.md), and [NURBS](nurbs.md)
pages; each curvature section below recalls only what it needs.

## Motivation

In many science and engineering applications it is necessary to
develop a smooth geometric parametrization. The level of smoothness of
a curve or surface can be measured by its degree of *geometric
continuity*:

- $G^0$, or position continuity,
- $G^1$, or tangency continuity,
- $G^2$, or curvature continuity,
- and so on.

When the geometry is parametrized by a single Bézier, B-spline, or
NURBS curve, it is straightforward to ensure the desired level of
continuity at the interior points: the curve is polynomial or rational
there, hence smooth, subject to the knot-multiplicity guarantees
already stated on the family pages. However, when the geometry is
described with more than one curve segment, it is necessary to pay
special attention to the continuity at the connection points — for
instance, where the pressure and suction sides of an airfoil meet at
the leading and trailing edges.

For a regular space curve $\mathbf C(u)$, parametrized by $u$, define
the unit tangent and curvature vector

```{math}
:label: orig-curvature-vector
\mathbf T = \frac{\mathbf C'}{\|\mathbf C'\|},
\qquad
\mathbf K = \frac{\mathbf C'' - \mathbf T(\mathbf T\cdot\mathbf C'')}{\|\mathbf C'\|^2}.
```

The curvature magnitude and radius of curvature are

```{math}
:label: orig-curvature-definition
\kappa = \|\mathbf K\| = \frac{\|\mathbf C''\times\mathbf C'\|}{\|\mathbf C'\|^3},
\qquad \rho = \frac{1}{\kappa}.
```

Attaining $G^2$ continuity at a join means matching **both** the
oriented unit tangent and the curvature vector there, not merely the
scalar curvature or radius of curvature. Suppose $\mathbf C_A(1)$
joins $\mathbf C_B(0)$, with both segments following the intended
direction of traversal:

| Continuity | Requirement at the join |
| --- | --- |
| $G^0$ | $\mathbf C_A(1)=\mathbf C_B(0)$ |
| $G^1$ | $G^0$ and $\mathbf T_A(1)=\mathbf T_B(0)$ |
| $G^2$ | $G^1$ and $\mathbf K_A(1)=\mathbf K_B(0)$ |

Equal curvature *magnitudes* $\|\mathbf K_A\|=\|\mathbf K_B\|$ are not
enough: the two curvature vectors can point in opposite directions.
For example, an endpoint velocity $(1,0)$ paired with accelerations
$(0,2)$ and $(0,-2)$ gives the same tangent and the same unsigned
curvature on both sides, but opposite curvature vectors. Matching a
common radius of curvature $\rho=1/\kappa$ is therefore sufficient for
$G^2$ only once matching position and oriented tangent, and matching
curvature *direction*, are also imposed.

For regular segments with one-sided second derivatives, an equivalent
statement of $G^2$ is that there exist $\lambda>0$ and $\mu\in\mathbb R$
with

```{math}
:label: orig-g2-join
\mathbf C'_B(0) = \lambda\, \mathbf C'_A(1),
\qquad
\mathbf C''_B(0) = \lambda^2\, \mathbf C''_A(1) + \mu\, \mathbf C'_A(1),
```

in addition to coincident endpoints. These follow from the chain rule
for an orientation-preserving reparametrization $u\mapsto\tilde u(u)$
with $\tilde u'(0)=\lambda$ and $\tilde u''(0)=\mu$: unlike parametric
$C^2$ continuity, which forces $\lambda=1$ and $\mu=0$, geometric
continuity allows the two segments to run at different parameter
speeds. This is the sense in which the phrase "continuity of the
radius of curvature" is used below and in the original note: it always
means matching the vector quantities above, not merely their norms.

The usual approach to ensure smoothness at the point connecting two
curves is to add extra control points at suitable locations. This is
done, for instance, to ensure $G^2$ continuity at the points connecting
the pressure and suction surfaces of an airfoil. The objective of the
rest of this page is to derive expressions for the endpoint curvature
of Bézier and B-spline curves, use them to place the additional control
points required for $G^2$ continuity between two spline segments, apply
the result to an airfoil parametrization that meets $G^2$ continuity at
the leading and trailing edges, and finally compare these relations
with the expression used for the 2D blade parametrization in
{cite:t}`Mykhaskiv2018`, suggesting a correction for the case of
B-spline curves.

## Bézier curve endpoint curvature formulas

Recall the Bézier curve {eq}`def-bezier-curve` and its Bernstein basis
from the [Bézier page](bezier.md). Its endpoint value, first
derivative, and second derivative are

```{math}
\mathbf C(u=0)=\mathbf P_0, \qquad \mathbf C(u=1)=\mathbf P_n,
```

```{math}
\dot{\mathbf C}(u=0) = n(\mathbf P_1-\mathbf P_0),
\qquad
\dot{\mathbf C}(u=1) = n(\mathbf P_n-\mathbf P_{n-1}),
```

```{math}
\ddot{\mathbf C}(u=0) = n(n-1)\big[(\mathbf P_2-\mathbf P_0)-2(\mathbf P_1-\mathbf P_0)\big],
```

```{math}
\ddot{\mathbf C}(u=1) = n(n-1)\big[(\mathbf P_{n-2}-\mathbf P_n)-2(\mathbf P_{n-1}-\mathbf P_n)\big].
```

See {cite:t}`NURBS_book`, pp. 9–25, for proofs of these endpoint
formulas. Inserting the first and second derivatives into the radius
of curvature definition {eq}`orig-curvature-definition`, and noting that the
cross product of two parallel vectors is zero, gives the endpoint
curvature formulas already stated as {eq}`orig-bezier-endpoint-curvature`:

```{math}
:label: g2-bezier-curvature-u0
\kappa(u=0) = \left(\frac{n-1}{n}\right)
\frac{\big\|(\mathbf P_2-\mathbf P_0)\times(\mathbf P_1-\mathbf P_0)\big\|}
{\|\mathbf P_1-\mathbf P_0\|^3},
```

```{math}
:label: g2-bezier-curvature-u1
\kappa(u=1) = \left(\frac{n-1}{n}\right)
\frac{\big\|(\mathbf P_{n-2}-\mathbf P_n)\times(\mathbf P_{n-1}-\mathbf P_n)\big\|}
{\|\mathbf P_{n-1}-\mathbf P_n\|^3}.
```

## B-spline curve endpoint curvature formulas

Recall the B-spline curve {eq}`def-bspline-curve` and its Cox–de Boor
basis from the [B-spline page](bspline.md). Its endpoint value, first
derivative, and second derivative are

```{math}
\mathbf C(u=0)=\mathbf P_0, \qquad \mathbf C(u=1)=\mathbf P_n,
```

```{math}
\dot{\mathbf C}(u=0) = \frac{p}{u_{p+1}}(\mathbf P_1-\mathbf P_0),
\qquad
\dot{\mathbf C}(u=1) = \frac{p}{1-u_n}(\mathbf P_n-\mathbf P_{n-1}),
```

```{math}
\ddot{\mathbf C}(u=0) = \frac{p(p-1)}{u_{p+1}}\left[
\frac{1}{u_{p+2}}(\mathbf P_2-\mathbf P_0)
-\left(\frac{1}{u_{p+1}}+\frac{1}{u_{p+2}}\right)(\mathbf P_1-\mathbf P_0)
\right],
```

```{math}
\ddot{\mathbf C}(u=1) = \frac{p(p-1)}{1-u_n}\left[
\frac{1}{1-u_{n-1}}(\mathbf P_{n-2}-\mathbf P_n)
-\left(\frac{1}{1-u_n}+\frac{1}{1-u_{n-1}}\right)(\mathbf P_{n-1}-\mathbf P_n)
\right].
```

See {cite:t}`NURBS_book`, pp. 81–100, for a proof of these endpoint
formulas. As before, inserting the first and second derivatives into
{eq}`orig-curvature-definition` and cancelling the cross product of the
parallel tangential term gives the endpoint curvature formulas already
stated as {eq}`orig-bspline-endpoint-curvature`:

```{math}
:label: g2-bspline-curvature-u0
\kappa(u=0) = \left(\frac{p-1}{p}\right)\left(\frac{u_{p+1}}{u_{p+2}}\right)
\frac{\big\|(\mathbf P_2-\mathbf P_0)\times(\mathbf P_1-\mathbf P_0)\big\|}
{\|\mathbf P_1-\mathbf P_0\|^3},
```

```{math}
:label: g2-bspline-curvature-u1
\kappa(u=1) = \left(\frac{p-1}{p}\right)\left(\frac{1-u_n}{1-u_{n-1}}\right)
\frac{\big\|(\mathbf P_{n-2}-\mathbf P_n)\times(\mathbf P_{n-1}-\mathbf P_n)\big\|}
{\|\mathbf P_{n-1}-\mathbf P_n\|^3}.
```

## NURBS curve endpoint curvature formulas

Recall the NURBS curve {eq}`def-nurbs-curve` and its rational basis
from the [NURBS page](nurbs.md). Its endpoint value, first derivative,
and second derivative are

```{math}
\mathbf C(u=0)=\mathbf P_0, \qquad \mathbf C(u=1)=\mathbf P_n,
```

```{math}
\dot{\mathbf C}(u=0) = \left(\frac{p}{u_{p+1}}\right)\left(\frac{w_1}{w_0}\right)(\mathbf P_1-\mathbf P_0),
\qquad
\dot{\mathbf C}(u=1) = \left(\frac{p}{1-u_n}\right)\left(\frac{w_{n-1}}{w_n}\right)(\mathbf P_n-\mathbf P_{n-1}),
```

```{math}
\ddot{\mathbf C}(u=0) = \frac{p(p-1)}{u_{p+1}}\left[
\frac{1}{u_{p+2}}\frac{w_2}{w_0}(\mathbf P_2-\mathbf P_0)
-\left(\frac{1}{u_{p+1}}+\frac{1}{u_{p+2}}\right)\frac{w_1}{w_0}(\mathbf P_1-\mathbf P_0)
\right]
```

```{math}
{}+\frac{2p^2}{u_{p+1}^2}\frac{w_1}{w_0}\left(1-\frac{w_1}{w_0}\right)(\mathbf P_1-\mathbf P_0),
```

```{math}
\ddot{\mathbf C}(u=1) = \frac{p(p-1)}{1-u_n}\left[
\frac{1}{1-u_{n-1}}\frac{w_{n-2}}{w_n}(\mathbf P_{n-2}-\mathbf P_n)
-\left(\frac{1}{1-u_n}+\frac{1}{1-u_{n-1}}\right)\frac{w_{n-1}}{w_n}(\mathbf P_{n-1}-\mathbf P_n)
\right]
```

```{math}
{}+\frac{2p^2}{(1-u_n)^2}\frac{w_{n-1}}{w_n}\left(1-\frac{w_{n-1}}{w_n}\right)(\mathbf P_{n-1}-\mathbf P_n).
```

See {cite:t}`NURBS_book`, pp. 117–127, for a proof of these endpoint
formulas. Inserting the first and second derivatives into
{eq}`orig-curvature-definition`, and again cancelling the cross product of
the parallel tangential term, gives the endpoint curvature formulas
already stated as {eq}`orig-nurbs-endpoint-curvature`:

```{math}
:label: g2-nurbs-curvature-u0
\kappa(u=0) = \left(\frac{p-1}{p}\right)\left(\frac{u_{p+1}}{u_{p+2}}\right)
\left(\frac{w_0w_2}{w_1^2}\right)
\frac{\big\|(\mathbf P_2-\mathbf P_0)\times(\mathbf P_1-\mathbf P_0)\big\|}
{\|\mathbf P_1-\mathbf P_0\|^3},
```

```{math}
:label: g2-nurbs-curvature-u1
\kappa(u=1) = \left(\frac{p-1}{p}\right)\left(\frac{1-u_n}{1-u_{n-1}}\right)
\left(\frac{w_nw_{n-2}}{w_{n-1}^2}\right)
\frac{\big\|(\mathbf P_{n-2}-\mathbf P_n)\times(\mathbf P_{n-1}-\mathbf P_n)\big\|}
{\|\mathbf P_{n-1}-\mathbf P_n\|^3}.
```

This equation applies to NURBS curves and includes Bézier and B-spline
curves as the special cases above.

## Application to airfoil parametrization

The endpoint curvature formulas for NURBS curves can be used to
specify the radius of curvature at the leading and trailing edges of
an airfoil and ensure $G^2$ continuity between the upper and lower
surfaces.

### Leading edge

Consider the construction of the suction side at the leading edge. The
suction side is defined by the set of control points
$\{\mathbf P_0,\mathbf P_2,\ldots,\mathbf P_{n-2},\mathbf P_n\}$, where
$\mathbf P_0$ is the start point of the camber line and $\mathbf P_2$
is computed from the thickness distribution. To impose a radius of
curvature $\rho=1/\kappa$ at the leading edge, insert an additional
control point $\mathbf P_1$ at distance $\ell_0=\|\mathbf P_1-\mathbf P_0\|$
from $\mathbf P_0$ along a chosen unit direction $\mathbf t$,

```{math}
\mathbf P_1 = \mathbf P_0 + \ell_0\,\mathbf t,
\qquad \ell_0>0.
```

In the original airfoil construction, $\mathbf t$ is the curve's unit
tangent direction at $u=0$, which for this construction happens to be
normal to the camber line. The distance $\ell_0$ is found by solving
{eq}`g2-nurbs-curvature-u0`, with $\mathbf P_1-\mathbf P_0=\ell_0\mathbf t$
and $\mathbf t\times\mathbf t=\mathbf 0$:

```{math}
:label: orig-leading-edge-distance
\ell_0^2 = \rho\left(\frac{p-1}{p}\right)\left(\frac{u_{p+1}}{u_{p+2}}\right)
\left(\frac{w_0w_2}{w_1^2}\right)\,
\big\|(\mathbf P_2-\mathbf P_0)\times\mathbf t\big\|.
```

This equation applies to NURBS curves and includes Bézier and
B-spline curves as special cases. If the component of
$\mathbf P_2-\mathbf P_0$ perpendicular to $\mathbf t$ vanishes, no
finite positive $\ell_0$ produces this curvature. Repeating the process
for the pressure side using the same radius of curvature, oriented
tangent, and curvature direction ensures $G^2$ continuity at the
leading edge.

### Trailing edge

The procedure for the trailing edge is analogous. Here, $\mathbf P_n$
is the endpoint of the camber line and $\mathbf P_{n-2}$ is computed
from the thickness distribution. Choose a unit direction
$\mathbf t_{\rm back}$ pointing from $\mathbf P_n$ toward the preceding
control point, and insert an additional control point

```{math}
\mathbf P_{n-1} = \mathbf P_n + \ell_1\,\mathbf t_{\rm back}, \qquad \ell_1>0.
```

The curve's forward tangent at $u=1$ then points along
$-\mathbf t_{\rm back}$. The distance $\ell_1$ is found by solving
{eq}`g2-nurbs-curvature-u1`:

```{math}
:label: orig-trailing-edge-distance
\ell_1^2 = \rho\left(\frac{p-1}{p}\right)\left(\frac{1-u_n}{1-u_{n-1}}\right)
\left(\frac{w_nw_{n-2}}{w_{n-1}^2}\right)\,
\big\|(\mathbf P_{n-2}-\mathbf P_n)\times\mathbf t_{\rm back}\big\|.
```

This equation applies to NURBS curves and includes Bézier and
B-spline curves as special cases. Repeating the process for the
pressure side using the same radius of curvature ensures $G^2$
continuity at the trailing edge.

Both constructions only fix a control polygon that reproduces a chosen
radius of curvature; verifying full $G^2$ continuity between the two
sides still requires checking that positions coincide and that the
oriented tangents and curvature vectors agree, as in the
[Motivation](#motivation) section above, including a consistent
traversal direction for the closed profile.
