# B-spline curves and surfaces

B-spline curves and surfaces generalize the Bézier construction to a
control net governed by a knot vector, giving local control over the
shape without raising the polynomial degree. This page defines the
B-spline basis functions and the curve or surface built from them, then
states their main properties — including local support, partition of
unity, and the convex-hull property — first for curves and then for
surfaces. Most of the material follows {cite:t}`NURBS_book`,
Chapters 2–3.

## B-spline curves

### Definition

A B-spline curve, shorthand for basis spline curve, is a parametric
curve defined by

```{math}
:label: def-bspline-curve
\mathbf C(u) = \sum_{i=0}^{n} N_{i,p}(u)\, \mathbf P_i,
\qquad 0\leq u\leq1,
```

where $p$ is the **degree** of the curve, the coefficients
$\mathbf P_i$ are called control points, and $N_{i,p}$ are the basis
functions defined on the nondecreasing knot vector $U$:

```{math}
U = [u_0,\ldots,u_r]\in\mathbb R^{r+1}
\qquad \text{with} \qquad r = n+p+1.
```

The B-spline basis functions are given by the recursive relation

```{math}
:label: orig-bspline-degree-zero
N_{i,0}(u) = \begin{cases}
1 & \text{if } u_i\leq u<u_{i+1},\\
0 & \text{otherwise},
\end{cases}
```

```{math}
:label: orig-cox-de-boor
N_{i,p}(u) = \frac{u-u_i}{u_{i+p}-u_i}\,N_{i,p-1}(u)
+ \frac{u_{i+p+1}-u}{u_{i+p+1}-u_{i+1}}\,N_{i+1,p-1}(u).
```

This recursive relation can produce a $0/0$ quotient, which is defined
to be zero by convention whenever the corresponding basis function has
no support there. The half-open intervals in
{eq}`orig-bspline-degree-zero` also need a right-endpoint convention: at a
clamped curve's final knot $u=1$, evaluate by the left limit, so that
$N_{n,p}(1)=1$ and every other basis function is zero there.

### Mathematical properties of B-spline basis functions

Here is a list of some important properties of the B-spline basis
functions:

1. The relation $r=n+p+1$ holds, where $n+1$ is the number of basis
   functions and $r+1$ is the number of elements of the knot vector $U$.

2. $N_{i,p}(u)$ is, at most, a polynomial of degree $p$ on each knot
   span.

3. Local support:
   1. $N_{i,p}(u)=0$ if $u$ is outside the interval
      $[u_i,u_{i+p+1})$.
   2. In any given knot interval $[u_{i_0},u_{i_0+1})$, at most $p+1$
      basis functions are nonzero, namely $N_{i,p}(u)$ with
      $i_0-p\leq i\leq i_0$.

4. Non-negativity: $N_{i,p}(u)\geq0$ for all $i$, $p$, and
   $u\in[0,1]$.

5. Partition of unity:

   ```{math}
   \sum_{i=0}^{n} N_{i,p}(u) = 1 \qquad \text{for all } u\in[0,1].
   ```

   In addition, for an arbitrary knot span $[u_{i_0},u_{i_0+1})$,

   ```{math}
   \sum_{i=i_0-p}^{i_0} N_{i,p}(u) = 1
   \qquad \text{for all } u\in[u_{i_0},u_{i_0+1}).
   ```

   This means that the sum of the nonzero basis functions of any knot
   span is unity.

6. Continuity and differentiability:
   1. The basis functions are infinitely differentiable in the
      interior of the knot intervals.
   2. At a knot of multiplicity $r$ with $1\leq r\leq p$, the basis
      functions are **guaranteed** to be $C^{p-r}$ continuous there; a
      knot of multiplicity $p+1$ can produce a jump discontinuity.
      Special choices of control points can make a curve built from
      this basis smoother than the guarantee.

7. Extrema: for $p\geq1$, $N_{i,p}(u)$ attains exactly one maximum in
   $u\in[0,1]$. A degree-zero basis function is piecewise constant on
   its span and, in general, has no isolated maximum.

8. The first derivative of the basis functions is, for $p\geq1$,

   ```{math}
   :label: orig-bspline-basis-derivative
   N'_{i,p}(u) = \frac{\mathrm dN_{i,p}}{\mathrm du}
   = p\left(\frac{N_{i,p-1}(u)}{u_{i+p}-u_i}
   - \frac{N_{i+1,p-1}(u)}{u_{i+p+1}-u_{i+1}}\right).
   ```

   This is proven by induction after a good deal of algebra. As in
   {eq}`orig-cox-de-boor`, a $0/0$ quotient is defined to be zero by
   convention.

9. The $k$-th order derivative of the basis functions is, for
   $p\geq k\geq1$,

   ```{math}
   N_{i,p}^{(k)}(u) = \frac{\mathrm d^{(k)}N_{i,p}}{\mathrm du^{(k)}}
   = p\left(\frac{N_{i,p-1}^{(k-1)}(u)}{u_{i+p}-u_i}
   - \frac{N_{i+1,p-1}^{(k-1)}(u)}{u_{i+p+1}-u_{i+1}}\right).
   ```

   This is derived by repeated differentiation, and again a $0/0$
   quotient is defined to be zero by convention. Derivatives of order
   greater than $p$ vanish on the interior of a span, and at a
   repeated interior knot, a derivative order beyond the guaranteed
   continuity should be read as a one-sided value.

10. When the first and last knots have multiplicity $p+1$, the knot
    vector is given by

    ```{math}
    U = \big[\underbrace{u_0,\ldots,u_p}_{p+1},\
    \underbrace{u_{p+1},\ldots,u_n}_{n-p},\
    \underbrace{u_{n+1},\ldots,u_{n+p+1}}_{p+1}\big],
    ```

    where $u_0=\cdots=u_p=0$ and $u_{n+1}=\cdots=u_{n+p+1}=1$, and it
    is called a *clamped knot vector*. Basis functions of clamped knot
    vectors satisfy two additional properties:
    1. $N_{0,p}(u=0)=1$ and $N_{i,p}(u=0)=0$ for $i\neq0$.
    2. $N_{n,p}(u=1)=1$ and $N_{i,p}(u=1)=0$ for $i\neq n$.

    In the remainder of this note, all knot vectors are understood to
    be clamped.

11. If the knot vector is clamped and $p=n$, the B-spline basis
    functions reduce to Bernstein polynomials, that is,
    $N_{i,p}(u)=B_{i,n}(u)$. This holds because the recursive
    definition of the B-spline basis functions reduces to the
    recursive definition of the Bernstein polynomials when the knot
    vector is

    ```{math}
    U = [\underbrace{0,\ldots,0}_{p+1},\ \underbrace{1,\ldots,1}_{p+1}].
    ```

### Mathematical properties of B-spline curves

Here is a list of some important properties of B-spline curves:

1. $\mathbf C(u)$ is a piecewise curve, and its components are
   polynomials of, at most, degree $p$ on each span.

2. The degree $p$, number of control points $n+1$, and number of knots
   $r+1$ are related by $r=n+p+1$.

3. If $n=p$ and the knot vector is clamped, then $\mathbf C(u)$ is a
   Bézier curve.

4. Affine invariance: B-spline curves are invariant under affine
   transformations such as rotations, displacements, and scalings. One
   can apply an affine transformation to the curve by applying it to
   its control points.

5. Convex hull property: all the points of a B-spline curve are
   contained in the convex hull of its control points,

   ```{math}
   \mathcal{CH}(P) = \Big\{\, \mathbf C = \sum_{k=0}^{N} a_k\, \mathbf P_k
   \ \text{such that}\ \sum_{k=0}^{N} a_k = 1 \text{ and } a_k\geq0
   \text{ for } k=0,1,\ldots,N \,\Big\},
   ```

   which follows from the non-negativity and partition-of-unity
   properties of the basis functions.

6. Strong convex hull property: if $u\in[u_{i_0},u_{i_0+1})$, then
   $\mathbf C(u)$ is contained in the convex hull of the control
   points $\mathbf P_i$ with $i_0-p\leq i\leq i_0$.

7. Local modification scheme: modifying the control point $\mathbf P_i$
   affects $\mathbf C(u)$ only on the interval $[u_i,u_{i+p+1})$. This
   follows from $N_{i,p}(u)=0$ outside that interval, and it implies
   that the shape of a B-spline curve can be modified locally without
   changing its shape globally.

8. The polygon formed by the set of control points is known as the
   *control polygon*. The control polygon represents a piecewise
   linear approximation to the B-spline curve.

9. Variation diminishing property: no straight line (or plane in three
   dimensions) intersects the B-spline curve more times than it
   intersects its control polygon. Intuitively, the curve does not
   wiggle more than its control polygon.

10. Continuity and differentiability:
    1. B-spline curves are infinitely differentiable in the interior
       of the knot intervals.
    2. B-spline curves are *at least* $p-k$ continuously differentiable
       at a knot with multiplicity $k$, for $1\leq k\leq p$.

11. First and higher order derivatives:

    The first and higher order derivatives of a B-spline curve are
    given directly by differentiating the basis functions,

    ```{math}
    \frac{\mathrm d\mathbf C}{\mathrm du} = \sum_{i=0}^{n} N'_{i,p}(u)\, \mathbf P_i,
    \qquad
    \frac{\mathrm d^{k}\mathbf C}{\mathrm du^{k}} = \sum_{i=0}^{n} N_{i,p}^{(k)}(u)\, \mathbf P_i.
    ```

    Alternatively, the derivative curve can itself be represented as a
    B-spline curve of one lower degree, with its own, shorter, knot
    vector and its own control points. For $0\leq k\leq p$, define
    derivative control points by

    ```{math}
    :label: orig-bspline-derivative-points
    \mathbf P_i^{(0)} = \mathbf P_i, \qquad
    \mathbf P_i^{(k)} = \frac{p-k+1}{u_{i+p+1}-u_{i+k}}
    \big(\mathbf P_{i+1}^{(k-1)} - \mathbf P_i^{(k-1)}\big)
    \quad \text{for } k\geq1,
    ```

    and a shortened knot vector by removing $k$ entries from **each**
    end of the original vector,

    ```{math}
    U^{(k)} = [u_k,\ldots,u_{n+p+1-k}].
    ```

    Then

    ```{math}
    :label: bspline-kth-derivative
    \frac{\mathrm d^{k}\mathbf C}{\mathrm du^{k}}
    = \sum_{i=0}^{n-k} N_{i,p-k}(u;\,U^{(k)})\, \mathbf P_i^{(k)}.
    ```

    The basis functions in {eq}`bspline-kth-derivative` are evaluated
    on the shortened knot vector $U^{(k)}$, **not** on the original
    knot vector $U$: differentiating the original basis directly, as
    in {eq}`orig-bspline-basis-derivative`, and evaluating a lower-degree
    basis on a shortened knot vector, as in
    {eq}`orig-bspline-derivative-points`–{eq}`bspline-kth-derivative`, are
    two different, equally valid representations of the same
    derivative curve. For a clamped, normalized knot vector, $U^{(k)}$
    is equivalent to repeating the first and last knots $p-k+1$ times
    each, with the interior knots unchanged. See also
    [Shene's derivation](https://pages.mtu.edu/~shene/COURSES/cs3621/NOTES/surface/bspline-derv.html)
    for a worked example of this construction.

12. Endpoint interpolation: the start and end points of a clamped
    B-spline curve coincide with the first and last control points,
    respectively:

    ```{math}
    \mathbf C(u=0) = \mathbf P_0, \qquad \mathbf C(u=1) = \mathbf P_n.
    ```

13. Endpoint tangency: for $p\geq1$, a clamped B-spline curve is
    tangent to the control polygon at the endpoints,

    ```{math}
    \frac{\mathrm d\mathbf C}{\mathrm du}\Big|_{u=0}
    = \left(\frac{p}{u_{p+1}}\right)(\mathbf P_1-\mathbf P_0),
    \qquad
    \frac{\mathrm d\mathbf C}{\mathrm du}\Big|_{u=1}
    = \left(\frac{p}{1-u_n}\right)(\mathbf P_n-\mathbf P_{n-1}).
    ```

14. Endpoint curvature: for $p\geq2$, the second derivative of a
    clamped B-spline curve at its endpoints is given by

    ```{math}
    \frac{\mathrm d^2\mathbf C}{\mathrm du^2}\Big|_{u=0}
    = \frac{p(p-1)}{u_{p+1}}\left[
    \left(\frac{1}{u_{p+2}}\right)(\mathbf P_2-\mathbf P_0)
    - \left(\frac{1}{u_{p+1}}+\frac{1}{u_{p+2}}\right)(\mathbf P_1-\mathbf P_0)
    \right],
    ```

    ```{math}
    \frac{\mathrm d^2\mathbf C}{\mathrm du^2}\Big|_{u=1}
    = \frac{p(p-1)}{1-u_n}\left[
    \left(\frac{1}{1-u_{n-1}}\right)(\mathbf P_{n-2}-\mathbf P_n)
    - \left(\frac{1}{1-u_n}+\frac{1}{1-u_{n-1}}\right)(\mathbf P_{n-1}-\mathbf P_n)
    \right].
    ```

    Provided the endpoint control-polygon edge is nonzero, the
    curvature of a clamped B-spline curve at its endpoints is given by

    ```{math}
    :label: orig-bspline-endpoint-curvature
    \kappa(u=0) = \left(\frac{p-1}{p}\right)\left(\frac{u_{p+1}}{u_{p+2}}\right)
    \frac{\big\|(\mathbf P_2-\mathbf P_0)\times(\mathbf P_1-\mathbf P_0)\big\|}
    {\|\mathbf P_1-\mathbf P_0\|^3},
    ```

    ```{math}
    \kappa(u=1) = \left(\frac{p-1}{p}\right)\left(\frac{1-u_n}{1-u_{n-1}}\right)
    \frac{\big\|(\mathbf P_{n-2}-\mathbf P_n)\times(\mathbf P_{n-1}-\mathbf P_n)\big\|}
    {\|\mathbf P_{n-1}-\mathbf P_n\|^3}.
    ```

    These formulas require $p\geq2$ and a nonzero endpoint tangent.
    They reduce to the Bézier endpoint curvature formulas
    {eq}`orig-bezier-endpoint-curvature` when $p=n$ and the knot vector is
    clamped, since then $u_{p+1}/u_{p+2}=(1-u_n)/(1-u_{n-1})=1$. A
    nonuniform B-spline that is not a single Bézier segment can still
    have these knot ratios equal to one: for example, with $p=2$ and
    $U=[0,0,0,0.4,0.4,1,1,1]$, both ratios equal one even though
    $n=4>p$. See [G² continuity](g2_continuity.md) for how these
    formulas are used to join curve segments smoothly.

## B-spline surfaces

### Definition

A B-spline surface, shorthand for basis spline surface, is a
parametric surface defined by

```{math}
:label: def-bspline-surface
\mathbf S(u,v) = \sum_{i=0}^{n}\sum_{j=0}^{m}
N_{i,p}(u)\, N_{j,q}(v)\, \mathbf P_{i,j},
\qquad 0\leq(u,v)\leq1,
```

where $p$ and $q$ are the degrees of the surface in the $u$- and
$v$-directions, the coefficients $\mathbf P_{i,j}$ are a bidirectional
net of control points, and $N_{i,p}(u)N_{j,q}(v)$ is the product of
univariate B-spline basis functions defined on the nondecreasing knot
vectors $U$ and $V$:

```{math}
U = [u_0,\ldots,u_r]\in\mathbb R^{r+1} \ \text{with}\ r=n+p+1,
\qquad
V = [v_0,\ldots,v_s]\in\mathbb R^{s+1} \ \text{with}\ s=m+q+1.
```

The $u$-direction basis functions are given by
{eq}`orig-bspline-degree-zero`–{eq}`orig-cox-de-boor`, while the $v$-direction
basis functions are defined analogously, replacing the variable $u$ by
$v$ and the indices $i$ and $p$ by $j$ and $q$, respectively.

### Mathematical properties of tensor-product B-spline basis functions

Here is a list of some important properties of the tensor-product
B-spline basis functions:

1. The relation $r=n+p+1$ holds, where $n+1$ is the number of basis
   functions in the $u$-direction and $r+1$ is the number of elements
   of the knot vector $U$. Likewise, $s=m+q+1$ relates $m+1$ and the
   $s+1$ elements of $V$.

2. $N_{i,p}(u)$ is, at most, a polynomial of degree $p$; $N_{j,q}(v)$
   is, at most, a polynomial of degree $q$.

3. Local support:
   1. $N_{i,p}(u)N_{j,q}(v)=0$ if $(u,v)$ is outside the rectangle
      $[u_i,u_{i+p+1})\times[v_j,v_{j+q+1})$.
   2. In any given knot rectangle
      $[u_{i_0},u_{i_0+1})\times[v_{j_0},v_{j_0+1})$, at most
      $(p+1)(q+1)$ basis functions are nonzero, namely
      $N_{i,p}(u)N_{j,q}(v)$ with $i_0-p\leq i\leq i_0$ and
      $j_0-q\leq j\leq j_0$.

4. Non-negativity: $N_{i,p}(u)N_{j,q}(v)\geq0$ for all $i$, $j$, $p$,
   $q$, and $(u,v)\in[0,1]\times[0,1]$.

5. Partition of unity:

   ```{math}
   \sum_{i=0}^{n}\sum_{j=0}^{m} N_{i,p}(u)\,N_{j,q}(v) = 1
   \qquad \text{for all } (u,v)\in[0,1]\times[0,1].
   ```

   In addition, for an arbitrary knot rectangle
   $[u_{i_0},u_{i_0+1})\times[v_{j_0},v_{j_0+1})$,

   ```{math}
   \sum_{i=i_0-p}^{i_0}\sum_{j=j_0-q}^{j_0} N_{i,p}(u)\,N_{j,q}(v) = 1
   \qquad \text{for all } (u,v)\in[u_{i_0},u_{i_0+1})\times[v_{j_0},v_{j_0+1}).
   ```

   This means that the sum of the nonzero basis functions of any knot
   rectangle is unity.

6. Continuity and differentiability:
   1. The basis functions are infinitely differentiable in the
      interior of the knot rectangles formed by $U$ and $V$.
   2. The basis functions are $p-k$ (respectively $q-k$) continuously
      differentiable in the $u$-direction (respectively $v$-direction)
      at a $u$-knot (respectively $v$-knot) with multiplicity $k$.

7. Extrema: for $p,q\geq1$, $N_{i,p}(u)N_{j,q}(v)$ attains exactly one
   maximum in $(u,v)\in[0,1]\times[0,1]$.

8. The first partial derivatives of the tensor-product basis functions
   are given by

   ```{math}
   \frac{\partial}{\partial u}\big(N_{i,p}(u)N_{j,q}(v)\big)
   = N_{j,q}(v)\,\frac{\partial}{\partial u}\big(N_{i,p}(u)\big),
   \qquad
   \frac{\partial}{\partial v}\big(N_{i,p}(u)N_{j,q}(v)\big)
   = N_{i,p}(u)\,\frac{\partial}{\partial v}\big(N_{j,q}(v)\big).
   ```

9. The $(k,l)$-th order derivatives of the tensor-product basis
   functions are given by

   ```{math}
   \frac{\partial^{k+l}}{\partial u^{k}\partial v^{l}}
   \big(N_{i,p}(u)N_{j,q}(v)\big)
   = \frac{\partial^{k}}{\partial u^{k}}\big(N_{i,p}(u)\big)\,
   \frac{\partial^{l}}{\partial v^{l}}\big(N_{j,q}(v)\big).
   ```

10. If the knot vectors are clamped and $(p,q)=(n,m)$, then the
    product of B-spline basis functions reduces to the product of
    Bernstein polynomials, that is,
    $N_{i,p}(u)\,N_{j,q}(v)=B_{i,n}(u)\,B_{j,m}(v)$.

### Mathematical properties of B-spline surfaces

Here is a list of some important properties of B-spline surfaces:

1. $\mathbf S(u,v)$ is a piecewise surface, and its components are
   bivariate polynomials of, at most, bidegree $(p,q)$: degree $p$ in
   $u$ and degree $q$ in $v$.

2. In the $u$-direction, the degree $p$, number of control points
   $n+1$, and number of knots $r+1$ are related by $r=n+p+1$. In the
   $v$-direction, the degree $q$, number of control points $m+1$, and
   number of knots $s+1$ are related by $s=m+q+1$.

3. If $n=p$, $m=q$, and both knot vectors are clamped, then
   $\mathbf S(u,v)$ is a Bézier surface.

4. Affine invariance: B-spline surfaces are invariant under affine
   transformations such as rotations, displacements, and scalings. One
   can apply an affine transformation to the surface by applying it to
   its control net.

5. Convex hull property: all the points of a B-spline surface are
   contained in the convex hull of its control points,

   ```{math}
   \mathcal{CH}(P) = \Big\{\, \mathbf S = \sum_{k=0}^{N} a_k\, \mathbf P_k
   \ \text{such that}\ \sum_{k=0}^{N} a_k = 1 \text{ and } a_k\geq0
   \text{ for } k=0,1,\ldots,N \,\Big\}.
   ```

6. Strong convex hull property: if
   $(u,v)\in[u_{i_0},u_{i_0+1})\times[v_{j_0},v_{j_0+1})$, then
   $\mathbf S(u,v)$ is contained in the convex hull of the control
   points $\mathbf P_{i,j}$ with $i_0-p\leq i\leq i_0$ and
   $j_0-q\leq j\leq j_0$.

7. Local modification scheme: modifying the control point
   $\mathbf P_{i,j}$ affects $\mathbf S(u,v)$ only on the rectangle
   $[u_i,u_{i+p+1})\times[v_j,v_{j+q+1})$. This follows from
   $N_{i,p}(u)N_{j,q}(v)=0$ outside that rectangle, and it implies that
   the shape of a B-spline surface can be modified locally without
   changing its shape globally.

8. If triangulated, the net of control points represents a piecewise
   planar approximation to the B-spline surface.

9. No known variation-diminishing property.

10. Continuity and differentiability:
    1. B-spline surfaces are infinitely differentiable in the interior
       of the knot rectangles formed by $U$ and $V$.
    2. B-spline surfaces are $p-k$ (respectively $q-k$) continuously
       differentiable in the $u$-direction (respectively
       $v$-direction) at a $u$-knot (respectively $v$-knot) with
       multiplicity $k$.

11. First and higher order derivatives:

    ```{math}
    :label: orig-bspline-surface-derivatives
    \frac{\partial\mathbf S}{\partial u}
    = \sum_{i=0}^{n}\sum_{j=0}^{m} N'_{i,p}(u)\,N_{j,q}(v)\, \mathbf P_{i,j},
    \qquad
    \frac{\partial\mathbf S}{\partial v}
    = \sum_{i=0}^{n}\sum_{j=0}^{m} N_{i,p}(u)\,N'_{j,q}(v)\, \mathbf P_{i,j},
    ```

    ```{math}
    \frac{\partial^{k+l}\mathbf S}{\partial u^{k}\partial v^{l}}
    = \sum_{i=0}^{n}\sum_{j=0}^{m} N_{i,p}^{(k)}(u)\,N_{j,q}^{(l)}(v)\, \mathbf P_{i,j}.
    ```

12. Corner point interpolation: the corners of a clamped B-spline
    surface coincide with the corner points of its control net,

    ```{math}
    \mathbf S(u=0,v=0) = \mathbf P_{0,0}, \qquad
    \mathbf S(u=1,v=0) = \mathbf P_{n,0}, \qquad
    \mathbf S(u=0,v=1) = \mathbf P_{0,m}, \qquad
    \mathbf S(u=1,v=1) = \mathbf P_{n,m}.
    ```

    More generally, when both knot vectors are clamped, each boundary
    row or column of the control net defines a B-spline curve that
    traces out the corresponding edge of the surface.
