# NURBS curves and surfaces

NURBS curves and surfaces extend B-splines with per-control-point
weights, making the basis functions rational rather than polynomial and
allowing exact representation of conics such as circular arcs. This page
defines the rational basis functions and the curve or surface built from
them, then states their main properties — including local support,
partition of unity, and the convex-hull property — first for curves and
then for surfaces. Most of the material follows {cite:t}`NURBS_book`,
Chapter 4.

## NURBS curves

### Definition

A Non-Uniform Rational Basis Spline (NURBS) curve is a parametric curve
defined by

```{math}
:label: def-nurbs-curve
\mathbf C(u) = \sum_{i=0}^{n} R_{i,p}(u)\, \mathbf P_i,
\qquad 0\leq u\leq1,
```

where $p$ is the **degree** of the curve, the coefficients
$\mathbf P_i$ are called control points, and $R_{i,p}$ are the
rational basis functions given by

```{math}
:label: def-rational-basis
R_{i,p}(u) = \frac{N_{i,p}(u)\, w_i}
{\sum\limits_{k=0}^{n} N_{k,p}(u)\, w_k},
```

in which $w_i$ are the weights of the control points and $N_{i,p}$
are B-spline basis functions defined on the nondecreasing knot vector
$U$:

```{math}
U = [u_0,\ldots,u_r]\in\mathbb R^{r+1}
\qquad \text{with} \qquad r=n+p+1.
```

The B-spline basis functions are given by the recursive relation

```{math}
N_{i,0}(u) = \begin{cases}
1 & \text{if } u_i\leq u<u_{i+1},\\
0 & \text{otherwise},
\end{cases}
\qquad
N_{i,p}(u) = \frac{u-u_i}{u_{i+p}-u_i}\,N_{i,p-1}(u)
+ \frac{u_{i+p+1}-u}{u_{i+p+1}-u_{i+1}}\,N_{i+1,p-1}(u).
```

This recursive relation can produce a $0/0$ quotient, which is defined
to be zero by convention.

### Mathematical properties of rational basis functions

Here is a list of some important properties of the rational basis
functions:

1. The relation $r=n+p+1$ holds, where $n+1$ is the number of basis
   functions and $r+1$ is the number of elements of the knot vector
   $U$.

2. The numerator and denominator of $R_{i,p}(u)$ are, at most,
   polynomials of degree $p$.

3. Local support:
   1. $R_{i,p}(u)=0$ if $u$ is outside the interval
      $[u_i,u_{i+p+1})$.
   2. In any given knot interval $[u_{i_0},u_{i_0+1})$, at most $p+1$
      basis functions are nonzero, namely $R_{i,p}(u)$ with
      $i_0-p\leq i\leq i_0$.

4. Non-negativity: if all weights are positive, $R_{i,p}(u)\geq0$ for
   all $i$, $p$, and $u\in[0,1]$. A mixed-sign choice of weights can
   make the denominator vanish or change sign, and is excluded here.

5. Partition of unity:

   ```{math}
   \sum_{i=0}^{n} R_{i,p}(u) = 1 \qquad \text{for all } u\in[0,1].
   ```

   In addition, for an arbitrary knot span $[u_{i_0},u_{i_0+1})$,

   ```{math}
   \sum_{i=i_0-p}^{i_0} R_{i,p}(u) = 1
   \qquad \text{for all } u\in[u_{i_0},u_{i_0+1}).
   ```

   This means that the sum of the nonzero basis functions of any knot
   span is unity.

6. Continuity and differentiability:
   1. The basis functions are infinitely differentiable in the
      interior of the knot intervals.
   2. The basis functions are $p-k$ continuously differentiable at a
      knot with multiplicity $k$, for $1\leq k\leq p$, provided the
      weighted denominator does not vanish there.

7. Extrema: for $p\geq1$ and positive weights, $R_{i,p}(u)$ attains
   exactly one maximum in $u\in[0,1]$.

8. Unlike the polynomial basis, there is no compact closed-form
   expression for the derivative of $R_{i,p}(u)$ purely in terms of
   other rational basis functions and knots; it also involves the
   weighted denominator $w(u)=\sum_kN_{k,p}(u)w_k$ and its derivatives.
   A usable recursive formula can still be obtained by differentiating
   $w(u)R_{i,p}(u)=w_iN_{i,p}(u)$ with the product rule $k$ times and
   solving for the highest-order term:

   ```{math}
   :label: rational-basis-derivative
   R_{i,p}^{(k)}(u) = \frac{1}{w(u)}\left[
   w_i N_{i,p}^{(k)}(u) - \sum_{j=1}^{k}\binom{k}{j} w^{(j)}(u)\, R_{i,p}^{(k-j)}(u)
   \right],
   ```

   where $w^{(j)}=\sum_kN_{k,p}^{(j)}(u)w_k$. This is the same
   Leibniz-rule technique used below for the coordinates of a NURBS
   curve, applied here to a single basis function instead of to
   $\mathbf C(u)$.

9. When the first and last knots have multiplicity $p+1$, the knot
   vector is a *clamped knot vector* as in the B-spline case, and
   basis functions of clamped knot vectors satisfy two additional
   properties:
   1. $R_{0,p}(u=0)=1$ and $R_{i,p}(u=0)=0$ for $i\neq0$.
   2. $R_{n,p}(u=1)=1$ and $R_{i,p}(u=1)=0$ for $i\neq n$.

10. If all the control-point weights are equal and nonzero, the
    rational basis functions reduce to B-spline basis functions, that
    is, $R_{i,p}(u)=N_{i,p}(u)$.

11. If all the control-point weights are equal and nonzero, the knot
    vector is clamped, and $p=n$, then the rational basis functions
    reduce to Bernstein polynomials, that is, $R_{i,p}(u)=B_{i,n}(u)$.

### Mathematical properties of NURBS curves

Here is a list of some important properties of NURBS curves:

1. $\mathbf C(u)$ is a piecewise curve, and its components are ratios
   of polynomials of, at most, degree $p$.

2. The degree $p$, number of control points $n+1$, and number of knots
   $r+1$ are related by $r=n+p+1$.

3. NURBS curves contain B-spline and rational/non-rational Bézier
   curves as special cases:
   1. If all the control-point weights are equal, the NURBS curve
      reduces to a B-spline curve.
   2. If $n=p$ and the knot vector is clamped, the NURBS curve reduces
      to a rational Bézier curve.
   3. If all the control-point weights are equal, $p=n$, and the knot
      vector is clamped, the NURBS curve reduces to a polynomial
      Bézier curve.

4. Affine invariance: NURBS curves are invariant under affine
   transformations such as rotations, displacements, and scalings. One
   can apply an affine transformation to the curve by applying it to
   its control points. Given an affine transformation
   $\phi(\mathbf v)=A\mathbf v+\mathbf b$,

   ```{math}
   \phi\big(\mathbf C(u)\big)
   = \phi\Big(\sum_{i=0}^{n} R_{i,p}(u)\,\mathbf P_i\Big)
   = A\sum_{i=0}^{n} R_{i,p}(u)\,\mathbf P_i + \mathbf b
   = A\sum_{i=0}^{n} R_{i,p}(u)\,\mathbf P_i
   + \mathbf b\sum_{i=0}^{n} R_{i,p}(u)
   ```

   ```{math}
   = \sum_{i=0}^{n} R_{i,p}(u)\,(A\mathbf P_i+\mathbf b)
   = \sum_{i=0}^{n} R_{i,p}(u)\,\phi(\mathbf P_i),
   ```

   using the partition of unity in the middle step.

5. Convex hull property: with positive weights, all the points of a
   NURBS curve are contained in the convex hull of its control points,

   ```{math}
   \mathcal{CH}(P) = \Big\{\, \mathbf C = \sum_{k=0}^{N} a_k\, \mathbf P_k
   \ \text{such that}\ \sum_{k=0}^{N} a_k = 1 \text{ and } a_k\geq0
   \text{ for } k=0,1,\ldots,N \,\Big\}.
   ```

   With mixed-sign weights this can fail: a two-point rational curve
   can evaluate outside the segment joining its control points.

6. Strong convex hull property: if $u\in[u_{i_0},u_{i_0+1})$, then
   $\mathbf C(u)$ is contained in the convex hull of the control
   points $\mathbf P_i$ with $i_0-p\leq i\leq i_0$.

7. Local modification scheme: modifying the control point
   $\mathbf P_i$ or weight $w_i$ affects $\mathbf C(u)$ only on the
   interval $[u_i,u_{i+p+1})$. This follows from $R_{i,p}(u)=0$
   outside that interval, and it implies that the shape of a NURBS
   curve can be modified locally without changing its shape globally.

8. The polygon formed by the set of control points is known as the
   *control polygon*. The control polygon represents a piecewise
   linear approximation to the NURBS curve.

9. Variation diminishing property: with positive weights, no straight
   line (or plane in three dimensions) intersects the NURBS curve more
   times than it intersects its control polygon. Intuitively, the
   curve does not wiggle more than its control polygon.

10. Continuity and differentiability:
    1. NURBS curves are infinitely differentiable in the interior of
       the knot intervals, wherever the weighted denominator does not
       vanish.
    2. NURBS curves are *at least* $p-k$ continuously differentiable
       at a knot with multiplicity $k$.

11. Homogeneous coordinates: $N$-dimensional rational functions with a
    common denominator can be represented as polynomial functions in
    $(N+1)$-dimensional space using *homogeneous coordinates*.

    Consider a three-dimensional point $\mathbf P=(x,y,z)$. It can be
    written in four-dimensional space as
    $\mathbf P^w=(wx,wy,wz,w)=(X,Y,Z,W)$, where $w\neq0$. The point
    $\mathbf P$ is recovered from $\mathbf P^w$ by the mapping
    $\mathcal H$,

    ```{math}
    \mathbf P = \mathcal H\{\mathbf P^w\}
    = \mathcal H\{(X,Y,Z,W)\}
    = \left(\frac{X}{W}, \frac{Y}{W}, \frac{Z}{W}\right).
    ```

    This lets an $N$-dimensional NURBS curve be represented as an
    $(N+1)$-dimensional B-spline curve, whose coordinates are then
    mapped back with $\mathcal H$. Given a NURBS curve $\mathbf C(u)$
    with control points $\mathbf P_i$ and weights $w_i$, construct the
    weighted control points

    ```{math}
    :label: nurbs-curve-weighted-points
    \mathbf P_i^w = (w_ix_i,\, w_iy_i,\, w_iz_i,\, w_i)
    ```

    and define the corresponding non-rational B-spline curve in
    four-dimensional space,

    ```{math}
    \mathbf C^w(u) = \sum_{i=0}^{n} N_{i,p}(u)\, \mathbf P_i^w,
    \qquad 0\leq u\leq1.
    ```

    The coordinates of the original NURBS curve are recovered using
    standard B-spline algorithms to evaluate $\mathbf C^w(u)$ and then
    applying $\mathcal H$:

    ```{math}
    \mathbf C(u) = \mathcal H\{\mathbf C^w(u)\}
    = \mathcal H\Big\{\sum_{i=0}^{n} N_{i,p}(u)\,\mathbf P_i^w\Big\}
    = \frac{\sum\limits_{i=0}^{n} N_{i,p}(u)\,w_i\,\mathbf P_i}
    {\sum\limits_{i=0}^{n} N_{i,p}(u)\,w_i}
    = \sum_{i=0}^{n} R_{i,p}(u)\,\mathbf P_i.
    ```

    This property is the key to evaluating the derivatives of NURBS
    curves.

12. First and higher order derivatives:

    The derivatives of a NURBS curve $\mathbf C(u)$ can be expressed in
    terms of the derivatives of the corresponding B-spline curve in
    homogeneous space, $\mathbf C^w(u)$. Write

    ```{math}
    \mathbf C(u) = \frac{\sum\limits_{i=0}^{n} N_{i,p}(u)\,w_i\,\mathbf P_i}
    {\sum\limits_{i=0}^{n} N_{i,p}(u)\,w_i}
    = \frac{\mathbf A(u)}{w(u)},
    ```

    where $\mathbf A(u)$ collects the first three coordinates of
    $\mathbf C^w(u)$ and $w(u)$ is its fourth coordinate. To compute
    the first derivative, clear the denominator, differentiate, and
    solve for $\mathbf C'(u)$:

    ```{math}
    w\,\mathbf C = \mathbf A
    \ \Longrightarrow\
    w'\,\mathbf C + w\,\mathbf C' = \mathbf A'
    \ \Longrightarrow\
    \mathbf C' = \frac{1}{w}\big(\mathbf A' - w'\,\mathbf C\big).
    ```

    To compute the $k$-th order derivative, differentiate $k$ times
    using Leibniz' rule for products, then solve for
    $\mathbf C^{(k)}(u)$:

    ```{math}
    :label: nurbs-curve-kth-derivative
    \big[w\,\mathbf C\big]^{(k)} = \mathbf A^{(k)}
    \ \Longrightarrow\
    \sum_{i=0}^{k}\binom{k}{i} w^{(i)}\,\mathbf C^{(k-i)} = \mathbf A^{(k)}
    \ \Longrightarrow\
    \mathbf C^{(k)} = \frac{1}{w}\left(
    \mathbf A^{(k)} - \sum_{i=1}^{k}\binom{k}{i} w^{(i)}\,\mathbf C^{(k-i)}
    \right).
    ```

    The derivatives of $\mathbf A(u)$ and $w(u)$ are obtained directly
    by differentiating $\mathbf C^w(u)$, which is a polynomial
    B-spline curve:

    ```{math}
    \big[\mathbf C^w\big]^{(k)}(u) = \sum_{i=0}^{n} N_{i,p}^{(k)}(u)\, \mathbf P_i^w.
    ```

    A related, useful fact is how the curve responds to a change in a
    single weight: differentiating {eq}`def-nurbs-curve` with respect
    to $w_i$ using {eq}`def-rational-basis` gives

    ```{math}
    \frac{\partial \mathbf C}{\partial w_i}
    = \frac{N_{i,p}(u)}{w(u)}\big(\mathbf P_i - \mathbf C(u)\big),
    ```

    so increasing $w_i$ pulls the curve toward $\mathbf P_i$ on the
    support of $N_{i,p}$, and rescaling every weight by the same
    nonzero constant leaves the curve unchanged.

13. Endpoint interpolation: the start and end points of a clamped
    NURBS curve coincide with the first and last control points,
    respectively:

    ```{math}
    \mathbf C(u=0) = \mathbf P_0, \qquad \mathbf C(u=1) = \mathbf P_n.
    ```

14. Endpoint tangency: for $p\geq1$, a clamped NURBS curve is tangent
    to the control polygon at the endpoints,

    ```{math}
    \frac{\mathrm d\mathbf C}{\mathrm du}\Big|_{u=0}
    = \left(\frac{p}{u_{p+1}}\right)\left(\frac{w_1}{w_0}\right)
    (\mathbf P_1-\mathbf P_0),
    \qquad
    \frac{\mathrm d\mathbf C}{\mathrm du}\Big|_{u=1}
    = \left(\frac{p}{1-u_n}\right)\left(\frac{w_{n-1}}{w_n}\right)
    (\mathbf P_n-\mathbf P_{n-1}).
    ```

15. Endpoint curvature: for $p\geq2$, the second derivative of a
    clamped NURBS curve at its endpoints is given by

    ```{math}
    \frac{\mathrm d^2\mathbf C}{\mathrm du^2}\Big|_{u=0} =
    \frac{p(p-1)}{u_{p+1}}\left[
    \left(\frac{1}{u_{p+2}}\right)\left(\frac{w_2}{w_0}\right)(\mathbf P_2-\mathbf P_0)
    - \left(\frac{1}{u_{p+1}}+\frac{1}{u_{p+2}}\right)\left(\frac{w_1}{w_0}\right)(\mathbf P_1-\mathbf P_0)
    \right]
    ```

    ```{math}
    {}+ \frac{2p^2}{u_{p+1}^2}\left(\frac{w_1}{w_0}\right)\left(1-\frac{w_1}{w_0}\right)(\mathbf P_1-\mathbf P_0),
    ```

    ```{math}
    \frac{\mathrm d^2\mathbf C}{\mathrm du^2}\Big|_{u=1} =
    \frac{p(p-1)}{1-u_n}\left[
    \left(\frac{1}{1-u_{n-1}}\right)\left(\frac{w_{n-2}}{w_n}\right)(\mathbf P_{n-2}-\mathbf P_n)
    - \left(\frac{1}{1-u_n}+\frac{1}{1-u_{n-1}}\right)\left(\frac{w_{n-1}}{w_n}\right)(\mathbf P_{n-1}-\mathbf P_n)
    \right]
    ```

    ```{math}
    {}+ \frac{2p^2}{(1-u_n)^2}\left(\frac{w_{n-1}}{w_n}\right)\left(1-\frac{w_{n-1}}{w_n}\right)(\mathbf P_{n-1}-\mathbf P_n).
    ```

    Provided the endpoint control-polygon edge is nonzero, the
    curvature of a clamped NURBS curve at its endpoints is given by

    ```{math}
    :label: orig-nurbs-endpoint-curvature
    \kappa(u=0) = \left(\frac{p-1}{p}\right)\left(\frac{u_{p+1}}{u_{p+2}}\right)
    \left(\frac{w_0w_2}{w_1^2}\right)
    \frac{\big\|(\mathbf P_2-\mathbf P_0)\times(\mathbf P_1-\mathbf P_0)\big\|}
    {\|\mathbf P_1-\mathbf P_0\|^3},
    ```

    ```{math}
    \kappa(u=1) = \left(\frac{p-1}{p}\right)\left(\frac{1-u_n}{1-u_{n-1}}\right)
    \left(\frac{w_nw_{n-2}}{w_{n-1}^2}\right)
    \frac{\big\|(\mathbf P_{n-2}-\mathbf P_n)\times(\mathbf P_{n-1}-\mathbf P_n)\big\|}
    {\|\mathbf P_{n-1}-\mathbf P_n\|^3}.
    ```

    These formulas require $p\geq2$ and a nonzero endpoint tangent.
    They reduce to the B-spline endpoint curvature formulas
    {eq}`orig-bspline-endpoint-curvature` when the endpoint weight ratios
    are one, and further to the Bézier formulas
    {eq}`orig-bezier-endpoint-curvature` under the additional B-spline
    special cases above. See [G² continuity](g2_continuity.md) for how
    these formulas are used to join curve segments smoothly.

## NURBS surfaces

### Definition

A Non-Uniform Rational Basis Spline (NURBS) surface is a parametric
surface defined by

```{math}
:label: def-nurbs-surface
\mathbf S(u,v) = \sum_{i=0}^{n}\sum_{j=0}^{m}
R_{i,j}^{p,q}(u,v)\, \mathbf P_{i,j},
\qquad 0\leq(u,v)\leq1,
```

where $p$ and $q$ are the degrees of the surface in the $u$- and
$v$-directions, the coefficients $\mathbf P_{i,j}$ are a bidirectional
net of control points, and $R_{i,j}^{p,q}(u,v)$ are the rational basis
functions given by

```{math}
:label: def-rational-surface-basis
R_{i,j}^{p,q}(u,v) = \frac{N_{i,p}(u)\,N_{j,q}(v)\, w_{i,j}}
{\sum\limits_{a=0}^{n}\sum\limits_{b=0}^{m}
N_{a,p}(u)\,N_{b,q}(v)\, w_{a,b}},
```

in which $w_{i,j}$ are the weights of the control points and
$N_{i,p}(u)N_{j,q}(v)$ is the product of univariate B-spline basis
functions defined on the nondecreasing knot vectors $U$ and $V$:

```{math}
U = [u_0,\ldots,u_r]\in\mathbb R^{r+1} \ \text{with}\ r=n+p+1,
\qquad
V = [v_0,\ldots,v_s]\in\mathbb R^{s+1} \ \text{with}\ s=m+q+1.
```

The denominator sums over **both** the $u$-degree basis $N_{a,p}(u)$
and the $v$-degree basis $N_{b,q}(v)$; the $v$-direction factor must
use the degree $q$, not $p$, since $p$ and $q$ can differ. The
$u$-direction basis functions are given by the recursive relation of
the B-spline case, while the $v$-direction basis functions are defined
analogously, replacing the variable $u$ by $v$ and the indices $i$ and
$p$ by $j$ and $q$, respectively.

### Mathematical properties of bivariate rational basis functions

Here is a list of some important properties of the bivariate rational
basis functions:

1. The relation $r=n+p+1$ holds, where $n+1$ is the number of basis
   functions in the $u$-direction and $r+1$ is the number of elements
   of the knot vector $U$. Likewise, $s=m+q+1$ relates $m+1$ and the
   $s+1$ elements of $V$.

2. The numerator and denominator of $R_{i,j}^{p,q}(u,v)$ are, at most,
   polynomials of bidegree $(p,q)$.

3. Local support:
   1. $R_{i,j}^{p,q}(u,v)=0$ if $(u,v)$ is outside the rectangle
      $[u_i,u_{i+p+1})\times[v_j,v_{j+q+1})$.
   2. In any given knot rectangle
      $[u_{i_0},u_{i_0+1})\times[v_{j_0},v_{j_0+1})$, at most
      $(p+1)(q+1)$ basis functions are nonzero, namely
      $R_{i,j}^{p,q}(u,v)$ with $i_0-p\leq i\leq i_0$ and
      $j_0-q\leq j\leq j_0$.

4. Non-negativity: with positive weights, $R_{i,j}^{p,q}(u,v)\geq0$ for
   all $i$, $j$, $p$, $q$, and $(u,v)\in[0,1]\times[0,1]$.

5. Partition of unity:

   ```{math}
   \sum_{i=0}^{n}\sum_{j=0}^{m} R_{i,j}^{p,q}(u,v) = 1
   \qquad \text{for all } (u,v)\in[0,1]\times[0,1].
   ```

   In addition, for an arbitrary knot rectangle
   $[u_{i_0},u_{i_0+1})\times[v_{j_0},v_{j_0+1})$,

   ```{math}
   \sum_{i=i_0-p}^{i_0}\sum_{j=j_0-q}^{j_0} R_{i,j}^{p,q}(u,v) = 1
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

7. Extrema: for $p,q\geq1$ and positive weights, $R_{i,j}^{p,q}(u,v)$
   attains exactly one maximum in $(u,v)\in[0,1]\times[0,1]$.

8. As in the curve case, there is no compact closed-form expression for
   the partial derivatives of $R_{i,j}^{p,q}(u,v)$ purely in terms of
   other rational basis functions and knots. They can, however, be
   obtained with the same Leibniz-rule technique used below for the
   coordinates of a NURBS surface, applied to a single basis function
   instead of to $\mathbf S(u,v)$.

9. If all the control-point weights are equal, the rational basis
   functions reduce to B-spline basis functions, that is,
   $R_{i,j}^{p,q}(u,v)=N_{i,p}(u)\,N_{j,q}(v)$.

10. If all the control-point weights are equal, the knot vectors are
    clamped, and $(p,q)=(n,m)$, then the rational basis functions
    reduce to Bernstein polynomials, that is,
    $R_{i,j}^{p,q}(u,v)=B_{i,n}(u)\,B_{j,m}(v)$.

### Mathematical properties of NURBS surfaces

Here is a list of some important properties of NURBS surfaces:

1. $\mathbf S(u,v)$ is a piecewise surface, and its components are
   ratios of bivariate polynomials of, at most, bidegree $(p,q)$.

2. In the $u$-direction, the degree $p$, number of control points
   $n+1$, and number of knots $r+1$ are related by $r=n+p+1$. In the
   $v$-direction, the degree $q$, number of control points $m+1$, and
   number of knots $s+1$ are related by $s=m+q+1$.

3. NURBS surfaces contain B-spline and rational/non-rational Bézier
   surfaces as special cases:
   1. If all the control-point weights are equal, the NURBS surface
      reduces to a B-spline surface.
   2. If $(p,q)=(n,m)$ and the knot vectors are clamped, the NURBS
      surface reduces to a rational Bézier surface.
   3. If all the control-point weights are equal, $(p,q)=(n,m)$, and
      the knot vectors are clamped, the NURBS surface reduces to a
      polynomial Bézier surface.

4. Affine invariance: NURBS surfaces are invariant under affine
   transformations such as rotations, displacements, and scalings.
   Given an affine transformation $\phi(\mathbf v)=A\mathbf v+\mathbf b$,

   ```{math}
   \phi\big(\mathbf S(u,v)\big)
   = \phi\Big(\sum_{i,j} R_{i,j}^{p,q}(u,v)\,\mathbf P_{i,j}\Big)
   = A\sum_{i,j} R_{i,j}^{p,q}(u,v)\,\mathbf P_{i,j}
   + \mathbf b\sum_{i,j} R_{i,j}^{p,q}(u,v)
   ```

   ```{math}
   = \sum_{i,j} R_{i,j}^{p,q}(u,v)\,(A\mathbf P_{i,j}+\mathbf b)
   = \sum_{i,j} R_{i,j}^{p,q}(u,v)\,\phi(\mathbf P_{i,j}),
   ```

   using the partition of unity in the middle step.

5. Convex hull property: with positive weights, all the points of a
   NURBS surface are contained in the convex hull of its control
   points, defined as in the curve case.

6. Strong convex hull property: if
   $(u,v)\in[u_{i_0},u_{i_0+1})\times[v_{j_0},v_{j_0+1})$, then
   $\mathbf S(u,v)$ is contained in the convex hull of the control
   points $\mathbf P_{i,j}$ with $i_0-p\leq i\leq i_0$ and
   $j_0-q\leq j\leq j_0$.

7. Local modification scheme: modifying the control point
   $\mathbf P_{i,j}$ or weight $w_{i,j}$ affects $\mathbf S(u,v)$ only
   on the rectangle $[u_i,u_{i+p+1})\times[v_j,v_{j+q+1})$. This
   follows from $R_{i,j}^{p,q}(u,v)=0$ outside that rectangle, and it
   implies that the shape of a NURBS surface can be modified locally
   without changing its shape globally.

8. If triangulated, the net of control points represents a piecewise
   planar approximation to the NURBS surface.

9. No known variation-diminishing property.

10. Continuity and differentiability:
    1. NURBS surfaces are infinitely differentiable in the interior of
       the knot rectangles formed by $U$ and $V$, wherever the
       weighted denominator does not vanish.
    2. NURBS surfaces are $p-k$ (respectively $q-k$) continuously
       differentiable in the $u$-direction (respectively
       $v$-direction) at a $u$-knot (respectively $v$-knot) with
       multiplicity $k$.

11. Homogeneous coordinates: as for curves, an $N$-dimensional
    rational surface can be represented as an $(N+1)$-dimensional
    polynomial B-spline surface using homogeneous coordinates, and the
    result mapped back with $\mathcal H$.

    Given a NURBS surface $\mathbf S(u,v)$ with control points
    $\mathbf P_{i,j}$ and weights $w_{i,j}$, construct the weighted
    control points $\mathbf P_{i,j}^w=(w_{i,j}\mathbf P_{i,j},\,w_{i,j})$,
    that is, in three spatial dimensions,

    ```{math}
    :label: nurbs-surface-weighted-points
    \mathbf P_{i,j}^w
    = (w_{i,j}x_{i,j},\, w_{i,j}y_{i,j},\, w_{i,j}z_{i,j},\, w_{i,j}),
    ```

    and define the corresponding non-rational B-spline surface in
    four-dimensional space, using the **polynomial** tensor-product
    basis, not the rational one:

    ```{math}
    :label: nurbs-surface-homogeneous
    \mathbf S^w(u,v)
    = \sum_{i=0}^{n}\sum_{j=0}^{m} N_{i,p}(u)\,N_{j,q}(v)\, \mathbf P_{i,j}^w,
    \qquad 0\leq(u,v)\leq1.
    ```

    The coordinates of the original NURBS surface are recovered using
    standard B-spline algorithms to evaluate $\mathbf S^w(u,v)$ and
    then applying $\mathcal H$:

    ```{math}
    \mathbf S(u,v) = \mathcal H\{\mathbf S^w(u,v)\}
    = \frac{\sum\limits_{i=0}^{n}\sum\limits_{j=0}^{m}
    N_{i,p}(u)\,N_{j,q}(v)\,w_{i,j}\,\mathbf P_{i,j}}
    {\sum\limits_{i=0}^{n}\sum\limits_{j=0}^{m}
    N_{i,p}(u)\,N_{j,q}(v)\,w_{i,j}}
    = \sum_{i=0}^{n}\sum_{j=0}^{m} R_{i,j}^{p,q}(u,v)\,\mathbf P_{i,j}.
    ```

    This property is the key to evaluating the derivatives of NURBS
    surfaces.

12. First and higher order derivatives:

    The derivatives of a NURBS surface $\mathbf S(u,v)$ can be
    expressed in terms of the derivatives of the corresponding
    B-spline surface in homogeneous space, $\mathbf S^w(u,v)$. Write

    ```{math}
    \mathbf S(u,v) = \frac{\sum\limits_{i,j}
    N_{i,p}(u)\,N_{j,q}(v)\,w_{i,j}\,\mathbf P_{i,j}}
    {\sum\limits_{i,j} N_{i,p}(u)\,N_{j,q}(v)\,w_{i,j}}
    = \frac{\mathbf A(u,v)}{w(u,v)},
    ```

    where $\mathbf A(u,v)$ collects the first three coordinates of
    $\mathbf S^w(u,v)$ and $w(u,v)$ is its fourth coordinate. To
    compute a first partial derivative with respect to
    $\alpha\in\{u,v\}$, clear the denominator, differentiate, and
    solve for $\mathbf S_\alpha$:

    ```{math}
    w\,\mathbf S = \mathbf A
    \ \Longrightarrow\
    w_\alpha\,\mathbf S + w\,\mathbf S_\alpha = \mathbf A_\alpha
    \ \Longrightarrow\
    \frac{\partial\mathbf S}{\partial\alpha}
    = \mathbf S_\alpha = \frac{1}{w}\big(\mathbf A_\alpha - w_\alpha\,\mathbf S\big).
    ```

    To compute the $(k,l)$-th order partial derivative, differentiate
    $k$ times in $u$ and $l$ times in $v$ using Leibniz' rule for
    products, then solve for $\mathbf S^{(k,l)}(u,v)$:

    ```{math}
    \big[w\,\mathbf S\big]^{(k,l)} = \mathbf A^{(k,l)}
    \ \Longrightarrow\
    \sum_{a=0}^{k}\sum_{b=0}^{l}
    \binom{k}{a}\binom{l}{b}\, w^{(a,b)}\,\mathbf S^{(k-a,l-b)} = \mathbf A^{(k,l)}.
    ```

    Separating the $(a,b)=(0,0)$ term from the rest and solving for
    $\mathbf S^{(k,l)}$ gives

    ```{math}
    :label: nurbs-surface-klderivative
    \mathbf S^{(k,l)} = \frac{1}{w}\left[
    \mathbf A^{(k,l)} -
    \sum_{\substack{0\leq a\leq k,\ 0\leq b\leq l\\ (a,b)\neq(0,0)}}
    \binom{k}{a}\binom{l}{b}\, w^{(a,b)}\,\mathbf S^{(k-a,l-b)}
    \right].
    ```

    Splitting the excluded index set into $a>0,\,b=0$; $a=0,\,b>0$;
    and $a>0,\,b>0$ recovers the three separate correction sums used
    for pure-$u$, pure-$v$, and mixed terms, respectively; the single
    sum above is only a more compact way to write the same
    correction. The derivatives of $\mathbf A(u,v)$ and $w(u,v)$ are
    obtained directly by differentiating $\mathbf S^w(u,v)$, which is
    a polynomial B-spline surface:

    ```{math}
    \big[\mathbf S^w\big]^{(k,l)}(u,v)
    = \sum_{i=0}^{n}\sum_{j=0}^{m} N_{i,p}^{(k)}(u)\,N_{j,q}^{(l)}(v)\, \mathbf P_{i,j}^w.
    ```

13. Corner point interpolation: the corners of a clamped NURBS surface
    coincide with the corner points of its control net,

    ```{math}
    \mathbf S(u=0,v=0) = \mathbf P_{0,0}, \qquad
    \mathbf S(u=1,v=0) = \mathbf P_{n,0}, \qquad
    \mathbf S(u=0,v=1) = \mathbf P_{0,m}, \qquad
    \mathbf S(u=1,v=1) = \mathbf P_{n,m}.
    ```

    More generally, each boundary row or column of the control net,
    together with its weights, defines a rational boundary curve of
    the surface.
