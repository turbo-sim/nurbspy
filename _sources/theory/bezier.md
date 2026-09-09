# Bézier curves and surfaces

Bézier curves and surfaces are polynomial parametrizations built from a
set of control points weighted by the Bernstein basis functions. This
page defines the Bernstein basis and the curve or surface built from it,
then states their main properties — including the convex-hull property,
endpoint interpolation, and derivatives — first for curves and then for
surfaces. Most of the material follows {cite:t}`NURBS_book`, Chapter 1.

## Bézier curves

### Definition

A Bézier curve is a parametric curve defined by

```{math}
:label: def-bezier-curve
\mathbf C(u)=\sum_{i=0}^{n} B_{i,n}(u)\, \mathbf P_{i}, \qquad 0 \leq u \leq 1,
```

where $n$ is the **degree** of the curve, the coefficients $\mathbf P_i$
are called control points, and $B_{i,n}$ are the basis functions, which
are $n$-th degree Bernstein polynomials given by the explicit formula

```{math}
:label: def-bernstein-explicit
B_{i,n}(u) = \binom{n}{i} (1-u)^{n-i} u^{i}
= \frac{n!}{i!\,(n-i)!} (1-u)^{n-i} u^{i}.
```

Equivalently, the Bernstein polynomials of degree $n$ can be defined
recursively by blending together two Bernstein polynomials of degree
$n-1$:

```{math}
:label: def-bernstein-recursive
B_{i,n}(u) = (1-u)\, B_{i,n-1}(u) + u\, B_{i-1,n-1}(u).
```

A basis function with an out-of-range index, $i<0$ or $i>n$, is zero by
convention; this makes the recursion above well-defined for $i=0$ and
$i=n$.

### Mathematical properties of Bernstein polynomials

Here is a list of some important properties of Bernstein polynomials:

1. $B_{i,n}(u)$ is a polynomial of degree $n$.

2. Symmetry of the basis functions:

   ```{math}
   B_{i,n}(u) = B_{n-i,n}(1-u).
   ```

3. Global support: $B_{i,n}(u)>0$ for $u\in(0,1)$.

4. Non-negativity: $B_{i,n}(u)\geq0$ for all $i$, $n$, and $u\in[0,1]$.

5. Partition of unity. Since $u+(1-u)=1$, the binomial theorem gives

   ```{math}
   \sum_{i=0}^{n} B_{i,n}(u) = \big[(1-u)+u\big]^{n} = 1
   \qquad \text{for all } u\in[0,1].
   ```

6. Continuity and differentiability: Bernstein polynomials are
   continuous and infinitely differentiable.

7. Extrema: for $n\geq1$, $B_{i,n}(u)$ attains exactly one maximum in
   $u\in[0,1]$, located at $u=i/n$. A degree-zero basis function is the
   constant $B_{0,0}(u)=1$ and has no isolated maximum.

8. The first derivative of the Bernstein polynomials is, for $n\geq1$,

   ```{math}
   :label: orig-bernstein-derivative
   B'_{i,n}(u) = \frac{\mathrm dB_{i,n}}{\mathrm du}
   = n\big(B_{i-1,n-1}(u) - B_{i,n-1}(u)\big).
   ```

9. $B_{0,n}(u=0)=1$ and $B_{i,n}(u=0)=0$ for $i\neq0$.

10. $B_{n,n}(u=1)=1$ and $B_{i,n}(u=1)=0$ for $i\neq n$.

### Mathematical properties of Bézier curves

Here is a list of some important properties of Bézier curves:

1. A Bézier curve of degree $n$ is defined by $n+1$ control points, and
   the components of $\mathbf C(u)$ are polynomials of degree **at
   most** $n$: a particular choice of control points can cancel the
   leading term and reduce the represented degree.

2. Affine invariance: Bézier curves are invariant under affine
   transformations such as rotations, displacements, and scalings. One
   can apply an affine transformation to the curve by applying it to
   its control points, because for $F(\mathbf x)=A\mathbf x+\mathbf b$
   the partition-of-unity property gives
   $F(\mathbf C(u))=\sum_i B_{i,n}(u)F(\mathbf P_i)$.

3. Convex hull property: all the points of a Bézier curve are contained
   in the *convex hull* of its control points. The convex hull of a
   set of points $P=\{\mathbf P_0,\ldots,\mathbf P_N\}$ is denoted
   $\mathcal{CH}(P)$ and is the set of all convex combinations of
   points:

   ```{math}
   \mathcal{CH}(P) = \Big\{\, \mathbf C = \sum_{k=0}^{N} a_k\, \mathbf P_k
   \ \text{such that}\ \sum_{k=0}^{N} a_k = 1 \text{ and } a_k\geq0
   \text{ for } k=0,1,\ldots,N \,\Big\}.
   ```

   The convex hull property follows from the non-negativity and the
   partition-of-unity properties of the basis functions.

4. Global modification scheme: modifying any of the interior control
   points affects the location of every point of the Bézier curve
   except at $u=0$ and $u=1$.

5. The polygon formed by the set of control points is known as the
   *control polygon*. The control polygon represents a piecewise
   linear approximation to the Bézier curve.

6. Variation diminishing property: no straight line (or plane in three
   dimensions) intersects the Bézier curve more times than it
   intersects its control polygon, counting multiplicities and
   tangential intersections consistently on both sides. Intuitively,
   the curve does not wiggle more than its control polygon.

7. Continuity and differentiability: Bézier curves are continuous and
   infinitely differentiable, since they are polynomial in $u$. Note
   that this only guarantees smoothness of the *parametrization*: the
   traced-out curve can still have a stationary point or a cusp where
   $\mathbf C'(u)=\mathbf 0$.

8. First and higher order derivatives:

   The first derivative of a Bézier curve is given by substituting
   {eq}`orig-bernstein-derivative` and shifting the summation index:

   ```{math}
   \frac{\mathrm d\mathbf C}{\mathrm du}
   = \sum_{i=0}^{n} B'_{i,n}(u)\, \mathbf P_i
   = \sum_{i=0}^{n} n\big(B_{i-1,n-1}(u)-B_{i,n-1}(u)\big)\mathbf P_i
   = n \sum_{i=0}^{n-1} B_{i,n-1}(u)\,(\mathbf P_{i+1}-\mathbf P_i).
   ```

   The first derivative of a Bézier curve of degree $n$ is itself a
   Bézier curve of degree $n-1$, called the *hodograph* of the original
   curve:

   ```{math}
   \frac{\mathrm d\mathbf C}{\mathrm du}
   = \sum_{i=0}^{n-1} B_{i,n-1}(u)\, \mathbf P_i^{(1)}
   \qquad \text{with} \qquad
   \mathbf P_i^{(1)} = n\,(\mathbf P_{i+1}-\mathbf P_i).
   ```

   Since the derivative of a Bézier curve is another Bézier curve, the
   original curve can be differentiated recursively to compute its
   $k$-th derivative, for $0\leq k\leq n$:

   ```{math}
   :label: bezier-kth-derivative
   \frac{\mathrm d^{k}\mathbf C}{\mathrm du^{k}}
   = \sum_{i=0}^{n-k} B_{i,n-k}(u)\, \mathbf P_i^{(k)},
   ```

   where

   ```{math}
   :label: bezier-derivative-points
   \mathbf P_i^{(k)} = \begin{cases}
   \mathbf P_i & \text{if } k=0,\\[2pt]
   (n-k+1)\big(\mathbf P_{i+1}^{(k-1)} - \mathbf P_i^{(k-1)}\big) & \text{if } k\geq1.
   \end{cases}
   ```

   Both control points on the right come from the *previous* derivative
   level $k-1$. Derivatives of order greater than $n$ vanish identically.

9. Endpoint interpolation: the start and end points of a Bézier curve
   coincide with the first and last control points, respectively:

   ```{math}
   \mathbf C(u=0) = \mathbf P_0, \qquad \mathbf C(u=1) = \mathbf P_n.
   ```

10. Endpoint tangency: the Bézier curve is tangent to the control
    polygon at the endpoints:

    ```{math}
    \frac{\mathrm d\mathbf C}{\mathrm du}\Big|_{u=0} = n\,(\mathbf P_1-\mathbf P_0),
    \qquad
    \frac{\mathrm d\mathbf C}{\mathrm du}\Big|_{u=1} = n\,(\mathbf P_n-\mathbf P_{n-1}).
    ```

11. Endpoint curvature: for $n\geq2$, the second derivative of a Bézier
    curve at its endpoints is given by

    ```{math}
    \frac{\mathrm d^2\mathbf C}{\mathrm du^2}\Big|_{u=0}
    = n(n-1)\big[(\mathbf P_2-\mathbf P_0) - 2(\mathbf P_1-\mathbf P_0)\big],
    \qquad
    \frac{\mathrm d^2\mathbf C}{\mathrm du^2}\Big|_{u=1}
    = n(n-1)\big[(\mathbf P_{n-2}-\mathbf P_n) - 2(\mathbf P_{n-1}-\mathbf P_n)\big].
    ```

    Provided the endpoint control-polygon edge is nonzero, the
    curvature of a Bézier curve at its endpoints is given by

    ```{math}
    :label: orig-bezier-endpoint-curvature
    \kappa(u=0) = \left(\frac{n-1}{n}\right)
    \frac{\big\|(\mathbf P_2-\mathbf P_0)\times(\mathbf P_1-\mathbf P_0)\big\|}
    {\|\mathbf P_1-\mathbf P_0\|^3},
    \qquad
    \kappa(u=1) = \left(\frac{n-1}{n}\right)
    \frac{\big\|(\mathbf P_{n-2}-\mathbf P_n)\times(\mathbf P_{n-1}-\mathbf P_n)\big\|}
    {\|\mathbf P_{n-1}-\mathbf P_n\|^3}.
    ```

    These formulas require $n\geq2$ and a nonzero endpoint tangent; for
    a planar curve the cross-product norm can be replaced by the
    absolute value of the corresponding $2\times2$ determinant. See
    [G² continuity](g2_continuity.md) for how this expression is used
    to join curve segments smoothly.

## Bézier surfaces

### Definition

A Bézier surface is a parametric surface defined by

```{math}
:label: def-bezier-surface
\mathbf S(u,v) = \sum_{i=0}^{n}\sum_{j=0}^{m}
B_{i,n}(u)\, B_{j,m}(v)\, \mathbf P_{i,j},
\qquad 0\leq(u,v)\leq1,
```

where $n$ and $m$ are the degrees of the surface in the $u$- and
$v$-directions, the coefficients $\mathbf P_{i,j}$ are a bidirectional
net of control points, and $B_{i,n}(u)B_{j,m}(v)$ is the product of
univariate Bernstein polynomials.

The $u$-direction Bernstein polynomials are given by the explicit
formula {eq}`def-bernstein-explicit` or by the recursive relation
{eq}`def-bernstein-recursive`, while the $v$-direction basis functions
are defined analogously, replacing the variable $u$ by $v$ and the
indices $i$ and $n$ by $j$ and $m$, respectively.

### Mathematical properties of tensor-product Bernstein polynomials

Here is a list of some important properties of the tensor-product
Bernstein polynomials:

1. $B_{i,n}(u)$ is a polynomial of degree $n$; $B_{j,m}(v)$ is a
   polynomial of degree $m$.

2. Global support: $B_{i,n}(u)B_{j,m}(v)>0$ for
   $(u,v)\in(0,1)\times(0,1)$.

3. Non-negativity: $B_{i,n}(u)B_{j,m}(v)\geq0$ for all $i$, $j$, $n$,
   $m$, and $(u,v)\in[0,1]\times[0,1]$.

4. Partition of unity:

   ```{math}
   \sum_{i=0}^{n}\sum_{j=0}^{m} B_{i,n}(u)\,B_{j,m}(v) = 1
   \qquad \text{for all } (u,v)\in[0,1]\times[0,1],
   ```

   which follows because the sum factors into the product of the two
   univariate partitions of unity.

5. Continuity and differentiability: tensor-product Bernstein
   polynomials are continuous and infinitely differentiable.

6. Extrema: for $n,m\geq1$, $B_{i,n}(u)B_{j,m}(v)$ attains exactly one
   maximum in $(u,v)\in[0,1]\times[0,1]$, located at $(u,v)=(i/n,j/m)$.

7. The first partial derivatives of the tensor-product Bernstein
   polynomials are given by

   ```{math}
   \frac{\partial}{\partial u}\big(B_{i,n}(u)B_{j,m}(v)\big)
   = B_{j,m}(v)\,\frac{\partial}{\partial u}\big(B_{i,n}(u)\big),
   \qquad
   \frac{\partial}{\partial v}\big(B_{i,n}(u)B_{j,m}(v)\big)
   = B_{i,n}(u)\,\frac{\partial}{\partial v}\big(B_{j,m}(v)\big).
   ```

### Mathematical properties of Bézier surfaces

Here is a list of some important properties of Bézier surfaces:

1. A Bézier surface of bidegree $(n,m)$ is defined by $(n+1)\times(m+1)$
   control points, and the components of $\mathbf S(u,v)$ are bivariate
   polynomials of degree at most $n$ in $u$ and at most $m$ in $v$.

2. Affine invariance: Bézier surfaces are invariant under affine
   transformations such as rotations, displacements, and scalings. One
   can apply an affine transformation to the surface by applying it to
   its control net.

3. Convex hull property: all the points of a Bézier surface are
   contained in the convex hull of its control net,

   ```{math}
   \mathcal{CH}(P) = \Big\{\, \mathbf S = \sum_{k=0}^{N} a_k\, \mathbf P_k
   \ \text{such that}\ \sum_{k=0}^{N} a_k = 1 \text{ and } a_k\geq0
   \text{ for } k=0,1,\ldots,N \,\Big\},
   ```

   which follows from the non-negativity and partition-of-unity
   properties of the basis functions.

4. Global modification scheme: modifying an interior control point
   $\mathbf P_{i,j}$, with $0<i<n$ and $0<j<m$, affects every interior
   point of the surface, but it leaves **all four boundary curves**
   unchanged, since $B_{i,n}(u)B_{j,m}(v)$ vanishes whenever $u$ or $v$
   is $0$ or $1$.

5. If triangulated, the net of control points represents a piecewise
   planar approximation to the Bézier surface.

6. No known variation-diminishing property: the curve result does not
   extend directly to surfaces.

7. Continuity and differentiability: Bézier surfaces are continuous and
   infinitely differentiable.

8. First and higher order derivatives:

   The first partial derivatives of a Bézier surface are given by

   ```{math}
   \frac{\partial\mathbf S}{\partial u}
   = n \sum_{i=0}^{n-1}\sum_{j=0}^{m}
   B_{i,n-1}(u)\,B_{j,m}(v)\,(\mathbf P_{i+1,j}-\mathbf P_{i,j}),
   \qquad
   \frac{\partial\mathbf S}{\partial v}
   = m \sum_{i=0}^{n}\sum_{j=0}^{m-1}
   B_{i,n}(u)\,B_{j,m-1}(v)\,(\mathbf P_{i,j+1}-\mathbf P_{i,j}).
   ```

   Since the derivative of a Bézier surface is again a Bézier surface,
   the original surface can be differentiated recursively to compute
   its $(k,l)$-th partial derivative:

   ```{math}
   :label: bezier-surface-kth-derivative
   \frac{\partial^{k+l}\mathbf S}{\partial u^{k}\partial v^{l}}
   = \sum_{i=0}^{n-k}\sum_{j=0}^{m-l}
   B_{i,n-k}(u)\,B_{j,m-l}(v)\, \mathbf P_{i,j}^{(k,l)},
   ```

   where

   ```{math}
   :label: bezier-surface-derivative-points
   \mathbf P_{i,j}^{(k,l)} = \begin{cases}
   \mathbf P_{i,j} & \text{if } k=l=0,\\[2pt]
   (m-l+1)\big(\mathbf P_{i,j+1}^{(0,l-1)} - \mathbf P_{i,j}^{(0,l-1)}\big)
   & \text{if } k=0,\ l\geq1,\\[2pt]
   (n-k+1)\big(\mathbf P_{i+1,j}^{(k-1,l)} - \mathbf P_{i,j}^{(k-1,l)}\big)
   & \text{if } k\geq1,\ l\geq0.
   \end{cases}
   ```

   The three cases first initialize the net, then apply all of the
   required $v$-differences, then apply all of the required
   $u$-differences to whatever $v$-difference level is needed; this is
   the order in which the recursion is actually evaluated. A
   derivative vanishes as soon as $k>n$ or $l>m$.

9. Corner point interpolation: the corners of a Bézier surface coincide
   with the corner points of its control net,

   ```{math}
   \mathbf S(u=0,v=0) = \mathbf P_{0,0}, \qquad
   \mathbf S(u=1,v=0) = \mathbf P_{n,0}, \qquad
   \mathbf S(u=0,v=1) = \mathbf P_{0,m}, \qquad
   \mathbf S(u=1,v=1) = \mathbf P_{n,m}.
   ```

   More generally, each edge of the control net traces out a boundary
   curve of the surface: $\mathbf S(u,0)=\sum_iB_{i,n}(u)\mathbf P_{i,0}$,
   $\mathbf S(u,1)=\sum_iB_{i,n}(u)\mathbf P_{i,m}$,
   $\mathbf S(0,v)=\sum_jB_{j,m}(v)\mathbf P_{0,j}$, and
   $\mathbf S(1,v)=\sum_jB_{j,m}(v)\mathbf P_{n,j}$, each of which is
   itself a Bézier curve.
