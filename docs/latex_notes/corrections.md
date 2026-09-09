# Theory notes: corrections and LaTeX–Markdown comparison

Review date: **9 September 2026**.

This report compares the original LaTeX notes in `docs/` with the current
Markdown theory in `docs/source/theory/`. It records what changed during
the documentation migration and the findings from a second, detailed pass.
**This review adds this report; it does not modify the LaTeX or Markdown
theory files.** Line references refer to the files as reviewed on this date.

The Markdown is an **edited, condensed version**, with additional material.
It is not a literal transcription. In particular:

- The original endpoint first derivatives, second derivatives, and curvature
  formulas are preserved mathematically, under their clamping, degree,
  weight, and regularity assumptions.
- Actual corrections include recursive derivative indices, surface summation
  bounds, the NURBS surface denominator, and homogeneous surface formulas.
- The G² discussion now requires matching oriented tangents and curvature
  vectors. Matching curvature magnitudes alone is insufficient.
- The four original G² illustrations, several intermediate derivation steps,
  and the original chapter/page citation locators were not carried over.
- The current Markdown still deserves a few small clarifications, listed in
  [Recommended follow-ups](#recommended-follow-ups).

## Reading this report

| Classification | Meaning |
| --- | --- |
| **Correction** | The original expression or claim needed a mathematical or indexing correction. |
| **Qualification** | The result is retained with explicit assumptions or a more precise scope. |
| **Equivalent** | Algebra, notation, or presentation changed without changing the result. |
| **Addition** | Material in Markdown that was not explicitly present in the LaTeX notes. |
| **Omission** | Material in LaTeX that is no longer explicitly present in the Markdown theory. |
| **Follow-up** | A remaining issue or suggested improvement; not silently applied by this audit. |

Whitespace, spelling, accents, and ordinary prose rewording are grouped as
editorial changes rather than listed word by word. Mathematical expressions,
assumptions, properties, derivations, illustrations, and citations are
compared by topic below. “Preserved” can mean retained in a more general
formula or moved to a linked page; the tables distinguish those cases.

## Files compared

| Original LaTeX | Markdown destination |
| --- | --- |
| [Bézier curves](<docs/A Briefing on Bezier Curves and Surfaces/Bezier_curves.tex>) | [bezier.md](docs/source/theory/bezier.md), curve sections |
| [Bézier surfaces](<docs/A Briefing on Bezier Curves and Surfaces/Bezier_surfaces.tex>) | [bezier.md](docs/source/theory/bezier.md), surface sections |
| [B-spline curves](<docs/A Briefing on B-Spline Curves and Surfaces/BSpline_curves.tex>) | [bspline.md](docs/source/theory/bspline.md), curve sections |
| [B-spline surfaces](<docs/A Briefing on B-Spline Curves and Surfaces/BSpline_surfaces.tex>) | [bspline.md](docs/source/theory/bspline.md), surface sections |
| [NURBS curves](<docs/A Briefing on NURBS Curves and Surfaces/NURBS_curves.tex>) | [nurbs.md](docs/source/theory/nurbs.md), curve sections |
| [NURBS surfaces](<docs/A Briefing on NURBS Curves and Surfaces/NURBS_surfaces.tex>) | [nurbs.md](docs/source/theory/nurbs.md), surface sections |
| [G² motivation](<docs/G2 continuity of NURBS curves/1_motivation.tex>) | [g2_continuity.md](docs/source/theory/g2_continuity.md), opening sections |
| [G² Bézier endpoint formulas](<docs/G2 continuity of NURBS curves/2_bezier_curvature_formulas.tex>) | [bezier.md](docs/source/theory/bezier.md#endpoints) and the G² special case |
| [G² B-spline endpoint formulas](<docs/G2 continuity of NURBS curves/3_bspline_curvature_formulas.tex>) | [bspline.md](docs/source/theory/bspline.md#clamped-endpoints) and the G² special case |
| [G² NURBS endpoint formulas](<docs/G2 continuity of NURBS curves/4_nurbs_curvature_formulas.tex>) | [nurbs.md](docs/source/theory/nurbs.md#clamped-endpoints) and the G² derivation |
| [Airfoil parametrization](<docs/G2 continuity of NURBS curves/5_airfoil_parametrization.tex>) | [g2_continuity.md](docs/source/theory/g2_continuity.md#constructing-control-points-for-a-specified-radius) |
| [Final remarks](<docs/G2 continuity of NURBS curves/6_final_remarks.tex>) | [g2_continuity.md](docs/source/theory/g2_continuity.md#relation-to-the-original-airfoil-notes) |
| [Original figure declarations](<docs/G2 continuity of NURBS curves/7_figures.tex>) | No direct equivalent; see omissions below |

The four main LaTeX documents, their preambles, and their `references.bib`
files were also checked for introductions, macros, included material, and
citations. On the Markdown side, this includes the
[theory introduction](docs/source/theory/index.md),
[reference page](docs/source/references/bibliography.md), and
[theory bibliography](docs/source/references/theory.bib).

The comparison uses the LaTeX source as the original text. The legacy
compiled PDFs were not rebuilt or assumed to be synchronized with it.
The figure declarations, captions, and omitted asset paths were checked;
the mathematical content drawn inside those PDF illustrations was not
independently audited.

## Changes common to several pages

| ID | Classification | Original → Markdown; assessment |
| --- | --- | --- |
| C01 | Correction | `n` or `p` is repeatedly called the “order” while the equations use it as the degree. Markdown calls it **degree** and defines order as degree plus one. Control-point counts and the actual polynomial exponents are unchanged. This affects all three briefings and the repeated definitions in G² sections 2–4. |
| C02 | Equivalent | `r=n+p+1` and `s=m+q+1` are largely replaced by the explicit knot counts `n+p+2` and `m+q+2`. The old highest-knot-index symbols disappear; the counts do not change. The symbol `r` is reused locally for knot multiplicity. |
| C03 | Qualification / addition | The theory introduction now states normalized clamping, real geometry, positive rational weights, one-sided endpoint derivatives, and regularity assumptions. The LaTeX introduces clamping in the B-spline/NURBS curve notes but does not state all these assumptions consistently across its standalone documents. Positive weights are sufficient for the stated guarantees, not necessary for every valid rational representation. |
| C04 | Qualification | The B-spline page explicitly gives the general domain $[u_p,u_{n+1}]$, then specializes to $[0,1]$. The LaTeX initially writes $[0,1]$ alongside an otherwise general nondecreasing knot vector. Surface-domain notation changes from $0\leq(u,v)\leq1$ to $[0,1]^2$ or inherits the common assumptions. |
| C05 | Qualification | The zero-denominator convention is stated as “a term with a zero denominator is zero,” rather than just defining $0/0=0$. The clamped right endpoint is explicitly evaluated by its left limit, reconciling the half-open degree-zero intervals with endpoint interpolation. |
| C06 | Qualification | Continuity at a knot of multiplicity $r$ is a **guarantee** of $C^{p-r}$ for $1\leq r\leq p$; multiplicity $p+1$ can permit a jump. Special control points can make a curve smoother. Markdown also distinguishes spanwise derivatives from nonexistent two-sided derivatives at a repeated knot. |
| C07 | Qualification | “Polynomial of degree $p$” becomes **piecewise** polynomial of degree at most $p$, where appropriate. Rational numerators/denominators are likewise described span by span. A surface degree written as $p\times q$ is interpreted as bidegree $(p,q)$, not total degree $pq$. |
| C08 | Equivalent | Dots and `d/du` notation are mostly replaced by primes; partial derivatives use subscripts or $(k,l)$. `\norm{...}` becomes `\|...\|`, `\mathds{R}` becomes `\mathbb R`, and the convex hull is written `\operatorname{conv}`. These substitutions do not change the operations. |

## Bézier notes

### B01 — Recursive curve derivative control points

**Correction.** [LaTeX line 139](<docs/A Briefing on Bezier Curves and Surfaces/Bezier_curves.tex#L139>)
uses the unknown current derivative level on the right:

$$
\mathbf P_i^{(k)}
=(n-k+1)\bigl(\mathbf P_{i+1}^{(k-1)}-\mathbf P_i^{(k)}\bigr).
$$

[Markdown: Derivatives and the hodograph](docs/source/theory/bezier.md#derivatives-and-the-hodograph)
uses the preceding level for both points:

$$
\mathbf P_i^{(k)}
=(n-k+1)\bigl(\mathbf P_{i+1}^{(k-1)}-\mathbf P_i^{(k-1)}\bigr).
$$

The factor, basis degree $n-k$, and summation limit $n-k$ are unchanged.
This is a substantive correction, not a notation change.

### B02 — Surface first derivative in the v direction

**Correction.** [LaTeX line 117](<docs/A Briefing on Bezier Curves and Surfaces/Bezier_surfaces.tex#L117>)
uses $\sum_{i=0}^{m-1}\sum_{j=0}^{m}$ for $\mathbf S_v$. The first limit
should follow the $u$ control-point count, and the second must stop before
the last $v$ point because the difference accesses $j+1$.

[Markdown: Partial derivatives](docs/source/theory/bezier.md#partial-derivatives)
correctly writes

$$
\mathbf S_v
=m\sum_{i=0}^{n}\sum_{j=0}^{m-1}
B_{i,n}(u)B_{j,m-1}(v)
(\mathbf P_{i,j+1}-\mathbf P_{i,j}).
$$

The original $\mathbf S_u$ formula is preserved.

### B03 — Surface derivative recursion conditions

**Correction / equivalent replacement.**
[LaTeX lines 127–129](<docs/A Briefing on Bezier Curves and Surfaces/Bezier_surfaces.tex#L127>)
have three inconsistent cases:

- The base case uses `k=0` alone, incorrectly including $l>0$.
- The $v$-difference case says `k>0 and l=0` but accesses derivative level $l-1$.
- The $u$-difference case requires `l>0`, leaving pure $u$ derivatives uncovered.

The corrected cases are

$$
\mathbf P_{i,j}^{(k,l)}=
\begin{cases}
\mathbf P_{i,j}, & k=l=0,\\
(m-l+1)(\mathbf P_{i,j+1}^{(0,l-1)}-\mathbf P_{i,j}^{(0,l-1)}),
 & k=0,\ l\geq1,\\
(n-k+1)(\mathbf P_{i+1,j}^{(k-1,l)}-\mathbf P_{i,j}^{(k-1,l)}),
 & k\geq1,\ l\geq0.
\end{cases}
$$

Markdown describes that sequence in prose and primarily displays the
equivalent forward-difference formula:

$$
\mathbf S^{(k,l)}
=\frac{n!}{(n-k)!}\frac{m!}{(m-l)!}
\sum_{i=0}^{n-k}\sum_{j=0}^{m-l}
B_{i,n-k}(u)B_{j,m-l}(v)\Delta_u^k\Delta_v^l\mathbf P_{i,j}.
$$

The factorial form is new presentation of the corrected recurrence,
not a different derivative definition.

### B04 — Effect of an interior surface control point

**Correction.** [LaTeX line 102](<docs/A Briefing on Bezier Curves and Surfaces/Bezier_surfaces.tex#L102>)
says an interior control point affects the surface everywhere except at
the four corners. [Markdown](docs/source/theory/bezier.md#tensor-product-bézier-surfaces)
states that it leaves **all four boundary curves** unchanged.

For an interior index $0<i<n$, $0<j<m$, the coefficient
$B_{i,n}(u)B_{j,m}(v)$ vanishes whenever $u$ or $v$ is an endpoint.
Thus the original claim was too broad.

### Other Bézier deviations

| ID | Classification | LaTeX location | Markdown result |
| --- | --- | --- | --- |
| B05 | Equivalent notation correction | [Curve line 54](<docs/A Briefing on Bezier Curves and Surfaces/Bezier_curves.tex#L54>) | $(1-t+t)^n$ becomes $[(1-u)+u]^n$. Both equal one, but `u` matches the defined parameter. |
| B06 | Equivalent notation correction | [Surface line 55](<docs/A Briefing on Bezier Curves and Surfaces/Bezier_surfaces.tex#L55>) | The second factor changes from $B_{j,m}(u)$ to $B_{j,m}(v)$. The original double sum also equals one, but it does not write the intended two-parameter basis. |
| B07 | Correction | [Surface lines 39, 41, 84](<docs/A Briefing on Bezier Curves and Surfaces/Bezier_surfaces.tex#L39>) | Degrees `p,q` attached to `B_i,n,B_j,m` and the surface degree `n × q` become consistent `n,m` notation and bidegree $(n,m)$. |
| B08 | Correction | [Surface line 95](<docs/A Briefing on Bezier Curves and Surfaces/Bezier_surfaces.tex#L95>) | The original convex-hull definition constrains $a_k\geq0$ only for $k=1,\ldots,N$, missing $a_0$. The shared Markdown convex-hull definition includes every coefficient, starting at zero. |
| B09 | Qualification | [Curve lines 63, 86](<docs/A Briefing on Bezier Curves and Surfaces/Bezier_curves.tex#L63>) | A unique maximizing parameter is restricted to $n>0$ and its location $i/n$ is added; degree zero is constant. Curve coordinates have degree **at most** $n$, since cancellations can reduce the represented degree. |
| B10 | Addition | [Surface corner formulas, line 139](<docs/A Briefing on Bezier Curves and Surfaces/Bezier_surfaces.tex#L139>) | The four corner identities are retained, with explicit boundary-curve formulas added. The nearby original description incorrectly calls this a clamped B-spline surface in the Bézier section. |
| B11 | Equivalent / qualification | [Curve line 111](<docs/A Briefing on Bezier Curves and Surfaces/Bezier_curves.tex#L111>) and [surface line 106](<docs/A Briefing on Bezier Curves and Surfaces/Bezier_surfaces.tex#L106>) | The curve variation-diminishing statement is retained with intersection-counting qualifications. The surface statement is phrased cautiously as the curve result not extending directly, rather than asserting a survey of all known results. |
| B12 | Equivalent / omission | [Curve lines 121–128](<docs/A Briefing on Bezier Curves and Surfaces/Bezier_curves.tex#L121>) | The intermediate basis-derivative substitution lines and separate first-derivative control-point display are condensed into the hodograph formula and general recurrence. |
| B13 | Omission | [Surface lines 64, 68–69, 104](<docs/A Briefing on Bezier Curves and Surfaces/Bezier_surfaces.tex#L64>) | The product-basis maximum statement, separate first product-basis derivative identities, and triangulated-control-net interpretation are not repeated. Product derivatives are implicit in the displayed surface derivatives. The positive-degree maximum location $(i/n,j/m)$ could be restored with $n,m>0$. |

The explicit Bernstein definition, its symmetry and recurrence,
nonnegativity, partition of unity, endpoint basis values, affine invariance,
convex-hull property, hodograph, curve endpoint interpolation/tangency,
and surface definition are otherwise preserved, with the qualifications above.
The separate factorial expansion of $\binom ni$ is omitted as standard notation.

## B-spline notes

### S01 — Degree index in partition of unity

**Correction.** [Curve lines 59 and 64](<docs/A Briefing on B-Spline Curves and Surfaces/BSpline_curves.tex#L59>)
use $N_{i,n}$, although the basis degree is $p$. Markdown uses

$$
\sum_{i=0}^nN_{i,p}(u)=1,\qquad
\sum_{i=s-p}^{s}N_{i,p}(u)=1
$$

on the full domain and the active span, respectively.
Changing `n` to `p` matters whenever degree and highest control-point
index differ.

### S02 — Derivative control points and shortened knot vectors

**Correction.**
[Curve line 195](<docs/A Briefing on B-Spline Curves and Surfaces/BSpline_curves.tex#L195>)
has the same current-level error as B01:
$\mathbf P_{i+1}^{(k-1)}-\mathbf P_i^{(k)}$.
[Markdown](docs/source/theory/bspline.md#curve-derivatives) uses

$$
\mathbf P_i^{(k)}
=\frac{p-k+1}{u_{i+p+1}-u_{i+k}}
(\mathbf P_{i+1}^{(k-1)}-\mathbf P_i^{(k-1)}).
$$

The denominator and factor were already correct. Separately,
[LaTeX lines 174–184](<docs/A Briefing on B-Spline Curves and Surfaces/BSpline_curves.tex#L174>)
repeat $\sum_{i=0}^nN'_{i,p}\mathbf P_i$ and then say it is evaluated
on a shortened vector. That is not the proper lower-degree representation:
the differentiated original basis uses the original vector.

Markdown separates the two valid alternatives:

$$
\mathbf C^{(k)}
=\sum_{i=0}^{n}N_{i,p}^{(k)}(u;U)\mathbf P_i
=\sum_{i=0}^{n-k}N_{i,p-k}(u;U^{(k)})\mathbf P_i^{(k)},
$$

where $U^{(k)}=[u_k,\ldots,u_{n+p+1-k}]$.
For the normalized clamped case, this shortened vector is equivalent
to the original repeated-endpoint list at LaTeX line 201.
The knot-shortening idea was present originally; the migration did not
introduce it. The distinction between the two representations was clarified.
See also [Shene's derivative derivation](https://pages.mtu.edu/~shene/COURSES/cs3621/NOTES/spline/B-spline/bspline-derv.html).

### S03 — Reduction of surface bases to Bernstein polynomials

**Correction / condensed presentation.**
[Surface line 97](<docs/A Briefing on B-Spline Curves and Surfaces/BSpline_surfaces.tex#L97>)
writes

$$
N_{i,p}(u)N_{j,q}(u)=B_{i,p}(u)B_{i,q}(u).
$$

With clamped knots and $(p,q)=(n,m)$, the intended identity is

$$
N_{i,p}(u)N_{j,q}(v)=B_{i,n}(u)B_{j,m}(v).
$$

The original has both a wrong second argument and a wrong second basis
index. Markdown retains the univariate reduction and states the resulting
surface reduction, but **does not explicitly display** this corrected
product identity.

### Other B-spline deviations

| ID | Classification | LaTeX location | Markdown result |
| --- | --- | --- | --- |
| S04 | Correction | [Surface line 129](<docs/A Briefing on B-Spline Curves and Surfaces/BSpline_surfaces.tex#L129>) | As in B08, $a_0\geq0$ was missing from the convex-hull set definition. Markdown uses the shared definition and the strong convex-hull statement. |
| S05 | Correction / condensed presentation | [Surface line 140](<docs/A Briefing on B-Spline Curves and Surfaces/BSpline_surfaces.tex#L140>) | $N_{j,q}(u)$ should be $N_{j,q}(v)$ in the local-support explanation. Markdown gives the correct tensor-product definition and affected rectangle without repeating that erroneous line. |
| S06 | Qualification / omission | [Curve line 76](<docs/A Briefing on B-Spline Curves and Surfaces/BSpline_curves.tex#L76>), [surface line 83](<docs/A Briefing on B-Spline Curves and Surfaces/BSpline_surfaces.tex#L83>) | The unconditional unique-maximum statements are removed. Degree-zero bases already give counterexamples to a unique maximizing parameter; no replacement theorem for arbitrary knots is asserted. |
| S07 | Equivalent | [Surface lines 87–94 and 154–156](<docs/A Briefing on B-Spline Curves and Surfaces/BSpline_surfaces.tex#L87>) | Separate first product-basis and surface derivative displays are consolidated into $\mathbf S^{(k,l)}=\sum_{i,j}N_{i,p}^{(k)}(u)N_{j,q}^{(l)}(v)\mathbf P_{i,j}$. |
| S08 | Equivalent / omission | [Surface lines 66–73](<docs/A Briefing on B-Spline Curves and Surfaces/BSpline_surfaces.tex#L66>) | The full and active-rectangle partition-of-unity equations are stated in words through the product basis and active support, not reproduced as separate displays. Active index ranges are recoverable from the curve basis section. |
| S09 | Addition | [Surface corner section, line 159](<docs/A Briefing on B-Spline Curves and Surfaces/BSpline_surfaces.tex#L159>) | Boundary rows/columns defining B-spline curves are explicitly described. The four original corner interpolation identities are retained in prose. |
| S10 | Omission | [Curve line 156](<docs/A Briefing on B-Spline Curves and Surfaces/BSpline_curves.tex#L156>), [surface lines 142–144](<docs/A Briefing on B-Spline Curves and Surfaces/BSpline_surfaces.tex#L142>) | The control-polygon approximation explanation is not repeated separately, and the triangulated-control-net and “no known variation diminishing property” surface bullets are omitted. |

The Cox–de Boor recurrence, basis derivative recurrence, support intervals,
active-basis count, clamping multiplicities, affine invariance, local shape
control, and general surface derivative formula retain their mathematical
content. The reference to de Boor's triangular computation and the
parenthetical reference to handwritten motivation were omitted.

## NURBS notes

### N01 — Rational surface denominator uses the v degree

**Correction.**
[Surface line 16](<docs/A Briefing on NURBS Curves and Surfaces/NURBS_surfaces.tex#L16>)
uses $N_{l,p}(v)$ in the denominator; it must use $N_{l,q}(v)$.
[Markdown](docs/source/theory/nurbs.md#tensor-product-nurbs-surfaces) writes

$$
R_{i,j}^{p,q}(u,v)=
\frac{N_{i,p}(u)N_{j,q}(v)w_{i,j}}
{\sum_{a=0}^n\sum_{b=0}^mN_{a,p}(u)N_{b,q}(v)w_{a,b}}.
$$

The dummy indices change from $k,l$ to $a,b$ as well; that part is
only notation. The degree correction matters when $p\ne q$.

### N02 — Homogeneous surfaces use polynomial bases

**Correction.**
[Surface line 177](<docs/A Briefing on NURBS Curves and Surfaces/NURBS_surfaces.tex#L177>)
writes $\mathbf S^w=\sum_{i,j}R_{i,j}^{p,q}\mathbf P^w_{i,j}$.
That applies rational weighting to points that already contain their
weights, so projection generally does not recover the intended surface.

[Markdown](docs/source/theory/nurbs.md#homogeneous-surface-and-partial-derivatives)
correctly uses

$$
\mathbf P^w_{i,j}=(w_{i,j}\mathbf P_{i,j},w_{i,j}),\qquad
\mathbf S^w(u,v)=
\sum_{i,j}N_{i,p}(u)N_{j,q}(v)\mathbf P^w_{i,j}.
$$

The analogous homogeneous **curve** expression at
[curve line 187](<docs/A Briefing on NURBS Curves and Surfaces/NURBS_curves.tex#L187>)
was already correct and is preserved.

### N03 — Arguments in homogeneous projection and derivatives

**Corrections.** The following surface expressions now consistently use
the second parameter $v$:

| Original location | Original factor | Correct factor represented in Markdown |
| --- | --- | --- |
| [Line 183](<docs/A Briefing on NURBS Curves and Surfaces/NURBS_surfaces.tex#L183>) | $N_{j,q}(u,v)$ in the homogeneous projection sum | $N_{j,q}(v)$ |
| [Line 184](<docs/A Briefing on NURBS Curves and Surfaces/NURBS_surfaces.tex#L184>) | $N_{j,q}(u)$ in the numerator | $N_{j,q}(v)$ |
| [Line 194](<docs/A Briefing on NURBS Curves and Surfaces/NURBS_surfaces.tex#L194>) | $N_{j,q}(u)$ in the denominator defining $w(u,v)$ | $N_{j,q}(v)$ |
| [Line 223](<docs/A Briefing on NURBS Curves and Surfaces/NURBS_surfaces.tex#L223>) | $N_{j,q}^{(l)}(u)$ in the homogeneous derivative | $N_{j,q}^{(l)}(v)$ |

Markdown combines these into the homogeneous definition,
$\mathbf S=\mathbf A/w$, and the displayed derivatives of $(\mathbf A,w)$.
It does not reproduce every intermediate line of the original projection.

### N04 — The surface quotient recurrence was already correct

**Equivalent.** The three sums in
[surface line 215](<docs/A Briefing on NURBS Curves and Surfaces/NURBS_surfaces.tex#L215>)
are mathematically correct. Markdown combines them into one sum:

$$
\mathbf S^{(k,l)}=
\frac1w\left[
\mathbf A^{(k,l)}-
\sum_{\substack{0\leq a\leq k,\ 0\leq b\leq l\\(a,b)\ne(0,0)}}
\binom ka\binom lb w^{(a,b)}\mathbf S^{(k-a,l-b)}
\right].
$$

Partitioning its index set into $a>0,b=0$, $a=0,b>0$, and $a>0,b>0$
recovers the original exactly. No mixed terms were removed or added.
The prose typo $\mathbf S^{(k)}$ at LaTeX line 207 becomes $(k,l)$.

### N05 — Useful rational basis derivative formulas added

**Addition / replacement of an overbroad claim.**
[Curve lines 82–84](<docs/A Briefing on NURBS Curves and Surfaces/NURBS_curves.tex#L82>)
say there is no compact and useful expression for derivatives of the
rational basis. Markdown supplies

$$
R_{i,p}^{(k)}
=\frac1w\left[
w_iN_{i,p}^{(k)}
-\sum_{j=1}^k\binom kj w^{(j)}R_{i,p}^{(k-j)}
\right].
$$

This follows directly by differentiating $wR_{i,p}=w_iN_{i,p}$.
The original statements about **bivariate** rational basis derivatives
at [surface lines 88–90](<docs/A Briefing on NURBS Curves and Surfaces/NURBS_surfaces.tex#L88>)
are omitted. Markdown gives surface-coordinate derivative formulas,
but does not separately write a bivariate rational-basis recurrence.

### Other NURBS deviations

| ID | Classification | LaTeX location | Markdown result |
| --- | --- | --- | --- |
| N06 | Correction / condensation | [Curve lines 63–68](<docs/A Briefing on NURBS Curves and Surfaces/NURBS_curves.tex#L63>) | The degree in $R_{i,n}$ becomes $R_{i,p}$. The full partition-of-unity equation is shown; its separate active-span version is implicit in the support statement. |
| N07 | Qualification | [Curve lines 58, 144](<docs/A Briefing on NURBS Curves and Surfaces/NURBS_curves.tex#L58>) and [surface lines 64, 136](<docs/A Briefing on NURBS Curves and Surfaces/NURBS_surfaces.tex#L64>) | Positivity of the weights and denominator is explicit for nonnegativity and convex-hull claims. Equal weights are specified as nonzero in the curve special-case table. |
| N08 | Addition | No explicit counterpart | The weight sensitivity $\partial\mathbf C/\partial w_i=N_{i,p}(\mathbf P_i-\mathbf C)/w$, its shape-control interpretation, common weight-scaling invariance, and possible poles for mixed-sign weights are new. |
| N09 | Qualification / omission | [Curve line 80](<docs/A Briefing on NURBS Curves and Surfaces/NURBS_curves.tex#L80>), [surface line 86](<docs/A Briefing on NURBS Curves and Surfaces/NURBS_surfaces.tex#L86>) | Unconditional unique-maximum statements are removed; no general extrema theorem is substituted. |
| N10 | Equivalent / omission | [Curve lines 133–139](<docs/A Briefing on NURBS Curves and Surfaces/NURBS_curves.tex#L133>), [surface lines 125–131](<docs/A Briefing on NURBS Curves and Surfaces/NURBS_surfaces.tex#L125>) | Affine invariance is retained, but its step-by-step proofs are omitted. The Bézier page contains a representative affine identity. |
| N11 | Equivalent / addition | [Curve lines 211–227](<docs/A Briefing on NURBS Curves and Surfaces/NURBS_curves.tex#L211>) | The first and general curve quotient-derivative formulas are preserved, with fewer derivation steps. An explicit second derivative, evaluation order, and the nonvanishing of rational derivatives above polynomial degree are added. |
| N12 | Equivalent / addition | [Surface lines 199–215](<docs/A Briefing on NURBS Curves and Surfaces/NURBS_surfaces.tex#L199>) | The generic first-partial formula becomes explicit $u$ and $v$ formulas; $uu$, $uv$, and $vv$ cases are added as expansions of the existing recurrence. |
| N13 | Correction / omission | [Curve lines 160–164](<docs/A Briefing on NURBS Curves and Surfaces/NURBS_curves.tex#L160>), [surface lines 152–154](<docs/A Briefing on NURBS Curves and Surfaces/NURBS_surfaces.tex#L152>) | The curve variation-diminishing bullet originally refers to “B-Spline” instead of NURBS; Markdown states the NURBS property under positive weights. The separate control-polygon approximation, triangulated-net interpretation, and surface variation-diminishing bullet are omitted. |
| N14 | Addition / condensation | [Surface corner section, line 226](<docs/A Briefing on NURBS Curves and Surfaces/NURBS_surfaces.tex#L226>) | Corner interpolation is retained in prose; rational boundary curves are explicitly mentioned. The full and active-rectangle basis sums are condensed into the statement that bases sum to one. |
| N15 | Addition | No counterpart in these LaTeX files | The entire “Surface normals and curvature” section is new: unit normal, first/second fundamental forms, signed mean curvature $H$, Gaussian curvature $K$, and dependence on normal orientation. |

The NURBS curve/surface definitions, special cases, local support,
active-control-point counts, homogeneous projection concept, and
endpoint formulas retain their intended meaning after these corrections.

## Endpoint formulas: what did not change

The compact Markdown notation can make unchanged formulas look different.
For both B-spline and NURBS endpoints,

$$
a=u_{p+1},\quad b=u_{p+2},\quad
c=1-u_n,\quad d=1-u_{n-1}.
$$

The NURBS page additionally uses

$$
\alpha=w_1/w_0,\quad\beta=w_2/w_0,\quad
\gamma=w_{n-1}/w_n,\quad\delta=w_{n-2}/w_n.
$$

Expanding these abbreviations recovers the original first and second
derivatives term by term, including the rational terms proportional to
$\alpha(1-\alpha)$ and $\gamma(1-\gamma)$.

The Bézier second derivative was simply expanded:

$$
(\mathbf P_2-\mathbf P_0)-2(\mathbf P_1-\mathbf P_0)
=\mathbf P_2-2\mathbf P_1+\mathbf P_0.
$$

The endpoint curvature displays reverse the order of the cross-product
factors in several places. This leaves their **norms** unchanged:

$$
\|\mathbf a\times\mathbf b\|=\|\mathbf b\times\mathbf a\|.
$$

It would reverse a **signed** planar determinant; the inherited formulas
are for unsigned curvature. The original factors
$(p-1)/p$, $u_{p+1}/u_{p+2}$, $(1-u_n)/(1-u_{n-1})$, and the endpoint
weight ratios were already present. They are not new corrections.

| Family | Original briefing | Repeated G² source | Assessment |
| --- | --- | --- | --- |
| Bézier | [Curve lines 148–173](<docs/A Briefing on Bezier Curves and Surfaces/Bezier_curves.tex#L148>) | [Lines 31–58](<docs/G2 continuity of NURBS curves/2_bezier_curvature_formulas.tex#L31>) | Equivalent after expansion and cross-product norm symmetry |
| B-spline | [Curve lines 209–241](<docs/A Briefing on B-Spline Curves and Surfaces/BSpline_curves.tex#L209>) | [Lines 34–65](<docs/G2 continuity of NURBS curves/3_bspline_curvature_formulas.tex#L34>) | Equivalent after substituting $a,b,c,d$ |
| NURBS | [Curve lines 235–266](<docs/A Briefing on NURBS Curves and Surfaces/NURBS_curves.tex#L235>) | [Lines 28–60](<docs/G2 continuity of NURBS curves/4_nurbs_curvature_formulas.tex#L28>) | Equivalent after substituting knot and weight abbreviations |

Curvature expressions using three endpoint control points require degree
at least two and nonzero endpoint tangents. Markdown makes these
restrictions more explicit than the originals.

## G² and airfoil notes

### G01 — Curvature magnitude versus geometric continuity

**Correction / qualification.**
[Motivation line 11](<docs/G2 continuity of NURBS curves/1_motivation.tex#L11>)
describes G² as continuity of radius. The airfoil section likewise says
that repeating the construction with the same radius ensures G²
([lines 18 and 34](<docs/G2 continuity of NURBS curves/5_airfoil_parametrization.tex#L18>)).

[Markdown](docs/source/theory/g2_continuity.md#position-tangency-and-curvature)
requires matching position, oriented unit tangent, and curvature vector.
For a regular curve,

$$
\mathbf T=\frac{\mathbf C'}{\|\mathbf C'\|},\qquad
\mathbf K=
\frac{\mathbf C''-\mathbf T(\mathbf T\cdot\mathbf C'')}
{\|\mathbf C'\|^2}.
$$

Equal magnitudes $\|\mathbf K_A\|=\|\mathbf K_B\|$ allow opposite bending
directions. For example, endpoint velocity $(1,0)$ and accelerations
$(0,2)$ versus $(0,-2)$ give the same tangent and unsigned curvature,
but opposite curvature vectors. Matching radius is sufficient only when
the other geometric conditions are also satisfied.

### G02 — Reparametrization conditions and signed curvature

**Additions.** The Markdown conditions

$$
\mathbf C'_B(0)=\lambda\mathbf C'_A(1),\qquad
\mathbf C''_B(0)=\lambda^2\mathbf C''_A(1)+\mu\mathbf C'_A(1),
\quad \lambda>0,
$$

are new explicit statements derived from the chain rule. They explain
why G² does not require equal parameter speeds, and distinguish it from
parametric C². Signed planar curvature, the zero-curvature case, and
the failure of the formulas at zero speed are also added.

The original scalar curvature formula is unchanged by reversing its
cross-product order inside a norm.

### G03 — Reorganization of the endpoint derivation

**Equivalent / addition / omission.** G² LaTeX sections 2–4 each repeat
the curve definition and endpoint derivatives. Markdown relocates these
to the three family pages and derives the common result once, using
$\mathbf C'(0)=a_1\mathbf A$ and
$\mathbf C''(0)=a_2\mathbf B+a_3\mathbf A$.
The explicit $a_1,a_2$ coefficients are new explanatory algebra.

The $f_0,f_1$ abbreviations contain the original endpoint factors.
Bézier and B-spline cases are recovered in prose by setting weights and
knot ratios appropriately. A separate end-at-$u=1$ derivation is not
shown, although its formula is retained.

The repeated NURBS definition at
[section 4 line 13](<docs/G2 continuity of NURBS curves/4_nurbs_curvature_formulas.tex#L13>)
reuses `i` as both a free index and a bound summation index. The
Markdown rational definition uses a different dummy index. This is
readability cleanup, not a change to a correctly scoped sum.
The nearby “B-Spline curve” label at line 24 becomes NURBS context.

### G04 — Airfoil control-point construction

**Equivalent with qualifications.**
The two original squared-distance formulas at
[lines 15 and 31](<docs/G2 continuity of NURBS curves/5_airfoil_parametrization.tex#L15>)
are already correct under their assumptions, and **already use**
$\rho=1/\kappa$.

The notation changes are:

| Original | Markdown |
| --- | --- |
| $\|\mathbf P_1-\mathbf P_0\|$ | $\ell_0$ |
| $\|\mathbf P_{n-1}-\mathbf P_n\|$ | $\ell_1$ |
| Leading-edge camber-line normal $\mathbf n$ | Unit curve tangent direction $\mathbf t$, with its camber-line interpretation stated |
| Trailing-edge direction $\mathbf n$ | $\mathbf t_{\rm back}$, pointing toward the preceding control point |
| Radius $\rho$ in each separate construction | $\rho_0$ and $\rho_1$ |
| Expanded degree/knot/weight factors | $f_0$ and $f_1$ |

Consequently,

$$
\ell_0^2=\rho_0f_0\|\mathbf t\times(\mathbf P_2-\mathbf P_0)\|,\qquad
\ell_1^2=\rho_1f_1\|\mathbf t_{\rm back}\times(\mathbf P_{n-2}-\mathbf P_n)\|
$$

are equivalent to the originals. The forward tangent at the final
endpoint is opposite to $\mathbf t_{\rm back}$; Markdown now states this.

New qualifications cover unit directions, positive finite target radii,
vanishing perpendicular offsets, matching curvature directions on both
sides, consistent traversal of the closed profile, and the distinction
between designing control points and shape-preserving knot insertion.
The original explicit pre-insertion set
$\{\mathbf P_0,\mathbf P_2,\ldots,\mathbf P_{n-2},\mathbf P_n\}$
is omitted, but the roles of the fixed points are retained.

### G05 — “Curvature” in the final comparison

**Notation clarification, not a correction to the preceding airfoil formulas.**
[Final remarks lines 7, 14, 19](<docs/G2 continuity of NURBS curves/6_final_remarks.tex#L7>)
use a literal variable named `curvature` in expressions for $AB^2$.
Markdown writes $\rho$ and explains the units:
$[\rho\,CH]=L^2$, whereas $[\kappa\,CH]=1$.

The published paper's Section 4.1, Eq. (8), also labels the multiplier
“curvature” while the surrounding discussion identifies radius as the
design variable. Reading that multiplier as radius is therefore a
dimensionally consistent interpretation, not evidence that the authors'
implementation used the reciprocal incorrectly.
[Mykhaskiv et al., p. 920](https://www.cad-journal.net/files/vol_15/CAD_15%286%29_2018_916-926.pdf).

The original final section displays the Bézier relation and both
B-spline amendments separately, with the added knot ratios in red.
Markdown condenses them into $AB^2=\rho f_0CH$, states the $f_1$ end
case, and includes NURBS weights in the same expression.
The identity $CH=AC\sin\alpha$ is replaced by the equivalent norm
of a cross product with a unit direction.

### G06 — “Only valid for Bézier” was stronger than necessary

**Qualification identified explicitly in this second pass.**
[Final remarks line 10](<docs/G2 continuity of NURBS curves/6_final_remarks.tex#L10>)
says the factor without knot correction is only valid for Bézier curves.
Markdown instead requires the general knot factor and describes the
Bézier special case.

A non-Bézier B-spline can also have that factor equal to one. For example,
with $p=2$ and
$U=[0,0,0,0.4,0.4,1,1,1]$, there are five control points, so $n=4>p$,
but $u_{p+1}/u_{p+2}=1$ and
$(1-u_n)/(1-u_{n-1})=1$. Thus the original statement is a useful
warning against applying the Bézier factor generally, but not a strict
“if and only if” characterization.

### G07 — Material added or removed from the motivation

The Markdown retains the need for smooth multi-segment geometry and the
airfoil motivation. It omits the broad assertion that G² is sufficient
“in most cases” and the claim that internal continuity is straightforward
for a single spline. Instead, the B-spline theory explains how repeated
knots affect internal smoothness.

The [curvature continuity example](docs/source/examples/curvature_continuity.md)
and its link at the end of the theory page are additions. It demonstrates
a two-segment join; it is not a reproduction of the original complete
airfoil construction.

## Illustrations and other omissions

The following omissions are part of the migration and should be reviewed
as content choices, rather than mistaken for mathematical corrections.

| Original material | Current status |
| --- | --- |
| [Curve-smoothness illustration](<docs/G2 continuity of NURBS curves/1figures/curve_smoothness_annotated.pdf>) | Not embedded in the Markdown theory |
| [Airfoil G² illustration](<docs/G2 continuity of NURBS curves/1figures/G2_continutity_blade_annotated.pdf>) | Not embedded |
| [Endpoint-curvature illustration](<docs/G2 continuity of NURBS curves/1figures/endpoint_curvature_annotated.pdf>) | Not embedded |
| [Leading-edge construction illustration](<docs/G2 continuity of NURBS curves/1figures/leading_edge_construction_annotated.pdf>) | Not embedded |
| Original figure captions and in-text figure references | Removed with the four figure placements; the files themselves remain |
| Repeated Bernstein and B-spline definitions in later surface/NURBS/G² sections | Consolidated into the preceding family/basis sections |
| Explicit convex-hull set definitions repeated for each family | One representative definition remains in the Bézier curve section |
| Separate product-basis derivative and active-partition equations | Often incorporated into a more general surface formula or explanatory prose, as itemized above |
| Affine-invariance proofs and several Leibniz derivation steps | Results retained, intermediate lines condensed |
| Surface control-net triangulation interpretation | Omitted for all three families |
| Original numbered lists of properties | Reorganized as topic paragraphs, equations, and tables |

## References, labels, and document structure

| ID | Classification | Change |
| --- | --- | --- |
| R01 | Equivalent structure | Separate curve and surface files are combined into one page per family. G² duplicate definitions are replaced by links. A shared notation/assumptions page is new. |
| R02 | Omission | The introductions cite *The NURBS Book*, Chapter 1 for Bézier, Chapters 2–3 for B-splines, and Chapter 4 for NURBS. Markdown retains the book citation but drops these chapter locators. |
| R03 | Omission | G² sections 2, 3, and 4 cite book pages 9–25, 81–100, and 117–127 respectively. These original page locators are not preserved. This audit records them as supplied; it does not certify their correspondence to every endpoint formula. |
| R04 | Addition / equivalent metadata | The book bibliography entry gains second-edition metadata and DOI `10.1007/978-3-642-59223-2`. Author, title, year, and publisher retain the original reference. The Mykhaskiv entry retains the same authors, journal, volume, issue, pages, and year, with capitalization protection. |
| R05 | Addition | Shene's derivative and NURBS-properties pages are new supplemental links. The retained Zotero library is separate from the two theory references; it was not part of the LaTeX theory bibliographies. |
| R06 | Equivalent presentation | LaTeX `\label`/`\RefEq`/`\RefFig` and citation commands become MyST labels, links, and citation roles. Old equation/section numbering is not maintained. |
| R07 | Correction / presentation | The Bézier surface definition uses the old label `eq:def_bspline_surface`, and its Bernstein labels duplicate curve labels. G² NURBS section 4 repeats `sec:bspline`. Markdown uses page-appropriate labels, eliminating those collisions/mislabels. |
| R08 | Equivalent presentation / omission | Title blocks, `\today`, forced page breaks, font/margin setup, `\resizebox`/`minipage`, and red amendment highlighting are replaced by Sphinx presentation. Roberto Agromayor's authorship is retained in the shared introduction and site metadata, rather than repeated title blocks. |

## Recommended follow-ups

These are review findings, **not additional edits made by this report**.

1. **Restore the four original illustrations if completeness is the goal.**
   They explain the airfoil construction better than the new generic
   two-curve example alone. Restoring them would require suitable
   HTML-compatible figures or linked PDF downloads and their captions.

2. **Restore the original chapter/page citation locators after checking the
   intended book edition.** This is a traceability loss, even though the
   book and article themselves remain in the bibliography.

3. **Make low-degree guards explicit at each derivative formula.**
   [bezier.md](docs/source/theory/bezier.md#bernstein-polynomials) starts at
   $n\geq0$ but displays a first-derivative formula involving degree $n-1$;
   it should explicitly say $n\geq1$, with the degree-zero derivative
   equal to zero. Similarly, the Bézier endpoint tangent formula uses
   $\mathbf P_1$, and the B-spline endpoint formula uses division by
   $u_{p+1}$, so those displays need $n\geq1$ or $p\geq1$.
   The higher B-spline basis-derivative recurrence should state $k\geq1$,
   and the derivative-control-point recurrence should distinguish its
   $k=0$ initialization from $1\leq k\leq p$. These are presentation gaps
   in the current Markdown, not newly discovered sign errors.

4. **Distinguish parameter smoothness from regular geometric shape.**
   A polynomial parametrization can be infinitely differentiable while
   having a stationary point or cusp. The common regularity assumption
   handles curvature, but a nearby reminder in the Bézier smoothness
   discussion would help readers.

5. **Clarify attribution in the airfoil comparison.**
   The knot-factor amendment was already in the original notes. The
   explicit radius interpretation, general G² vector conditions, and
   discussion of traversal were added during the migration. The closing
   sentence “This is the correction derived in the repository notes”
   could distinguish these contributions more clearly.

6. **Optionally restore the compact product identities and approximation
   interpretations.** They are mostly redundant mathematically, but the
   omissions in B13, S08–S10, and N14 may matter if the intended document
   is a comprehensive property list rather than an introductory narrative.

7. **Keep the separate scope of the added surface-curvature section clear.**
   It supplies useful context for the API, but it was not transcribed from
   these LaTeX notes. A primary reference beside that section would improve
   attribution; the
   [MIT treatment of the second fundamental form](https://web.mit.edu/hyperbook/Patrikalakis-Maekawa-Cho/node29.html)
   is one suitable source for the normal-curvature conventions.

## Verification performed for this pass

The review compared the displayed mathematical blocks and the surrounding
claims, including repeated endpoint formulas in the G² notes. It also
checked the equivalence of expanded and abbreviated expressions, the
surface recurrence index sets, the dimensional consistency of the airfoil
distance relation, and counterexamples to overbroad statements.

| Check | Result and scope |
| --- | --- |
| Existing [documentation mathematics tests](tests/test_documentation_math.py) | **30 passed** on this pass. These cover endpoint formulas for polynomial/rational curves, degrees 2, 3, 5, nonuniform/repeated interior knots, shortened-knot derivative control points against SciPy, Bézier surface differences, a rational mixed partial, radius construction at both ends, and the demonstrated G² join. |
| Independent rational curve derivatives | Orders 0–5 checked against polynomial-numerator differentiation of $A/w$, using $A_{k+1}=wA_k'-(k+1)w'A_k$ and denominator $w^{k+2}$. Passed; maximum absolute difference was approximately $3.21\times10^{-10}$ across the chosen samples, within the relative/absolute tolerances. This includes orders above the cubic numerator degree. |
| Original three surface sums versus the compact Markdown sum | Matching, nonduplicated index sets for all $0\leq k,l\leq5$, together with the algebraic partition explained in N04. |
| Same scalar curvature, different curvature vectors | Explicit planar counterexample confirmed; supports G01. |
| Non-Bézier spline with endpoint knot factor one | Explicit repeated-interior-knot example confirmed; supports G06. |
| Mixed-sign weights and convex hull | A two-control-point rational example evaluates outside its control segment; confirms the need for weight qualifications. |
| Unique maximizing parameter | A degree-zero B-spline is constant on its span, confirming the missing restriction in the original unconditional statement. |
| Markdown math-source checks | All **66 displayed math blocks** have balanced unescaped braces. The files contain no Unicode replacement characters. This checks source structure, not browser rendering or mathematical validity by itself. |
| External cross-checks | Shene's derivative notes and the published Mykhaskiv article were consulted for knot-vector interpretation and the wording of the airfoil comparison. The latter distinguishes an ambiguous variable name from a demonstrated implementation error. |

The numerical checks are corroborating evidence, not proofs of every
statement for all degrees, knots, weights, and degeneracies. They execute
independent formulas or Python checks, not the Markdown equations as code.
The main mathematical corrections are justified by the algebra and index
arguments above. A successful Sphinx build alone would not establish
equation correctness.
