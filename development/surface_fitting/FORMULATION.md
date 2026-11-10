# Surface-fitting formulations and point correspondences

These are the current five demos. Every nonlinear surface-fitting optimization
uses Pounce interior point with an exact sparse Hessian. Point projection reuses
the existing NurbsPy batched JAX bounded-Newton implementation.
All cases use 4, 8, 16 and 32 controls per direction, with independent flat starts.

Cases 01-04 call the public `nrb.fit_surface` API. Their reusable algorithms,
variable layout, exact sparse derivatives and correspondence strategies live in
[src/nurbspy/fitting](../../src/nurbspy/fitting/). This folder retains the analytic
target, explicit bounds and penalties, refinement runs, numerical archives and
reports. Case 05 retains its custom constrained Pounce problem and full
multiplier-weighted Lagrangian Hessian. The shorter single-fit demonstrations
are in [demos/surface_fitting](../../demos/surface_fitting/README.md).

| Demo | Point parameters | Surface unknowns | Solution method |
| --- | --- | --- | --- |
| 01 | Known and fixed | All coordinates, unit weights fixed | Direct linear least squares |
| 02 | Known and fixed | All coordinates and weights | Exact sparse-Hessian IPM |
| 03 | Unknown, jointly optimized | All coordinates and weights | Joint exact sparse-Hessian IPM with recentered local parameter windows |
| 04 | Unknown, updated by projection | All coordinates and weights | Alternating exact sparse-Hessian surface IPM and existing NurbsPy projection |
| 05 | Known and fixed | Heights and interior weights | Constrained IPM with exact sparse Lagrangian Hessian and sparse constraint Jacobian |

Cases 03 and 04 use the same objective, bounds and regularization. Their solution
strategies differ. The generating parameter labels never enter either fitting
objective. Initial parameter estimates come from measured x/y coordinates,
which is appropriate for this graph-like example, not a general scan initializer.

## Common geometry, data and notation

There are $n$ control points in each parameter direction, so $K=n^2$ control
points in total. All surfaces are cubic in both directions, with open uniform
knot vectors. Let $N_i(u)$ and $M_j(v)$ denote their B-spline basis functions.
The rational surface is

$$
\mathbf S(u,v;\mathbf P,\mathbf w)
=\frac{\displaystyle\sum_{i=0}^{n-1}\sum_{j=0}^{n-1}
N_i(u)M_j(v)w_{ij}\mathbf P_{ij}}
{\displaystyle\sum_{i=0}^{n-1}\sum_{j=0}^{n-1}
N_i(u)M_j(v)w_{ij}}
=\sum_{i,j}R_{ij}(u,v;\mathbf w)\mathbf P_{ij}.
$$

Here $\mathbf P_{ij}\in\mathbb R^3$ and $w_{ij}>0$. With unit weights,
partition of unity makes the denominator one.

The analytic target is a graph,

$$
\mathbf T(u,v)=
\begin{bmatrix}u\\v\\z(u,v)\end{bmatrix},
\qquad
z(u,v)=\sum_{\ell=1}^{6}a_\ell
\exp\left[-\frac12\left(
\frac{(u-c_{\ell,u})^2}{s_{\ell,u}^2}
+\frac{(v-c_{\ell,v})^2}{s_{\ell,v}^2}
\right)\right].
$$

Its parameters are:

| $\ell$ | $a_\ell$ | $c_{\ell,u}$ | $c_{\ell,v}$ | $s_{\ell,u}$ | $s_{\ell,v}$ |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.48 | 0.23 | 0.25 | 0.12 | 0.14 |
| 2 | 0.42 | 0.75 | 0.67 | 0.14 | 0.12 |
| 3 | 0.34 | 0.25 | 0.82 | 0.12 | 0.11 |
| 4 | -0.38 | 0.55 | 0.30 | 0.13 | 0.13 |
| 5 | -0.32 | 0.50 | 0.78 | 0.11 | 0.12 |
| 6 | -0.28 | 0.85 | 0.27 | 0.10 | 0.13 |

Sample $m$ equally spaced parameter values per direction, including the
boundaries. This gives $N=m^2$ observations. Write the generating labels as
$\bar{\boldsymbol\xi}_k=(\bar u_k,\bar v_k)$ and the observed positions as

$$
\mathbf q_k=\mathbf T(\bar u_k,\bar v_k)+\boldsymbol\varepsilon_k,
\qquad k=1,\ldots,N.
$$

The defaults are $m=45$, $N=2025$ and zero noise. Optional noise is independent
Gaussian noise in all three position coordinates, with standard deviation
$\sigma$ and seed 2407. It does not change the generating parameter labels.
Demos 01, 02 and 05 retain those labels; demos 03 and 04 use only the observed
positions in their fitting objectives.

The displayed initial surface has unit weights and control coordinates
$\mathbf P^0_{ij}=(g_i,g_j,0)$, where $g_i$ are the Greville abscissae. It
represents $\mathbf S_0(u,v)=(u,v,0)$. Each nonlinear refinement starts from this
flat surface independently, without using a coarser fitted surface or a
B-spline fit as its initial guess.

The default grids are 4, 8, 16 and 32 controls per direction. All five demos use the same four grids. Refinement changes the control net and knots between fits; the optimizer
does not move the knots within a fit.

## What point projection would mean

For a fixed surface, closest-point projection of an observation would solve

$$
\boldsymbol\xi_k^*(\boldsymbol\theta)
\in\operatorname*{arg\,min}_{(u,v)\in[0,1]^2}
\frac12\left\|\mathbf S(u,v;\boldsymbol\theta)-\mathbf q_k\right\|^2,
$$

where $\boldsymbol\theta$ collects the surface variables. A geometric fitting
objective can then minimize these distances over the surface:

$$
\min_{\boldsymbol\theta}\frac{1}{2N}\sum_{k=1}^{N}
\min_{(u,v)\in[0,1]^2}
\left\|\mathbf S(u,v;\boldsymbol\theta)-\mathbf q_k\right\|^2.
$$

Demos 01, 02 and 05 do not solve the inner projection problems. They evaluate
the surface at the fixed labels $\bar{\boldsymbol\xi}_k$. Their residuals are
3D correspondence errors and need not be normal to the fitted surface.

Demo 03 introduces all point parameters as joint unknowns. Demo 04 solves the
same unknown-correspondence problem by alternating projection and surface fitting.

## Demo 01: linear B-spline fitting

All weights stay one. Only the $3K$ control coordinates are unknown:

$$
\min_{\mathbf P}\ f_1(\mathbf P)
=\frac{1}{2N}\sum_{k=1}^{N}
\left\|\sum_{i,j}N_i(\bar u_k)M_j(\bar v_k)\mathbf P_{ij}
-\mathbf q_k\right\|^2.
$$

Let $B_{k,(i,j)}=N_i(\bar u_k)M_j(\bar v_k)$, let $C$ contain the control
coordinates as a $K\times3$ matrix, and let $Q$ contain the observations. Then

$$
\min_C\frac{1}{2N}\|BC-Q\|_F^2.
$$

This is linear least squares. On the Cartesian observation grid, each
coordinate can also be written as $Q_d\approx A_u C_d A_v^T$. The implementation
uses two one-dimensional NumPy least-squares solves, giving

$$
C_d=A_u^+Q_d(A_v^+)^T,\qquad d\in\{x,y,z\},
$$

where $A_u^+$ and $A_v^+$ are the least-squares pseudoinverses. The code does not
explicitly form normal equations or run an iterative nonlinear optimizer.

There are no coordinate bounds, regularization penalties or point projections.
The flat initial geometry is provided for comparison in the viewer; a direct
linear solve does not require an iterative starting guess.

Implementation: [public-fit study adapter](_experiment.py) and the
[shared linear solver](../../src/nurbspy/fitting/_linear.py).

## Demo 02: nonlinear rational fitting at fixed point parameters

Both the control coordinates and weights are optimized. Positive weights use
$w_{ij}=\exp(\alpha_{ij})$. Fix $\alpha_{00}=0$, hence $w_{00}=1$, and eliminate
that variable. This removes the common weight-scale ambiguity, since multiplying
all weights by the same positive constant does not change the surface.

The variable vector has $3K+(K-1)=4K-1$ entries. The problem is

$$
\begin{aligned}
\min_{\mathbf P,\boldsymbol\alpha}\quad
f_2(\mathbf P,\boldsymbol\alpha)
&=\frac{\gamma}{2N}\sum_{k=1}^{N}
\left\|\mathbf S(\bar u_k,\bar v_k;
\mathbf P,\exp(\boldsymbol\alpha))-\mathbf q_k\right\|^2,\\
\text{subject to}\quad
-\log(10)&\leq\alpha_{ij}\leq\log(10),
\qquad (i,j)\ne(0,0),\\
\alpha_{00}&=0,
\qquad \gamma=1000.
\end{aligned}
$$

All three control coordinates are free and unbounded, including their $x$ and
$y$ coordinates. The weight bounds are $0.1\leq w_{ij}\leq10$. There is no
fairness or weight regularization term. The positive objective scaling does not
change the mathematical minimizers, but it affects the numerical scale of
gradients and stopping criteria.

The labels $(\bar u_k,\bar v_k)$ never change. An iteration moves the control
net and changes the rational weights, then compares the new surface at those
same labels with the fixed observed positions. It does not find a nearest
surface point for each observation. In particular, this is a 3D correspondence
fit, not merely a least-squares fit of the height coordinate.

Local polynomial basis values at the fixed parameter pairs are precomputed and
reused. JAX differentiates the scalar objective in double precision. If
$\mathbf r_k=\mathbf S(\bar u_k,\bar v_k;\boldsymbol\theta)-\mathbf q_k$
and $J_k=\partial\mathbf r_k/\partial\boldsymbol\theta$, its full Hessian is

$$
\nabla^2 f_2
=\frac{\gamma}{N}\sum_{k=1}^{N}
\left[
J_k^TJ_k+
\sum_{d=1}^{3}r_{k,d}\nabla^2 S_d(\bar u_k,\bar v_k;\boldsymbol\theta)
\right].
$$

The residual-dependent second term is included. The implementation does not
replace the Hessian by the Gauss-Newton approximation $J^TJ$.

Cubic local support makes this Hessian sparse. Structural coloring recovers
its lower-triangle entries from exact AD Hessian-vector products. PySolverView
passes these entries to Pounce's interior-point solver; it does not construct
a dense Hessian or a full residual Jacobian during the solve.

Pounce solves primal-dual Newton systems, with its own globalization,
regularization and bound-barrier contributions. For intuition, a bound-barrier
subproblem has the form

$$
f_2(\boldsymbol\theta)
-\mu\sum_{(i,j)\ne(0,0)}
\left[
\log(\alpha_{ij}-\alpha_{\min})
+\log(\alpha_{\max}-\alpha_{ij})
\right],
\qquad \mu>0.
$$

The user-supplied Hessian is the objective Hessian; Pounce owns the bound
terms. Defaults are an adaptive barrier, native tolerance $10^{-8}$ and at most
1000 iterations. Separate PySolverView KKT checks and KKT-based stopping are
disabled. Native convergence is a local numerical stopping result, not proof
of a global minimum or smallest reference-grid error.

Implementation: [terrain settings and public-fit adapter](_nonlinear.py), with
[shared exact sparse derivatives and solver](../../src/nurbspy/fitting/_nonlinear.py).

## Demo 03: all unknowns optimized jointly

Write $\boldsymbol\theta=(\mathbf P,\boldsymbol\alpha)$, with
$\alpha_{00}=0$ and $w_{ij}=\exp(\alpha_{ij})$. For each observation introduce
$\boldsymbol\xi_k=(u_k,v_k)$. The total number of unknowns is

$$
n_{\mathrm{var},3}=4K-1+2N.
$$

The objective, before the common positive scaling $\gamma=1000$, is

$$
\begin{aligned}
F(\boldsymbol\theta,\boldsymbol\xi)
={}&\frac{1}{2N}\sum_{k=1}^N
\|\mathbf S(u_k,v_k;\boldsymbol\theta)-\mathbf q_k\|^2\\
&+\frac{\lambda_{xy}}{4K}\sum_{i,j}
\|\mathbf P_{ij,xy}-\mathbf P^0_{ij,xy}\|^2
+\frac{\lambda_w}{2(K-1)}\sum_{(i,j)\ne(0,0)}\alpha_{ij}^2,
\\
&\lambda_{xy}=10^{-4},\qquad\lambda_w=10^{-6}.
\end{aligned}
$$

The code minimizes $\gamma F$ subject to

$$
\begin{gathered}
-0.25\leq P_{ij,x},P_{ij,y}\leq1.25,\qquad
-0.6\leq P_{ij,z}\leq0.6,\\
-\log(10)\leq\alpha_{ij}\leq\log(10),\qquad
\alpha_{00}=0,\qquad 0\leq u_k,v_k\leq1.
\end{gathered}
$$

The normalizations follow the implemented residuals: $2K$ planar coordinate
penalties and $K-1$ log-weight penalties. These weak terms stabilize redundant
parameterizations. All planar coordinates remain optimization variables.
Initial point parameters are measured x/y clipped to
$[10^{-10},1-10^{-10}]^2$. For the zero-noise graph these happen to agree with
the generating labels initially, but the labels are not supplied to the fit.

The joint Hessian contains the full control/weight block, mixed surface/point
blocks and independent two-by-two point blocks:

$$
H_3=\nabla^2(\gamma F)=
\begin{bmatrix}
H_{\theta\theta}&H_{\theta\xi}\\
H_{\xi\theta}&\operatorname{blkdiag}(H_{\xi_1\xi_1},\ldots,H_{\xi_N\xi_N})
\end{bmatrix}.
$$

All residual-dependent second derivatives and mixed terms are included. JAX
forms the sparse blocks; there is no Gauss-Newton or limited-memory replacement.

Changing point parameters can change their active B-spline supports. To avoid
including every possible control/point pair in a single enormous fixed pattern,
each IPM pass uses a local box $\Omega_k^{(t)}\subseteq[0,1]^2$ covering up to two
adjacent knot spans per direction around the current point. The candidate
support is at most five controls per direction. The declared Hessian pattern
contains every nonzero possible throughout that entire box, not merely the
nonzeros at its current point.

Each pass solves jointly for all surface and point variables, with
$\boldsymbol\xi_k\in\Omega_k^{(t)}$. If artificial window boundaries prevent
stationarity on the true domain, the windows are recentered and another joint
IPM pass starts from the returned full state. No projection or surface-only
optimization step is inserted.

The global stationarity check uses the native bound multipliers, retaining only
those associated with genuine parameter-domain boundaries at zero or one:

$$
\left\|\nabla_{\xi}(\gamma F)
-\mathbf z_{L,\mathrm{true}}+\mathbf z_{U,\mathrm{true}}\right\|_\infty
\leq10^{-8}.
$$

A local-window native success is not reported as global convergence unless this
check passes. Inner IPM tolerance is $10^{-9}$ and its barrier floor is
$\mu_{\min}=10^{-14}$, so artificial-window barrier forces can decay below the
global stationarity tolerance. The default budget permits eight window passes
and 1000 iterations per pass. A budget exit remains unsuccessful.
This is local nonlinear fitting; neither stationary correspondences nor exact
Hessians certify globally nearest points or a global optimum.

Implementation: [terrain joint-fit adapter](_joint.py), with
[shared joint derivative kernels](../../src/nurbspy/fitting/_joint.py) and
[correspondence orchestration](../../src/nurbspy/fitting/_api.py).

## Demo 04: alternating projection and surface optimization

This case minimizes exactly the same $\gamma F$ as case 03, with the same
geometric bounds and regularization. The unknown count remains
$4K-1+2N$, but the variables are split between two optimization blocks.

Start with the flat surface and x/y-based parameter estimates. One cycle is

$$
\boldsymbol\theta^{t+1}\approx
\operatorname*{arg\,min}_{\boldsymbol\theta\in\mathcal B_\theta}
\gamma F(\boldsymbol\theta,\boldsymbol\xi^t),
$$

followed by

$$
\boldsymbol\xi_k^{t+1}\approx
\operatorname*{arg\,min}_{(u,v)\in[0,1]^2}
\frac12\|\mathbf S(u,v;\boldsymbol\theta^{t+1})-\mathbf q_k\|^2.
$$

Here $\mathcal B_\theta$ denotes the control and weight bounds listed for case
03. The first block holds all point parameters fixed and optimizes $4K-1$
surface variables. The second holds the surface fixed and optimizes $2N$ point
parameters. Its regularization terms are constant and need not be evaluated.

The surface block uses Pounce IPM with an exact sparse Hessian. Projection calls
the public NurbsPy JAX prepare_surface_projection/project_points API. It uses
bounded Newton steps and the exact two-by-two Hessian for each independent
unscaled point objective. Let
$D_k=[\mathbf S_u\ \mathbf S_v]$ and
$\mathbf r_k=\mathbf S(u_k,v_k)-\mathbf q_k$. The two-by-two block is

$$
H_{\xi_k\xi_k}=D_k^TD_k+\sum_{d=1}^3r_{k,d}\nabla^2_{(u,v)}S_d.
$$

Different observations do not couple. The library batches the independent
two-variable solves rather than assembling a global Hessian. Basis supports can
change freely over the full parameter domain. Projection is a local solve;
sampled-surface nearest candidates initialize its bounded-Newton refinement.
The adapter augments those candidates with valid surface points at the preceding
fitted parameters. It contains no duplicate projection algorithm or derivatives.

Compiled fitting and projection kernels are reused as correspondences and
geometry change. Prepared surface samples are rebuilt after each geometry
change, so they cannot become stale. A projection update is recorded as a
separate phase; the surface itself does not move during that phase.

Convergence requires a successful native fit and checked library projection,
plus stationarity of both blocks
at the same final pair $(\boldsymbol\theta^{t+1},\boldsymbol\xi^{t+1})$. The
surface gradient is rechecked after projection, using the preceding fit's native
bound multipliers. Both block residuals must be at most $10^{-8}$. Defaults
allow 100 cycles, 1000 native fitting iterations per cycle and 200 Newton
iterations per projected point. The projector uses gradient tolerance $10^{-9}$
and step/objective tolerances $10^{-12}$. Finishing a cycle or converging an inner
fit alone is not sufficient for overall success.

Alternating updates can converge much more slowly than a joint Newton solve.
A budget-limited result is saved and explicitly labeled Not converged; tolerances
are not relaxed to manufacture convergence.

Implementation: [terrain alternating adapter](_alternating.py),
[shared correspondence orchestration](../../src/nurbspy/fitting/_api.py),
[existing-projector adapter](../../src/nurbspy/fitting/_projection.py) and
[shared fixed-parameter loss](../../src/nurbspy/fitting/_nonlinear.py).

## Demo 05: fixed-parameter fitting with area and orientation constraints

Fix every planar control coordinate to its initial Greville position. Fix all
boundary weights to one. Optimize all $K$ control heights $z_{ij}$ and the
$(n-2)^2$ interior log weights:

$$
\boldsymbol\theta_5=
(\{z_{ij}\}_{\mathrm{all}},\{\alpha_{ij}\}_{\mathrm{interior}}),
\qquad n_{\mathrm{var},5}=K+(n-2)^2.
$$

The bounds are

$$
-0.6\leq z_{ij}\leq0.6,\qquad
\log(0.5)\leq\alpha_{ij}\leq\log(2)
\quad\text{for interior weights}.
$$

Boundary heights remain free. Fixing the planar control net and boundary
weights preserves the unit-square XY outline; it does not prescribe the
target's 3D edge curves. Interior rational weights can change the interior
$x(u,v)$ and $y(u,v)$ even though planar control coordinates are fixed.

The data objective retains the supplied point labels:

$$
\min_{\boldsymbol\theta_5}\quad
f_5(\boldsymbol\theta_5)
=\frac{\gamma}{2N}\sum_{k=1}^{N}
\|\mathbf S(\bar u_k,\bar v_k;\boldsymbol\theta_5)-\mathbf q_k\|^2.
$$

There is no point projection and no demo-03 regularization term.

The physical surface-area constraint is based on

$$
A(\mathbf S)=
\int_0^1\int_0^1
\|\mathbf S_u(u,v)\times\mathbf S_v(u,v)\|\,du\,dv,
\qquad
A_{\max}=1+0.85(A_{\mathrm{terrain}}-1).
$$

For the analytic graph,

$$
A_{\mathrm{terrain}}=
\int_0^1\int_0^1\sqrt{1+z_u(u,v)^2+z_v(u,v)^2}\,du\,dv.
$$

The code approximates this reference integral independently with
$100\times100$ Gauss-Legendre quadrature, giving approximately
$A_{\mathrm{terrain}}=1.86741317022$ and $A_{\max}=1.73730119469$.
The cap deliberately excludes the exact analytic target, while the flat
surface has area one and is feasible.

Optimization uses a knot-span Gaussian quadrature approximation $A_Q$ to the
fitted area. It also restricts the signed planar Jacobian

$$
J_{xy}(u,v)=x_u y_v-x_v y_u
$$

on a uniform $(2n+1)\times(2n+1)$ guard grid. In the implemented nonpositive
constraint convention, the nonlinear inequalities are

$$
g_A(\boldsymbol\theta_5)=\frac{A_Q(\mathbf S)}{A_{\max}}-1\leq0,
\qquad
g_{\ell}(\boldsymbol\theta_5)=0.05-J_{xy}(u_\ell,v_\ell)\leq0.
$$

The area constraint is normalized by its cap. The orientation constraint is
sampled; it does not certify nonfolding everywhere between guard points.

The objective scaling is $\gamma=1000$. Native Pounce IPM receives the
nonpositive constraint vector $\mathbf g$ directly, without an outer penalty or
augmented-Lagrangian loop. Its constraint Jacobian is sparse: the area row can
involve all variables, while each orientation row involves only nearby interior
weights.

For objective factor $\sigma$ and native constraint multipliers
$\boldsymbol\lambda$, the supplied exact Lagrangian Hessian is

$$
H_L=\sigma\nabla^2 f_5+\sum_\ell\lambda_\ell\nabla^2g_\ell.
$$

JAX differentiates this weighted scalar expression and recovers its sparse
lower triangle. It does not store a separate Hessian for each constraint and
does not build a dense constraint Jacobian. The area is a sum of local
quadrature contributions, so its Hessian remains local even though its gradient
has global support. Pounce owns all barrier terms, Newton/KKT solves and
globalization.

Defaults use adaptive-barrier IPM, native tolerance $10^{-8}$ and at most 1000
iterations. Strict native success is required. Optimization area quadrature uses
$\max(6,\lceil20/(n-3)\rceil)$ points per knot span per direction. The final audit
uses $\max(8,\lceil24/(n-3)\rceil)$ and an orientation grid of
$(4n+1)\times(4n+1)$. The result is rejected if audited area exceeds
$A_{\max}(1+2\times10^{-5})$ or minimum audited orientation is below
$0.05-10^{-6}$. These remain finite quadrature/sampling checks, not a global
nonfolding certificate.

Implementation: [ConstrainedSurfaceProblem and constrained](_constrained.py),
with analytic sparse first derivatives in [constrained_problem](_advanced.py).

## Errors, convergence and timings

For cases 01, 02 and 05, training RMS is

$$
E_{\mathrm{train}}=\sqrt{\frac1N\sum_k
\|\mathbf S(\bar u_k,\bar v_k)-\mathbf q_k\|^2}.
$$

Their reference RMS compares fitted and analytic surfaces at the same labels on
a separate $193\times193$ grid. Neither metric uses projection. For case 02,
$E_{\mathrm{train}}=\sqrt{2f_2/1000}$.

Cases 03 and 04 report training RMS at their optimized point parameters,
excluding regularization penalties. Reference RMS uses the existing NurbsPy
local projector on an independent $65\times65$ analytic sample grid. Initializer
candidates include library surface samples and fitted parameters from nearest
observed XYZ points, preserving the learned parameterization without using
generating labels. The
report's B-spline baseline uses the same geometric reference metric. The
baseline surface is a separate known-label linear fit used only for comparison;
it is never an initializer for the unknown-parameter methods. The fixed-label
$193\times193$ error is retained as parametric_reference_rms, and the fixed-label
baseline is retained separately as bspline_reference_rms. Projection stopping
status is also saved; local distances are not certified global nearest distances.

Each grid's setup includes optional imports, basis/sparsity construction,
compilation and kernel warmup. Warm solve excludes geometry reconstruction,
final audits and report generation. Case 03 sums joint IPM passes; case 04 sums
surface IPM solves and library projection and also reports them separately. Native
iteration/evaluation counts describe fitting only; the public projection API
does not expose Newton iteration counts. Reference
projections are postprocessing and excluded from these solve times. Compilation
is not included in warm time. Report animation samples real stored observations
and endpoints, preserving actual iteration and cycle numbers without interpolation.

Separate PySolverView KKT diagnostics/stopping remain disabled. Native IPM
checks, the artificial-window check in 03, block checks in 04 and finer geometry
audits in 05 determine their respective convergence labels. A budget exit is
not success. All nonlinear fits are local and can have different training and
reference errors, even when their native stopping criteria are satisfied.

Source navigation: [public fitting API](../../src/nurbspy/fitting/_api.py),
[target and linear-fit study adapter](_experiment.py),
[known-parameter adapter](_nonlinear.py), [joint adapter](_joint.py),
[alternating adapter](_alternating.py), [projection compatibility import](_projection.py),
[constrained solver](_constrained.py), [archives and metrics](_study.py),
and [running all cases](README.md).
