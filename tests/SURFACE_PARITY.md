# Surface backend parity

This inventory tracks the JAX surface implementation against the NumPy API.
Shared numerical checks belong in `geometry/`; differentiation and execution
checks belong in `backends/`.

| API | Implementation stage | Verification |
| --- | --- | --- |
| `NurbsSurface`, `P/W/p/q/U/V`, surface type and dimensions | Complete | `geometry/test_surfaces.py`: construction modes, scalar and paired-array evaluation |
| `get_value`, polynomial/rational coordinate helpers | Complete | Exact patches, independent SciPy tensor-product bases and direct helper calls |
| `get_derivative`, polynomial/rational derivative helpers | Complete | `geometry/test_surface_derivatives.py`: pure/mixed derivatives, rational orders above degree |
| `get_normals`, `get_curvature` | Complete | Planes, cylinders, spheres; `backends/test_jax_surface_autodiff.py`: exact paraboloid gradients |
| Bilinear, ruled, extruded, revolution and Coons generators | Complete | `geometry/test_surface_generators.py`: corners, rational boundaries, exact revolution geometry; both AD modes for generator inputs |
| `get_isocurve_u/v`, `attach_nurbs_udir/vdir` | Complete | `geometry/test_surface_operations.py`: values, tangents, seams and distinct knot vectors; AD through sections and attachment |
| `make_offset_surface` | Complete | Signed planar offsets and JAX offset-distance gradients |
| `project_point_to_surface`, projection problem helper | Complete | Shared interior/edge/corner checks; `backends/test_jax_surface_projection.py`: query and geometry derivatives against independent references |
| `project_points` (JAX convenience method) | Complete | Compiled query batches returning `(2,N)` parameters |
| `plot`, surface/boundary/isocurve/control-net/normal plotting, rescaling and curvature plotting | Complete | `geometry/test_surface_plotting.py`: headless scalar/2D/3D plots and constant-curvature colors |
| `U_values/U_mults/V_values/V_mults` | Complete | OpenCascade knot metadata checked against distinct knots and multiplicities |

Degrees, derivative orders, control-net sizes and iteration limits define
static structure. Control points, weights, parameter values and knot values
are array inputs. Knot derivatives are meaningful only while knot ordering
and multiplicities remain valid and evaluation avoids changes of span.

JAX-specific boundaries:

- Pass changing parameter values as arrays to reuse compiled evaluations;
  Python scalars are static under `eqx.filter_jit`, matching JAX curves.
- Revolution keeps automatic arc-count selection for concrete angles.
  Traced angles require a static `n_arcs` from one to four, with each arc
  spanning at most pi/2. This fixes topology while preserving angle gradients.
- Plotting converts geometry to NumPy on the host and reuses the existing
  plotting implementation. Unique-knot metadata has variable output size and
  is also host-only. Neither belongs inside JIT or differentiation.
- Offsets preserve the existing sampled approximation; general offsets are
  not exact NURBS representations of the parallel surface.
- Projection minimizes squared distance with a bounded multistart Newton
  solve. Implicit derivatives require a locally unique regular minimum,
  a nonsingular reduced Hessian and a stable active set. Ties, singular
  patches, knot nonsmoothness and active-set transitions have no general
  derivative guarantee. Multiple starts do not prove global optimality.
- Coons construction retains the NumPy homogeneous control-net blend and
  requires compatible clamped boundaries; it does not elevate degrees or
  insert knots to reconcile incompatible boundary spaces.
- NumPy/Numba complex-step checks remain in `backends/test_numpy_complex_step.py`.
  JAX guarantees are checked with forward/reverse automatic differentiation.

Validation on 26 October 2026: the complete suite passed with Numba and JAX
required (338 passed). With both optional packages hidden through import
blocking, the full default-backend run had 147 passed and 144 skipped; this
simulates missing dependencies rather than installing a clean environment.
A wheel built from this checkout was installed into a temporary directory
outside the repository and passed the isolated installed-package smoke check,
including a gradient through an exported JAX surface generator. The nine
full-suite warnings are existing SciPy deprecations in curve projection.
