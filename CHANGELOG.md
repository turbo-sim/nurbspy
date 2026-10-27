# Changelog

Backfilled from the git history in September 2026, since no changelog was
kept as the project evolved. Entries summarize the significant, user-facing
changes for each release; routine dependency bumps, typo fixes, and
README-only edits are omitted.

## Unreleased

Made `get_arclength()` a static JAX dispatcher over the explicit
`get_arclength_quad_adaptive()`, `get_arclength_quad_fixed()` and
`get_arclength_ode()` methods. Adaptive remains the default. Removed the
ambiguous `get_arclength_quad()` name without a compatibility alias. Options
are validated by the selected method; fixed quadrature rejects tolerances.

Made JAX curve arc length adaptive by default, using public Gauss-Kronrod
quadrature with explicit knot breakpoints, separate absolute/relative controls,
optional differentiation strategies and convergence diagnostics. Retained an
explicit knot-aware ODE alternative and the existing fixed Clenshaw-Curtis API.
Added optional prepared spanwise Gauss tables for repeated differentiable
evaluations with changing control points and positive weights.

Retained dense reciprocal-ODE arc-length parameterization, with independent
knot-span solves, exact endpoint anchors and sampled regularity/monotonicity
validation. Queries remain interpolation-only and preserve scalar/array shape;
sampled zero-speed/flat curves and invalid queries are rejected explicitly.
Added focused accuracy, interval, failure and AD regressions. Consolidated the
standalone production-API benchmark under `development/arclength/`.

Changed JAX curve and surface point projection to knot-aware sampled
initialization followed by exactly one bounded local solve per query. General
and prepared surface projections now share this single-start implementation.
The curve's legacy `tol` is honored, and sample/iteration budgets are
configurable. Safeguards use spatially scaled stationarity, stable Hessian
regularization, feasible descent steps, explicit failure checks and exact
implicit derivatives with moving bounds. Added regressions for conditioned
rational curves, knot corners, general parameter domains and sensitivities.
The standalone development benchmark calls production APIs; obsolete projection
prototype implementations were removed. Finite sampling does not guarantee a
global closest point on arbitrary geometry.

Updated the optional JAX backend to JAX/JAXlib 0.12, Equinox 0.13,
Diffrax 0.7.2, Optimistix 0.1 and Quadax 0.3. This development version now
requires Python 3.12 or 3.13. CI checks the JAX backend at both Python
boundaries, and the documentation workflow uses Python 3.12.

Adapted the fixed Clenshaw-Curtis quadrature call to Quadax's polynomial-order
API, preserving the default 41-node rule. Other positive node-count requests
round upwards to a supported $4k+1$-node rule (at least five nodes). Removed
the obsolete SciPy L-BFGS-B display option. These dependency compatibility
changes preserve the existing ODE and fixed-quadrature algorithms.

## [1.5.0] - 2026-10-26

Reduced JAX tracing and compilation overhead by vectorizing the B-spline basis
recurrence across basis indices and samples, sharing lower-degree tables across
derivative orders, and contracting curve derivative rows together. Curve and
surface callers request only the polynomial derivative rows they need. Existing
JAX curve/surface APIs, dynamic knots and parameters, dtype conventions and
safe inactive zero-denominator branches are retained. Keeping complete
lower-degree support also fixes incorrect nonclamped basis derivatives.

Added explicit `nurbspy.jax.prepare_curve` and `prepare_surface` APIs for repeated
evaluation at fixed knots and sample locations, with changing control points
and rational weights. Surface grids use factorized one-dimensional tables.
Prepared objects support JIT, batching, forward/reverse AD and rational
derivatives above the polynomial degree. Preparation remains opt-in because it
has setup/storage costs and tables must be rebuilt when their defining inputs
change.

Added `prepare_surface_projection` for repeated query batches on a fixed
surface. It reuses the coarse initialization grid while retaining the existing
dynamic Newton refinement, bounds, tolerances and implicit derivatives. General
and prepared projections share the solver implementation. Documented the
different geometry/basis lifetimes, valid differentiation paths and numerical
limitations in the usage guide and API reference.

Carried over the numerical and differentiation regression tests from the
compilation investigation, including independent SciPy/exact references,
repeated knots/endpoints, nonclamped support, dtype compatibility, paired/grid
evaluation and prepared projection. No benchmark scripts/results or new runtime
dependencies were added. The default NumPy/Numba backend is unchanged.

## [1.4.1] - 2026-10-26

Fixed the JAX extrusion docstring's formula formatting, which Sphinx treated
as an undefined substitution and rejected during documentation deployment.
Regenerated the API pages to include all JAX surface modules and verified
the complete documentation build from freshly generated pages.

## [1.4.0] - 2026-10-26

Separated the implementations into sibling `nurbspy.numpy` and `nurbspy.jax`
subpackages, with shared graphics and package information at the root.
Existing `import nurbspy` usage still selects NumPy; legacy module imports
such as `nurbspy.nurbs_curve` and previously pickled NumPy curves and surfaces
remain supported through explicit module aliases. New pickles use the
`nurbspy.numpy` paths and are not guaranteed to load in older releases.
Added import-identity, legacy-serialization, and installed-wheel checks.

Updated generated NumPy API page paths and cross-references to the canonical
modules so Sphinx continues to include their classes and functions after the
move. Existing Python import paths remain supported.

Moved the importable package into `src/nurbspy` and updated packaging,
release versioning, documentation paths, and installation checks. Local
development now uses an editable installation; tests exercise the installed
package without adding the repository root to the import path.

Added JAX NURBS surfaces with values, arbitrary mixed partial derivatives,
normals, mean/Gaussian curvature, isocurves, attachment, approximate offsets,
and bounded point projection. Added bilinear, ruled, extruded, revolution,
and Coons generators, plus host-side plotting and knot metadata. Revolution
accepts an explicit static `n_arcs` when differentiating traced angles.
Projection supports forward/reverse implicit derivatives at locally unique,
regular minima with stable active bounds; ambiguous or singular minima and
active-set transitions are outside that differentiability guarantee.

Extended shared geometry tests to both surface backends and added JAX
surface differentiation, generator, operation, and projection checks against
independent references. Fixed NumPy v-direction attachment using the wrong
knot vector, revolution misclassifying radial vectors with zero component
sum, incomplete Coons corner validation, scalar inputs to a surface derivative
helper, obsolete Matplotlib tick access, and zero-range curvature colors.

Reorganized tests by mathematical feature, backend execution, installation,
and documentation. Strengthened numerical assertions and added public
projection, surface-operation, torsion, and complex-step sensitivity checks.
Fixed JAX rational curve derivatives above the polynomial degree.

Added dedicated JAX tests for transformations, automatic differentiation,
shape gradients, arc length, inverse parametrizations, and batched projection.
Fixed NaN reverse-mode gradients at repeated knots and in fixed quadrature
for constant-speed curves, plus scalar inputs to the JAX basis-derivative
functions. The test runner now discovers the full suite.

Numba is now an optional extra (`pip install "nurbspy[numba]"`). The default
NumPy backend and the JAX extra no longer require Numba or inherit its NumPy
version restrictions. Basis functions use NumPy when Numba is unavailable,
and warn before falling back if an installed Numba raises an import error.

CI now checks default, Numba, and JAX installations separately on Ubuntu
and Windows, plus a built wheel with both optional extras. Installed-package
checks run outside the source checkout, and PyPI publishing is restricted
to version-tag pushes after the checks pass.

Added optimization examples for curve fitting with point inversion and
minimum bending energy, with guidance on differentiating projected-distance
objectives. Added a JOSS paper draft. These additions followed the v1.3.2 tag.

## [1.3.2] - 2026-09-09

Refreshed locked dependencies and corrected package-version metadata before
tagging the release.

## [1.3.1] - 2026-09-09

Updated the Black and pytest development dependencies and refreshed the
dependency lockfile.

## [1.3.0] - 2026-09-09

Documentation overhaul: a full Sphinx site (getting started, theory pages
migrated from the original LaTeX notes, a worked-examples gallery, API
reference), plus `docs/generate_example_images.py` to keep example images
in sync with the scripts that produce them. 3D plots got consistent tick
styling and a fully closed bounding box (Matplotlib only draws 3 of the 6
faces by default). Point projection (`project_point_to_curve` /
`project_point_to_surface`) gained optional `maxiter`/`ftol`/`gtol`
arguments and dedicated documented examples. CI now installs JAX as an
optional extra rather than a hard dependency.

Removed a deprecated SciPy solver option from surface projection to avoid
deprecation warnings.

Migrated release versioning from bump2version to bump-my-version, with
configuration to update the package and project versions together.

## [1.2.4] - 2025-11-06

Reworked the NURBS basis-function derivatives to use explicit loops instead
of vectorized NumPy operations, trading some raw speed for JAX
autodiff-compatibility.

## [1.2.3] - 2025-11-06

Finished making `NurbsCurve` fully compatible with JAX transformations
(`jit`, `grad`, `vmap`).

## [1.2.0] – [1.2.2] - 2025-10-29

Introduced the optional `nurbspy.jax` backend, giving curves automatic
differentiation and JIT compilation via JAX. Scaffolded the Sphinx
documentation site and its GitHub Pages deploy workflow.

## [1.1.3] – [1.1.5] - 2025-03-31

Modernized the project after roughly five years dormant: migrated from
`setup.py` to Poetry, dropped support for old Python versions, and updated
dependencies for compatibility with newer NumPy/Numba/SciPy releases. No
functional changes to the NURBS classes.

## [1.1.2] - 2020-06-15

Added OpenCascade-format (OCC) knot values and multiplicities to
`NurbsCurve` and `NurbsSurface`. Last release before the project went
dormant until 2025.

## Earlier history (2020)

The initial implementation: `NurbsCurve` and `NurbsSurface` with Bézier,
B-Spline, and NURBS construction, evaluation, and analytical derivatives;
the first PyPI release; and point projection (point inversion) for both
curves and surfaces, added shortly after as the first major feature beyond
the initial release. These versions predate the `v1.1.2` tag and were not
individually tagged in git.
