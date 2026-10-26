# Changelog

Backfilled from the git history in September 2026, since no changelog was
kept as the project evolved. Entries summarize the significant, user-facing
changes for each release; routine dependency bumps, typo fixes, and
README-only edits are omitted.

## Unreleased

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
