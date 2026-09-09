# Changelog

Backfilled from the git history in September 2026, since no changelog was
kept as the project evolved. Entries summarize the significant, user-facing
changes for each release; routine dependency bumps, typo fixes, and
README-only edits are omitted.

## Unreleased

Documentation overhaul: a full Sphinx site (getting started, theory pages
migrated from the original LaTeX notes, a worked-examples gallery, API
reference), plus `docs/generate_example_images.py` to keep example images
in sync with the scripts that produce them. 3D plots got consistent tick
styling and a fully closed bounding box (Matplotlib only draws 3 of the 6
faces by default). Point projection (`project_point_to_curve` /
`project_point_to_surface`) gained optional `maxiter`/`ftol`/`gtol`
arguments and dedicated documented examples. CI now installs JAX as an
optional extra rather than a hard dependency.

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
