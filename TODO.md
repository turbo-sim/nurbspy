# TODO

## Documentation examples

- [x] **Add a point-projection example.** Added
  `demos/documentation/point_projection_curve.py` and
  `point_projection_surface.py` (NumPy interface, docstring, printed
  orthogonality checks against the point-inversion optimality condition),
  with matching `docs/source/examples/point_projection_curve.md` and
  `point_projection_surface.md` pages that derive that condition before
  the walkthrough, and wired both into the examples `toctree` and
  `docs/generate_example_images.py` (auto-discovers `demos/documentation/`,
  no separate wiring needed). One target point in the curve script was
  swapped after a grid-search sanity check showed the solver landing in a
  local minimum for the original choice `(2.0, -2.0)` -- worth keeping in
  mind that `project_point_to_curve`/`project_point_to_surface` are not
  globally robust for every input.

- [ ] **Add a JAX-vs-NumPy "blade matching" example.** Worth pinning down
  first what this should actually demonstrate — most likely differentiating
  a blade-matching objective (e.g. G2 curvature matching at a segment join)
  with `nurbspy.jax`'s autodiff, contrasted with the finite-difference or
  complex-step approach available in the NumPy/Numba interface. The new
  `demos/blade_parametrization/quadratic_bezier_endpoint_radius.py` and
  `blade_profile_boundary.py` scripts (NumPy interface, G2-continuity
  theory) are a natural non-JAX half of this comparison to build on.

## README

- [x] **Fix the broken math-background links.** Fixed -- now points to
  `docs/latex_notes/<family>/...` (the files' current location) and also
  links the Markdown theory pages under `docs/source/theory/` as a
  no-PDF-viewer-needed alternative.
- [x] **Trim and de-duplicate.** Condensed the curve/surface capabilities
  lists into one shared list plus what's different per type, simplified
  the inline-image HTML, fixed two typos ("succesful", "Methods to for"),
  and added one line on the JAX backend that wasn't mentioned before.
- [x] **Point at the current examples.** Both "Check out the ... demos"
  links now point at `demos/documentation/` instead of the legacy
  `demos/demos_curves/` / `demos/demos_surfaces/`. Worth revisiting once
  the demo-folder consolidation item below is settled, in case the target
  directory changes again.

## Possibly missing from this list

- **Consolidate or retire the legacy demo folders.** `demos/demos_curves/`,
  `demos/demos_surfaces/`, and `demos/demos_basis_polynomials/` predate
  `demos/documentation/` and cover much of the same ground with rougher,
  undocumented scripts (e.g. compare `demos/demos_curves/demo_nurbs_curve.py`
  to the polished `demos/documentation/curve_comparison.py`). Worth
  deciding whether to delete them now that `demos/documentation/` + the
  Sphinx examples pages are the canonical set, or keep them as a separate
  "quick reference" tier and say so explicitly somewhere -- otherwise
  readers land on two different versions of the same example with no
  indication which one is current.
- **Commit the in-progress blade_parametrization rewrite.** The old
  `endpoint_curvature_v1/v2/v3.py` scripts are already deleted locally and
  replaced with `quadratic_bezier_endpoint_radius.py` and
  `blade_profile_boundary.py` (properly docstringed, linked to the
  G2-continuity theory page) -- but as of now that deletion and both new
  files are still uncommitted. Also worth deciding whether these two
  scripts should get their own `docs/source/examples/` page, or stay as
  standalone demos outside the Sphinx site.
- **Wire `docs/generate_example_images.py --check` into CI.** The script
  now exists and catches doc images that no longer match their generating
  script, but nothing currently runs it automatically -- a future edit to
  a `demos/documentation/*.py` script can still silently desync its image
  from the committed PNG with no warning, same as before the script
  existed.
- [x] **No CHANGELOG.** Added `CHANGELOG.md`, backfilled from the git/tag
  history. Keep it updated on future releases -- a backfilled changelog
  that isn't maintained going forward just goes stale again.
