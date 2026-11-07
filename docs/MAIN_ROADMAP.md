# nurbspy roadmap for main

Prepared 7 November 2026. This is a plan for selective changes, not approval to merge the experimental branch wholesale. No item is implemented on main merely because a version exists on `feature/interactive-plotting`.

The aim is to improve the existing numerical library, make geometry construction easier, support dependable mesh export, and ultimately provide an optional desktop editor with direct control-point manipulation. The numerical backend remains Python with NumPy and optional JAX.

## Working agreement

- Implement one numbered item at a time when requested, in the order below unless dependencies justify a change.
- Make one focused commit per item, including its relevant tests, examples, and documentation. If an item proves too broad, agree a subdivision before implementing it; do not quietly combine unrelated changes.
- For this batch (items 1–6 and demo migration), the user explicitly requested implementation and separate commits directly on main. Extract and review relevant changes from the experimental branch; do not cherry-pick its mixed commits wholesale.
- Before each item, check the current repository state and preserve unrelated work, including the untracked oloid demo. Do not switch branches across conflicting user edits.
- Run checks appropriate to the item. Record the actual results and any limitations in the completion report, then mark the item complete here with its commit hash.
- Commit on the branch requested by the user. Direct commits on main are authorized for this batch; pushing remains a separate action and has not been requested.
- Treat the paths below as current reference locations, not an obligation to retain an experimental module structure.

## Design decisions to preserve

- Keep existing constructor names, arguments, coordinate-first array conventions, and numerical return shapes.
- Accept optional `Point` objects through `control_points`; do not introduce a separate `from_points()` method.
- Numeric arrays and numeric lists/tuples retain the existing coordinate-first interpretation. Never infer orientation from shape or give lists and arrays different layout semantics.
- Convert convenience inputs to backend arrays at construction. Numerical kernels operate on arrays, not collections of Point objects.
- Preserve existing arbitrary-dimensional and complex-coordinate numerical use cases. Display and Point convenience restrictions must not restrict the core array API.
- Keep optional dependencies optional, and import them only when their functionality is requested.
- Use the shared visualization core and serverless pywebview desktop. The optional Dash adapter embeds in a host-owned application; numerical imports and desktop usage do not start a server. This later user-approved design supersedes the original Qt/PyVista proposal.
- Prefer established mesh processing and file writers over maintaining a general-purpose mesh repair/export implementation.

## Sequence at a glance

| Item | Change | Source | Depends on |
| --- | --- | --- | --- |
| 1 | Repair shared graphics helpers | Selective port | None |
| 2 | Make figure and axes handling consistent | Selective port | 1 |
| 3 | Correct diagnostic plots | Selective port | 2 |
| 4 | Share NumPy/JAX plotting implementation | Selective port | 2, 3 |
| 5 | Add Point input and curve normalization | New implementation | None |
| 6 | Add Point grids for surfaces and geometric constructors | New implementation | 5 |
| 7 | Add shared YAML geometry save/load | Extract and redesign | 5, 6 |
| 8 | Provide a small surface-to-triangle adapter | Extract and redesign | None |
| 9 | Delegate mesh processing and STL export | Replace experimental approach | 8 |
| 10 | Provide plotted torus and oloid STL examples | Adapt experimental demo | 9 |
| 11 | Establish efficient evaluation for interactive edits | New implementation | 5, 6 |
| 12 | Build the optional interactive desktop interface | New implementation | 7, 8, 9, 11 |

The order goes from small repairs to larger API additions and interaction work. Later items are not required in order to release earlier improvements.

## 1. Repair shared graphics helpers

**Status:** Complete. **Commit:** `36bb6ce`. **Validation:** 15 graphics helper tests passed. **Suggested commit:** `Fix graphics helpers and optional image export`.

**Purpose:** Remove small, verified defects without changing the plotting API.

**What and how:**

- Selectively port changes from `src/nurbspy/graphics.py`.
- Correct the imageio availability flag and provide accurate errors when optional image/video dependencies are missing.
- Support the documented color-palette names and validate custom palettes before changing plotting settings.
- Reuse the decorative 3D box artist on repeated calls, recreate it after axes clearing, and prevent it from changing data limits.

**Acceptance:** Graphics imports without imageio; palette behavior and errors are predictable; repeated box updates do not accumulate artists or alter limits. Port the corresponding focused regression tests.

## 2. Make figure and axes handling consistent

**Status:** Complete. **Commit:** `fb97008`. **Validation:** 28 graphics and plot-handle tests passed. **Suggested commit:** `Standardize curve and surface plotting handles`.

**Purpose:** Make plots compose reliably inside existing figures, subplots, and applications.

**What and how:**

- Review the handle-resolution changes in `numpy/nurbs_curve.py` and `numpy/nurbs_surface.py`.
- Move shared plotting-handle resolution to an appropriate graphics utility rather than making surfaces depend on a private curve helper.
- Handle supplied figures, supplied axes, and incompatible projections consistently. Reject mismatched figure/axes pairs clearly.
- Apply visibility and layout options to the intended axes and figure; replace obsolete Matplotlib tick access.
- Preserve existing signatures, returned handles, and defaults except where a documented defect is being corrected.

**Acceptance:** Cover figure-only, axes-only, explicit pairs, projection mismatch, multiple figures, and visibility options. Existing representative plotting demos continue to run headlessly.

## 3. Correct diagnostic plots

**Status:** Complete. **Commit:** `ca6fa49`. **Validation:** 51 graphics and diagnostic-plot tests passed across NumPy and JAX. **Suggested commit:** `Fix curvature and differential-geometry plots`.

**Purpose:** Ensure plots of geometric quantities are meaningful and stable on ordinary and degenerate cases.

**What and how:**

- Port the relevant curvature colorbar, constant-field normalization, and rectangular sampling-grid fixes.
- Correct planar Frenet-frame handling, normal-vector scaling, and constant torsion/aspect behavior.
- Handle undefined zero-length vectors explicitly without producing invalid plotting coordinates. Do not silently alter the underlying differential-geometry definitions.

**Acceptance:** Exercise planar curves, constant curvature, zero curvature, non-square surface grids, and normal/frame plots. Check numerical values and artist configuration rather than relying only on screenshots.

## 4. Share NumPy/JAX plotting implementation

**Status:** Complete. **Commit:** `cde7258`. **Validation:** 60 plotting and installation checks passed. **Suggested commit:** `Share host plotting between NumPy and JAX backends`.

**Purpose:** Remove duplicated visualization code so fixes apply to both backends.

**What and how:**

- Adapt the experimental JAX curve-to-NumPy plotting delegation and review consistency with the existing JAX surface plotting path.
- Keep host conversion at the visualization boundary. Preserve signatures, styling arguments, and return values.
- Document plotting as a host-side operation outside JIT and differentiation. Numerical evaluation remains on its selected backend.
- Remove only the duplicated implementations actually replaced by the shared path.

**Acceptance:** Representative NumPy and JAX plots have equivalent geometry and options. Plotting leaves the original JAX model unchanged. Numerical gradient and import-compatibility checks remain successful.

## 5. Add Point input and curve normalization

**Status:** Complete. **Commit:** `d3c0e97`. **Validation:** 98 curve, Point, derivative, and installation checks passed, including JAX JIT/grad and complex-step. **Suggested commit:** `Accept Point sequences in curve constructors`.

**Purpose:** Let users construct curves naturally without remembering a matrix transpose.

**What and how:**

- Add a lightweight immutable `Point(x, y)` / `Point(x, y, z)` value with coordinate access and clear dimension errors. Avoid an extensive vector algebra API in this change.
- Extend the existing `control_points` argument to recognize lists or tuples of Point objects. For curves, convert the sequence to internal shape `(dimension, number_of_points)`.
- Keep numeric input interpretation unchanged. Normalize numeric lists/tuples as coordinate-first inputs, matching the documented ndarray convention.
- Use a small input-normalization layer shared in behavior across NumPy and JAX. Preserve complex dtypes, backend arrays, and traced numeric values; never route JAX tracers through NumPy or Python `float`.
- Reject mixed Point/numeric structures and inconsistent Point dimensions with useful errors. Keep weights separate from Point.
- Points are values copied into geometry, not live shared handles: reusing one Point does not couple different curves.

**Proposed usage:**

```python
curve = NurbsCurve(
    control_points=[Point(0, 0), Point(1, 2), Point(3, 2), Point(4, 0)],
    degree=3,
)
```

**Acceptance:** Point input matches equivalent array input for polynomial Bezier, rational Bezier, B-spline, and NURBS curves. Existing numerical construction and NumPy/JAX differentiation behavior remain intact. Include explicit square-array ambiguity regression cases.

## 6. Add Point grids for surfaces and geometric constructors

**Status:** Complete. **Commit:** `192a7c1`. **Validation:** 156 construction checks passed; the final 82 Point checks also cover integer-coordinate/fractional-weight precision. **Suggested commit:** `Accept Point grids in surface construction`.

**Purpose:** Extend the readable construction convention consistently to surfaces and applicable geometric helpers.

**What and how:**

- Accept a rectangular grid of Point objects in the existing surface `control_points` argument.
- Define `points[i][j]` as internal `P[:, i, j]`: the outer index advances in $u$, the inner index in $v$. Weight grids use the same `(i, j)` ordering.
- Validate rectangularity, uniform coordinate dimensions, and weight-grid shape without restricting legacy numeric dimensions or complex coordinates.
- Adapt constructors accepting individual geometric points, such as bilinear patches and circular arcs, where appropriate. Preserve the meaning and ordering of their current arguments; do not interpret direction vectors as points implicitly.
- Document equivalent Point-grid and ndarray examples. Shared Point values do not create automatic continuity constraints between patches.

**Acceptance:** Use unequal grid dimensions to expose accidental transposes. Verify equivalent geometry, derivatives, and weights for Point and numerical input across both backends and applicable helpers.

## 7. Add shared YAML geometry save/load

**Status:** Complete. **Commit:** `0e1c3d4`. **Checks before commit:** 154 passed. **Validation:** round trips, malformed inputs, independent arrays, cross-backend loading, and JAX differentiation covered by new tests.

**Purpose:** Save editable geometry and support undo/reset without tying model state to a particular GUI.

**What and how:**

- Extract the useful serialization idea from `viewer.py`, not its Dash state schema or viewer classes.
- Provide an explicit snapshot/restore boundary for curves and surfaces containing control points, weights, degrees, and knots. Choose the smallest public API consistent with the existing package when implementing this item.
- Keep in-memory copies independent and backend-aware. Restoring creates a new valid model rather than partially mutating a model with stale derived data.
- Define a versioned YAML representation, dimension validation, and clear errors for unsupported versions. Complex arrays retain their dtype and explicit real/imaginary data.
- Use safe YAML loading; serialize geometry only, not arbitrary Python objects or backend runtime state. A collection of patches can be a list of geometry records; a new assembly model is not required.

**Acceptance:** Round-trip coordinates, weights, knots, and evaluations; ensure snapshots do not alias their originals. Verify invalid records, backend restoration, and the documented complex-coordinate behavior.

## 8. Provide a small surface-to-triangle adapter

**Status:** Complete. **Commit:** `9ef9bfe`. **Checks before commit:** 188 passed, covering knot-aware sampling, convergence, degenerate boundaries, NumPy/JAX connectivity, surface regressions, YAML, and installation.

**Purpose:** Supply common sampled geometry to desktop rendering and external mesh tools.

**What and how:**

- Retain the NURBS-specific portion of experimental `mesh.py`: evaluate the actual knot domains, include knot breaks, and construct triangle connectivity.
- Return plain host arrays `(vertices, faces)`. Use existing surface evaluation for NumPy and JAX models, with explicit host conversion at this boundary.
- Review diagonal selection and degenerate parametric-edge behavior against more than the oloid example. Reuse rendering-library triangulation where it reduces custom logic without losing knot-aware geometry.
- Keep sampling, mesh processing, and file writing separate. Do not carry over a general-purpose welding/orientation/repair engine as an assumed mature feature.
- State that count-based sampling gives an approximation, not a geometric error bound. Adaptive, tolerance-controlled tessellation is deferred unless separately scoped.

**Acceptance:** Check knot-domain coverage, planar and curved patches, non-square grids, singular boundaries, backend agreement, and increasing-resolution convergence on a known surface. Invalid dimensions and non-finite coordinates fail clearly.

## 9. Delegate mesh processing and STL export

**Status:** Complete. **Commit:** see `git log --grep="Add optional mesh-library-backed STL export"`. **Validation:** full suite 738 passed, 6 skipped; after the final normal-precision guard, 121 focused I/O/installation checks passed.

**Purpose:** Offer useful export while reducing the burden of maintaining mesh algorithms and file-format code.

**What and how:**

- Keep a small nurbspy-facing export wrapper around the adapter from item 8.
- Selected standard Trimesh plus NetworkX for headless mesh processing and STL writing. NumPy and JAX surfaces share `export()`; `export_stl()` handles patch collections. No viewer dependency is added.
- Delegate supported vertex merging, face cleanup, winding operations, and STL writing. Make repair policy explicit and avoid silent hole filling or other changes to the intended geometry.
- Retain the relevant experimental tests as acceptance cases, rather than assuming an external library automatically meets every requirement.
- Distinguish exporting an open surface from requesting a closed solid. Explain seam sampling limitations, welding tolerance, unitless STL coordinates, and float32 precision limits.
- Define hollow/nested-shell behavior explicitly. Do not orient every closed component outward and claim that this represents cavities correctly.
- Do not equate watertightness with freedom from self-intersections or general printability. If unsupported geometry cannot be validated, report that limitation clearly.

**Acceptance:** Independently reload exported files; check sampled geometry and topology on representative open patches, closed multi-patch solids, and problematic seams. Exercise scale/precision cases, missing optional dependencies, and either supported or explicitly rejected nested shells. Mesh export must work without any GUI or web server.

## 10. Provide plotted torus and oloid STL examples

**Status:** Complete. **Commit:** see `git log --grep="Add illustrated torus and oloid STL printing example"`. **Validation:** 36 example/export tests passed; preview, print, and fine files reloaded as closed volumes; plots rendered and inspected.

**Purpose:** Demonstrate a practical construction-to-export workflow using native nurbspy geometry.

**What and how:**

- Adapt the two ruled patches from the experimental oloid demo into a concise script using the existing geometry classes and export wrapper.
- Preserve or reconcile the user's existing untracked ruled-surface example; do not overwrite it blindly.
- Expose only useful radius/resolution/output options. Keep the script usable without a GUI and without Dash.
- Explain that its sampled degree-one circle boundaries approximate an oloid; do not present the current construction as an exact analytic NURBS representation.
- Keep example geometry generation reusable by the eventual desktop demo without moving an oloid-specific constructor into the public API unnecessarily.

**Acceptance:** Verify the generating-circle relationship, ruling lengths, closed exported topology for the supported sampling scheme, and resolution convergence. A documented command produces an STL using the repository environment.

## 11. Establish efficient evaluation for interactive edits

**Status:** Planned. **Suggested commit:** `Add reusable evaluation state for interactive geometry edits`.

**Purpose:** Make dragging responsive while preserving the numerical library's correctness.

**What and how:**

- First benchmark existing curve/surface evaluation at representative preview sizes. Record results without promising a universal frame rate.
- Introduce a small editor-side evaluator that can reuse basis values when knots, degrees, and sampling parameters are unchanged. Keep it outside core numerical kernels unless reuse clearly warrants a shared utility.
- Control-point and weight edits recompute the rational geometry from current arrays. Invalidate cached data when topology, knots, degrees, or sampling changes; never store a stale weighted denominator.
- Use a cheaper preview during manipulation and a finer evaluation after release. Coalesce rapid events instead of queuing obsolete updates.
- Do not impose display-only restrictions such as positive real weights on the general numerical constructors. Apply and document those restrictions at the editor boundary.

**Acceptance:** Cached and uncached results agree after point and weight changes. Invalidation works after relevant structural changes. Measure responsiveness for representative models and identify practical limits. Numerical differentiation paths remain unaffected.

## 12. Shared interactive viewer and editor

**Status:** Ported on main, validation recorded in `MAIN_PORT_REPORT.md`.
The later user-approved scope replaces the initial Qt/PyVista proposal with a
shared Three.js viewport, serverless pywebview desktop, and optional host-owned
Dash embedding. Undo/redo is deferred; native save/load remains in Python.

- `show()` inspects; `edit()` returns a new NumPy host geometry on Apply.
- NumPy/JAX classes delegate to the same `visualization/` package. Point inputs,
  YAML, numerical routines, and existing static plotting signatures are preserved.
- One spatial renderer supports perspective and XY/YZ/ZX projections, shared
  display/edit panes, coordinate/weight editing, constraints, and ViewCube.
- Main's Trimesh/NetworkX processing/writer handles STL, including downloads.
- Optional extras are lazy; importing numerical geometry does not import a GUI.
- Current behaviour is specified in `VIEWER_EDITOR_BEHAVIOR.md`.

## Deliberately outside this roadmap

- Merging the experimental branch wholesale, starting a hidden Dash server from desktop/numerical calls, or replacing main's external mesh writer.
- Replacing the Python NURBS numerical backend with VTK's geometry algorithms.
- A general CAD kernel, trimmed-surface mesher, Boolean solid modeller, or automatic multi-patch continuity solver.
- Guaranteeing self-intersection-free or printable solids from arbitrary surface inputs.
- An extensive mutable Point object hierarchy or implicit live links between shared control points.
- A claim that the experimental tests establish complete maturity or constitute an audit of every numerical routine.

## Evidence and reference material

- Experimental branch: `feature/interactive-plotting`. Existing commits mix concerns and are reference material for selective ports.
- Shared graphics/plotting tests: `test_graphics_helpers.py`, `test_legacy_plotting.py`, and `test_jax_plotting.py` under `tests/geometry/`.
- Experimental mesh/example tests: `test_mesh_export.py` and `test_oloid_demo.py`. Adapt these to the selected external backend and independently verify exports.
- Existing viewer tests document prototype behavior; only backend-independent cases should survive a desktop redesign.
- [PyVista movable point widgets](https://docs.pyvista.org/api/plotting/_autosummary/pyvista.plotter.add_sphere_widget) and [PyVistaQt desktop integration](https://qt.pyvista.org/).
- [PyVista mesh/STL writing](https://docs.pyvista.org/api/core/_autosummary/pyvista.polydata.save) and [Trimesh processing and validation](https://trimesh.org/trimesh.html#trimesh.Trimesh.process).

The dependency candidates above were checked against their official documentation during the assessment. Recheck compatibility with nurbspy's supported Python, NumPy, and operating-system versions before adding them.

## Completed demo migration (7 November 2026)

- Items 1–6 are committed directly on main, one commit per item. Items 7–12 remain planned.
- Updated explicit geometry definitions in all 41 curve/surface/optimization demos to Point sequences, Point grids, or individual Point locations. The three basis-only demos have no geometric Point inputs and remain unchanged.
- Numerical optimization and derivative calculations still use backend arrays, obtained from the constructed geometry where appropriate. Direction vectors stay numerical.
- Preserved and migrated the existing oloid demo; this adds no STL exporter or viewer to main.
- Before/after headless execution: all 44 demos succeeded; 513 recorded geometry arrays and sampled evaluations agreed within numerical tolerance; all 58 rendered figures were pixel-identical. Runtime measurements are naturally variable.
- Full regression suite with Numba and JAX required: 636 passed, 6 skipped.
- The package-level minimal example also uses Point inputs and runs headlessly.
- See [the executive report](MAIN_PORT_REPORT.md) for the scope and verification summary. No push was performed.


## YAML, tessellation, and STL batch (7 November 2026)

- Item 7: `to_yaml()` / `from_yaml()` shared across NumPy/JAX curves and surfaces,
  with versioned safe YAML and explicit complex-array/dtype preservation.
- Item 8: `surface.tessellate()` and `tessellate_surfaces()` return host arrays,
  include knot breaks, and keep patch processing separate from export.
- Item 9: `surface.export("file.stl")` and `export_stl(patches, "file.stl")`
  delegate mesh processing/writing to standard Trimesh/NetworkX dependencies.
- YAML was committed after 154 checks passed; tessellation after 188 checks passed.
- Export full regression: 738 passed, 6 skipped with Numba and JAX required.
  The final small-scale normal guard was additionally covered by 121 focused
  I/O/installation checks. One existing JAX mixed-dtype scatter warning remains.
- Lock validation and final wheel build succeeded. No push was requested.
- Items 10, 11, and 12 are not implemented by this batch.


## Illustrated STL printing examples (7 November 2026)

Item 10 now includes one standalone torus/oloid script with native plotting,
reloaded-STL comparison panels, three quality presets, and a full parameter guide.
The torus uses exact rational geometry; the sampled oloid explicitly separates
boundary approximation from export sampling. All six preset/object exports were
verified as closed volumes. Generated files live under ignored
`output/stl_printing/`; the illustrated guide includes a committed comparison PNG.
Item 11's full basis-table reuse remains future work; item 12's reviewed viewer/editor is now ported. The viewer already caches sampling/topology and reuses render buffers.
