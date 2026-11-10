# Executive report: nurbspy main improvements

7 November 2026. Implemented directly on main as requested; the experimental browser branch remains separate.

## Delivered

- Fixed graphics helpers, figure/axes handling, curvature plots, and frame/normal visualization, including undefined directions.
- Consolidated JAX curve plotting through the NumPy host implementation while retaining numerical differentiation on the selected backend.
- Added immutable 2D/3D Point values through the existing control_points argument. Curves accept sequences; surfaces accept rectangular grids indexed [u][v]. Numeric inputs retain their coordinate-first convention.
- Added Point support for circular-arc centers, bilinear corners, and revolution-axis locations. Points are copied into numerical control arrays; weights remain separate. Integer Point literals preserve fractional weights.
- Migrated all 41 geometry-bearing demos and the package minimal example. Numerical algorithms continue using arrays; the three basis-only demos need no change.

## Commits

| Roadmap item | Change | Commit |
| --- | --- | --- |
| 1 | Fix graphics helpers and optional image export | `36bb6ce` |
| 2 | Standardize curve and surface plotting handles | `fb97008` |
| 3 | Fix curvature and differential-geometry plots | `ca6fa49` |
| 4 | Share host plotting between NumPy and JAX backends | `cde7258` |
| 5 | Accept Point sequences in curve constructors | `d3c0e97` |
| 6 | Accept Point grids in surface construction | `192a7c1` |

The demo migration and this report are a separate final commit; locate it with git log --grep="Migrate demos to Point construction".

## Verification

- Full test suite, with both Numba and JAX required: **636 passed, 6 skipped**.
- All **44 demos** ran before and after migration in isolated headless processes.
- All **513 geometry records** (control arrays, weights, knots, sampled values, and derivatives) matched within numerical tolerance.
- All **58 rendered figures** were pixel-identical before and after the demo migration. Comparison baseline was captured after plotting repairs, so intentional earlier bug fixes are not claimed to preserve defective plots.
- The package minimal example ran headlessly. NumPy complex-step and JAX JIT/gradient behavior are covered by the regression checks.

## Scope

No browser server, desktop GUI, mesh exporter, or new runtime dependency was added. Roadmap items 7–12 remain pending. STL and desktop interaction will be handled in their own later steps. Tests and demo checks ran on the current Windows/Python 3.13 environment; this is not a claim of verification on every supported platform.

Changes are committed locally on main and have not been pushed.


## Subsequent geometry I/O batch

Implemented sequentially on main, verifying each item before its commit:

- **7: YAML save/load** (`0e1c3d4`): both backends and both geometry classes share
  `to_yaml()` and classmethod `from_yaml()`. Coordinates, weights, degrees, full
  knots, numeric dtypes, and complex arrays survive safe versioned YAML exchange.
  Restoring chooses the calling class's backend and creates independent arrays.
  Validation before commit: **154 passed**.
- **8: Surface tessellation** (`9ef9bfe`): a small backend-independent adapter
  returns NumPy vertices/triangles using actual knot domains and knot breaks.
  NumPy/JAX surfaces share `tessellate()`. No mesh dependency is needed for
  sampling. Validation before commit: **188 passed**.
- **9: STL export** (locate with `git log --grep="Add optional mesh-library-backed STL export"`):
  surfaces share `export("file.stl")`; root `export_stl()` supports patch collections.
  Trimesh/NetworkX support delegates mesh processing and binary writing.
  Tests cover independent binary parsing/reloading, closed cubes and an oloid,
  open patches, unmatched seams, non-manifold edges, float32 limits, and missing
  dependencies. Validation included a full **738 passed / 6 skipped** regression
  run with Numba and JAX required. A final tiny-scale normal guard was followed
  by **121 passing** focused I/O/installation checks. The final export-only
  run with Numba and JAX required also passed all **25 cases**.

The final wheel builds; an isolated-process smoke test of YAML, tessellation,
and STL directly from that wheel passed. Its dependency metadata includes the mesh dependencies. Poetry reports existing metadata-deprecation warnings. The regression
suite reports a JAX mixed float32/float64 scatter FutureWarning; this work does
not change the numerical basis implementation to address it.

STL remains sampled and unitless. Closed export supports one connected shell;
nested cavity shells and disconnected solids are explicitly rejected in that
mode. No holes are filled, and self-intersections are not validated. Unresolvable
float32 geometry or zero mesh-library normals fails before writing. YAML stores
mathematical geometry, not convenience Bezier/B-spline labels or arbitrary Python
state. Items 10?12 and all viewer work remain outside this batch. No push occurred.


### Dependency policy update

Trimesh and NetworkX are now standard dependencies, installed with nurbspy.
The separate mesh extra is removed. Imports remain lazy, so ordinary geometry
and native tessellation do not initialize the mesh libraries. Export tests now
require these dependencies rather than skipping when they are absent.

Validation of the dependency-policy update: 43 export/installation tests passed;
lockfile validation and wheel build passed; wheel metadata confirms unconditional
Trimesh/NetworkX requirements and no mesh extra.


## Illustrated 3D-printing example

Added `demos/02_surfaces/15_stl_printing.py`, constructing an exact NURBS
torus and two sampled ruled oloid patches. It exports solid STLs, reloads them
for validation, plots native geometry beside actual STL triangles, and saves a
comparison PNG. Preview/print/fine presets separate oloid boundary density from
mesh sampling. The walkthrough explains all sampling/export controls, seam
rounding, units, convergence, and slicer responsibilities.

All six preset/object exports reloaded as closed positive volumes. The focused
example/export suite passed 36 checks, including analytic torus geometry,
periodicity, volume convergence, oloid contact geometry, and the headless CLI
producing both STLs and a plot. A pre-existing JAX curve docstring indentation
was corrected after strict documentation validation identified it.

A clean strict Sphinx HTML build passed after the documentation correction.


## Shared viewer/editor port to main ? 7 November 2026

The reviewed prototype is now integrated on main in focused local commits:

- `a1dd97b`: in-memory STL bytes through the existing external mesh pipeline.
- `72f9d82`: shared desktop/embedded viewer/editor, optional dependencies, plotting
  extraction, Point/backend/YAML compatibility checks, and frontend tests.
- The documentation/example commit adds seven Point-based entry points, the
  visualization guide, and `VIEWER_EDITOR_BEHAVIOR.md`.

### Resulting behaviour

`show()` opens the new shared geometry viewer; `edit()` returns an independent
NumPy geometry on Apply or None on cancellation. The desktop uses a direct
pywebview bridge and starts no HTTP server. `GeometryView`/`GeometryEditor`
provide the same renderer inside a host-owned Dash application. Both NumPy/JAX
classes have thin wrappers; accepted JAX edits are host NumPy objects and can be
saved/restored through either backend's YAML API.

The interface includes solid defaults, horizontal Display/Edit inspectors,
Navigation dropdown, ViewCube, XY/YZ/ZX projections, ticked planar axes, grids
behind geometry, matched point sizes, XYZ fields, weight editing, and normal/
axis/plane constraints. Python retains all NURBS and diagnostic calculations.
Undo/redo and viewer file dialogs are intentionally omitted as agreed.

Main's Point constructors, serialization, and numerical method bodies/class
bases were preserved. Plotting functions were extracted from main's current
implementations into `visualization/plotting.py`; the four numerical classes
keep their existing Matplotlib signatures and public help text. Examples use
`show()`/`edit()` as the primary interactive geometry experience.

STL previews/downloads reuse main's Trimesh/NetworkX processing and writer.
The custom experimental exporter, Plotly application, and legacy assembly
viewer were not ported. `export_stl_bytes()` supports downloads without temporary
files; existing file exports retain their validation and Path return semantics.
The lockfile was regenerated from main. Desktop/web dependencies are optional;
vendor licenses are included and hashes match LF-normalized committed assets.

### Validation

- Full test tree with JAX and Numba explicitly required: **876 passed, 6 skipped**.
  The six skips are the NumPy-specific complex-step checks under the JAX fixture.
- Shared frontend suites: **70 checks passed**.
- Main Point/backend/YAML integration and existing Point/YAML checks: **142 passed**.
- External STL bytes/file export and tessellation: **52 passed**.
- All seven demos passed code-only geometry/settings or embedded layout/asset checks.
- Strict headless Sphinx HTML build passed after fixing its two new download links.
- Main's wheel builds. An isolated wheel smoke check exercised Point construction,
  native bridge acceptance/cancel-free return, NumPy/JAX YAML exchange, static
  plotting, Dash asset serving, self-contained desktop HTML, and external STL.
- Vendored checksums and packaged UI assets were checked against committed source.
- One pre-existing JAX float32/float64 scatter FutureWarning remains in a YAML test.

No computer control or live GUI session was used during this port, as requested.
Prior Windows prototype interaction checks remain useful evidence, but this is
not a claim of new cross-platform interactive verification. Changes are committed
locally on main and have not been pushed.
