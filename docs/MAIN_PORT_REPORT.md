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
