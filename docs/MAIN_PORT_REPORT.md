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
