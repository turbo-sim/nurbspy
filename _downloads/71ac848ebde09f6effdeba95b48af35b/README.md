# Numerical illustrations for the theory pages

From the repository root, in an environment with the project's development
dependencies and JAX extra installed:

```bash
python docs/generate_theory_images.py
python docs/generate_theory_images.py --only frames surface_geometry
```

Each topic module constructs geometry with the public nurbspy APIs. Values,
derivatives, frames, sections, inverse maps and projections are calculated by
the package. Surface integration is explicitly assembled from public partial
derivatives and a tensor Gauss rule; the package has no dedicated area method.
Independent exact formulas or SciPy integration check the illustrated results.
Assertions fail the command if a numerical check fails.

`common.py` calls `nurbspy.set_plot_options()` and applies consistent portable
fonts, physical 3D proportions and a restrained color palette. Each figure is
saved under `docs/source/theory/images/` as a 300 dpi PNG and editable vector
SVG/PDF. The pages embed SVG; PNGs are convenient for inspection and PDFs for
publication. SVG/PDF dates are omitted so regeneration avoids timestamp noise.
`validation.json` records numerical results, without timing thresholds.

The scripts are figure generators, not alternative production solvers. The
independent comparisons are limited to the illustrated geometries and do not
certify general global projection or quadrature accuracy. Selected generation
updates only the requested figures and validation entries. Use `--output-dir`
for a separate rendering directory when comparing style or environments.
