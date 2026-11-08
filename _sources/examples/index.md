# Examples

The numbered [checkout tutorials](https://github.com/turbo-sim/nurbspy/blob/main/demos/README.md)
progress through `00_basis_polynomials`, `01_curves`, `02_surfaces`, and
`03_viewer`. Each script explains its purpose, defines geometry, and displays
one editor per curve or surface script with a background grid. Geometry examples
use the public Three.js
`Figure` API without image/report generation. See the [Figure guide](../figures.md)
for saving PNGs or standalone read-only interactive HTML files. The separate
`docs/export_demo_figures.py` command exports both formats under each demo folder
in `output/<script_name>/`.

These pages walk through basis polynomials, curve derivatives, circular
arcs, curvature continuity, surface construction, and point projection
onto both curves and surfaces, showing the inputs alongside the printed
results and plots they produce. Their complete scripts live in
[`demos/documentation/`](https://github.com/turbo-sim/nurbspy/tree/main/demos/documentation)
and use the NumPy interface (`import nurbspy as nrb`); the wider
[`demos/`](https://github.com/turbo-sim/nurbspy/tree/main/demos)
directory also has scripts using the `nurbspy.jax` interface and a
blade parametrization example.

```{toctree}
:maxdepth: 1

basis_polynomials
curves
surfaces
circular_arc
bilinear_surface
ruled_surface
extruded_surface
revolution_surface
coons_patch
curvature_continuity
point_projection_curve
point_projection_surface
stl_printing

```
