# Point projection development benchmark

This directory measures the production `nurbspy.jax` point projector. Geometry
fixtures and independent references are kept here; the implementation is only
in `src/nurbspy/jax/` and its correctness tests are in `tests/backends/`.

Read [the report](report.md) for the algorithm, retained measurements and limits.
Run a short check from the repository root with an installed JAX extra:

```powershell
python development/point_projection/benchmark.py --smoke
```

Use `--case NAME` to choose fixtures, or `--all` for all retained cases. Stress
cases intentionally expose limitations of finite sampled initialization;
their inclusion does not imply every closest-point basin is guaranteed found.
The benchmark never imports a development implementation of projection.
