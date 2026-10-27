# Numerical benchmarks

- [Arc length](arclength/report.md): production adaptive integration, ODE length,
  optional prepared quadrature and dense ODE parameterization. The benchmark
  calls production APIs and records cold stages, warm execution and independent
  accuracy. Correctness regressions belong in `tests/`; timings do not.
- [Point projection](point_projection/report.md): production curve/surface
  sampled initialization and bounded refinement, with independent references
  and explicit numerical limitations.

Keep one documented benchmark per feature with measured data and source/
environment fingerprints. Git history preserves previous implementations.
Development code is not part of the installed runtime package.
