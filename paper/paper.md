---
title: 'nurbspy: A differentiable Python library for NURBS curves and surfaces'
tags:
  - Python
  - geometry
  - NURBS
  - computer-aided design
  - shape optimization
  - automatic differentiation
authors:
  - name: Roberto Agromayor
    orcid: 0000-0000-0000-0000 # TODO: fill in your ORCID iD
    affiliation: 1
    corresponding: true
  - name: Lars O. Nord
    orcid: 0000-0000-0000-0000 # TODO: fill in Lars' ORCID iD, or remove this author if not appropriate
    affiliation: 1
affiliations:
  - name: Department of Energy and Process Engineering, Norwegian University of Science and Technology (NTNU), Norway
    index: 1
date: TODO # e.g. 09 September 2026
bibliography: paper.bib
---

# Summary

`nurbspy` is a lightweight Python library for constructing, evaluating, and
differentiating Non-Uniform Rational B-Spline (NURBS) curves and surfaces.
NURBS are the standard mathematical representation for free-form geometry in
computer-aided design (CAD) and computer-aided engineering (CAE) systems, and
`nurbspy` implements the core algorithms for their construction and
evaluation as described in the standard reference text on the subject
[@piegl_nurbs_1997]. Curves and surfaces are represented as rational or
non-rational Bézier, B-Spline, or NURBS geometries built directly from
control points, weights, and knot vectors. Both curves and surfaces support
evaluation of coordinates and arbitrary-order analytic derivatives,
curvature (and, for surfaces, mean and Gaussian curvature), point projection
and point inversion, and Matplotlib-based visualization. Curves additionally
provide the Frenet-Serret frame (tangent, normal, and binormal vectors),
torsion, and arc length by numerical quadrature; surfaces additionally
provide unit normal vectors, isoparametric curves, and dedicated
constructors for common special surfaces (bilinear, ruled, extruded,
revolution, and Coons). All core algorithms, including the Cox-de Boor basis
function recursion and its analytic derivatives, are implemented directly in
vectorized NumPy [@harris_array_2020] and accelerated with Numba's
just-in-time compilation [@lam_numba_2015], rather than wrapping an existing
geometry kernel. An optional JAX [@bradbury_jax_2018] backend
(`nurbspy.jax`) mirrors this implementation, making `nurbspy` composable
with JAX-based automatic differentiation, JIT compilation, and the
surrounding ecosystem of differentiable optimization and modeling
libraries.

# Statement of need

Gradient-based shape optimization is widely used to design free-form
components in engineering, from turbomachinery blades to aircraft wings
[@martins_engineering_2021]. In this setting, a NURBS geometry is
parameterized by its control points, and the derivatives of geometric
quantities (surface coordinates, normals, curvature) with respect to those
control points are needed to compute the optimization objective's gradient
efficiently and accurately. Existing open-source and commercial NURBS
libraries and CAD kernels, such as OpenCascade, are designed as general
geometry kernels and do not expose their internal geometric evaluation to
automatic differentiation, and computing shape derivatives through these
tools therefore typically falls back to finite differences, which trade off
truncation and round-off error and can be inaccurate or expensive when many
design variables are involved [@martins_complex-step_2003].

`nurbspy` addresses this need with an optional JAX [@bradbury_jax_2018]
backend (`nurbspy.jax`) that mirrors the full NumPy implementation and
exposes NURBS curve and surface evaluation to JAX's automatic
differentiation transforms and JIT compilation. This makes `nurbspy`
directly composable with the surrounding ecosystem of JAX-based
differentiable optimization and modeling libraries, including Equinox,
Optimistix, and Diffrax, which `nurbspy` supports as optional dependencies.
To our knowledge, `nurbspy` is the only JAX-compatible Python NURBS library,
which allows shape derivatives with respect to control points to be
embedded directly in a differentiable optimization or machine-learning
pipeline rather than computed as a separate, external step.

As a complementary feature, `nurbspy`'s NumPy implementation is also
real-and-complex-safe, so the same shape derivatives can independently be
computed to machine precision with the complex-step derivative method
[@martins_complex-step_2003], without any finite-difference truncation or
round-off error and without modifying user code. This provides a convenient,
independent way to verify the correctness of the JAX-computed gradients, a
useful safeguard given that automatic differentiation implementations can
silently produce incorrect derivatives for certain constructs.

`nurbspy` deliberately targets a narrower scope than a full CAD kernel: it
does not provide solid modeling, boundary representation, or CAD file I/O.
This focus keeps the codebase small, dependency-light, and easy to embed
inside gradient-based design-optimization pipelines, which is the setting
the library was originally developed for, as part of a PhD project on
turbomachinery shape optimization at the Norwegian University of Science and
Technology (NTNU). `nurbspy` has since been used and extended beyond that
original context, and its documentation includes a full derivation of the
underlying Bézier, B-Spline, NURBS, and surface-continuity theory alongside
worked, reproducible examples for every capability of the library.

# Acknowledgements

TODO: acknowledge funding sources, if any, and any contributors not listed
as authors above.

# References
