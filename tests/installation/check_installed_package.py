"""Smoke-check an installed wheel with ``python -I`` outside the checkout.

This script deliberately lives outside pytest discovery: source-checkout tests
and installation checks need different import paths. CI runs it after installing
each supported dependency selection, including the combined optional extras.
"""

import argparse
from importlib import metadata
import importlib.util
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

import nurbspy as nrb


def check_import_compatibility():
    """Verify canonical and legacy paths in the installed wheel, without JAX."""
    import sys
    import nurbspy.numpy as numpy_backend
    import nurbspy.nurbs_curve as legacy_curve
    import nurbspy.nurbs_surface as legacy_surface
    from nurbspy.graphics import set_plot_options

    assert nrb.NurbsCurve is numpy_backend.NurbsCurve is legacy_curve.NurbsCurve
    assert nrb.NurbsSurface is numpy_backend.NurbsSurface is legacy_surface.NurbsSurface
    assert legacy_curve is numpy_backend.nurbs_curve
    assert legacy_surface is numpy_backend.nurbs_surface
    assert nrb.set_plot_options is set_plot_options
    assert "jax" not in sys.modules
    assert "nurbspy.jax" not in sys.modules


def check_distribution_metadata():
    """Verify the imported package is installed and declares optional extras."""
    distribution = metadata.distribution("nurbspy")
    checkout = Path(__file__).resolve().parents[2]
    imported_file = Path(nrb.__file__).resolve()
    installed_file = Path(distribution.locate_file("nurbspy/__init__.py")).resolve()
    assert not imported_file.is_relative_to(checkout), imported_file
    assert imported_file == installed_file, (imported_file, installed_file)
    assert nrb.__version__ == distribution.version
    assert {"numba", "jax"} <= set(distribution.metadata.get_all("Provides-Extra", []))

    # Metadata must keep every optional dependency out of a default install.
    requirements = [Requirement(value) for value in distribution.requires or []]
    optional_dependencies = {
        "numba": "numba",
        "jax": "jax",
        "jaxlib": "jax",
        "equinox": "jax",
        "diffrax": "jax",
        "optimistix": "jax",
        "quadax": "jax",
    }
    for name, extra in optional_dependencies.items():
        matches = [
            item for item in requirements if canonicalize_name(item.name) == name
        ]
        assert matches, f"Missing dependency metadata for {name}."
        assert all(
            item.marker is not None and not item.marker.evaluate({"extra": ""})
            for item in matches
        ), f"{name} must not be required by a default installation."
        assert any(item.marker.evaluate({"extra": extra}) for item in matches)


def check_installed_backends(backend):
    """Check selected extras, basic geometry, and their defining capabilities."""
    expect_numba = backend in {"numba", "all"}
    expect_jax = backend in {"jax", "all"}
    assert (importlib.util.find_spec("numba") is not None) == expect_numba
    assert (importlib.util.find_spec("jax") is not None) == expect_jax

    points = np.array([[0.0, 1.0, 2.0], [0.0, 1.0, 0.0]])
    curve = nrb.NurbsCurve(control_points=points)
    assert_allclose(curve.get_value(0.25)[:, 0], [0.5, 0.375])
    assert_allclose(curve.get_value(0.25 + 1e-30j).imag[:, 0] / 1e-30, [2.0, 1.0])
    if expect_numba:
        assert nrb.compute_basis_polynomials.nopython_signatures
    else:
        assert not hasattr(nrb.compute_basis_polynomials, "py_func")

    if expect_jax:
        import jax
        import nurbspy.jax as jax_nrb

        jax_curve = jax_nrb.NurbsCurve(control_points=points)
        value = jax_curve.get_value(0.25)[:, 0]
        derivative = jax.jacfwd(lambda u: jax_curve.get_value(u)[:, 0])(0.25)
        assert_allclose(value, [0.5, 0.375])
        assert_allclose(derivative, [2.0, 1.0])
        # Exercise the Quadax order adapter from the installed wheel too.
        # This parabola's length is sqrt(2) + asinh(1).
        assert_allclose(
            jax_curve.get_arclength_quad(), np.sqrt(2.0) + np.arcsinh(1.0),
            rtol=1e-11,
        )
        assert_allclose(jax_curve.get_arclength(tol=1e-10),
                        np.sqrt(2.0) + np.arcsinh(1.0), rtol=1e-10)
        arclength_table = jax_nrb.prepare_arclength(jax_curve)
        assert_allclose(arclength_table.value(points),
                        np.sqrt(2.0) + np.arcsinh(1.0), rtol=1e-10)
        inverse, total = jax_curve.reparametrize_by_arclength(tol=1e-9)
        assert_allclose(jax.jit(inverse)(np.array([0.0, float(total)])),
                        [0.0, 1.0], atol=1e-10)

        # Verify prepared APIs are exported and included in the built wheel.
        basis = jax_nrb.prepare_curve(jax_curve, [0.25], max_order=1)
        assert_allclose(basis.get_value(points)[:, 0], [0.5, 0.375])
        assert_allclose(basis.get_derivative(points)[:, 0], [2.0, 1.0])

        plane_points = np.array([
            [[0.0, 0.0], [1.0, 1.0]],
            [[0.0, 1.0], [0.0, 1.0]],
            [[0.0, 0.0], [0.0, 0.0]],
        ])
        plane = jax_nrb.NurbsSurface(plane_points)
        grid = jax_nrb.prepare_surface(
            plane, [0.25], [0.75], max_order_u=0, max_order_v=0, grid=True
        )
        grid_values = grid.get_value(plane_points)
        assert grid_values.shape == (3, 1, 1)
        assert_allclose(grid_values[:, 0, 0], [0.25, 0.75, 0.0])
        projection = jax_nrb.prepare_surface_projection(plane)
        assert_allclose(
            projection.project_point(np.array([0.25, 0.75, 1.0])),
            [0.25, 0.75], atol=1e-10,
        )

        # Exercise the exported surface generators and mixed-partial API from
        # the installed wheel, including a gradient through construction.
        def surface_value(distance):
            """Construct an extrusion and expose its signed-distance sensitivity."""
            curve = jax_nrb.NurbsCurve(np.array([[0.0, 1.0], [0.0, 0.0], [0.0, 0.0]]))
            surface = jax_nrb.NurbsSurfaceExtruded(
                curve, np.array([0.0, 1.0, 0.0]), distance
            ).NurbsSurface
            return surface.get_value(0.25, 0.75)[:, 0]

        assert_allclose(surface_value(2.0), [0.25, 1.5, 0.0])
        assert_allclose(jax.jacfwd(surface_value)(2.0), [0.0, 0.75, 0.0])


def main():
    """Run the isolated smoke checks for the requested dependency selection."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend", choices=("numpy", "numba", "jax", "all"), required=True
    )
    arguments = parser.parse_args()
    check_distribution_metadata()
    check_import_compatibility()
    check_installed_backends(arguments.backend)
    print(f"Installed nurbspy {nrb.__version__}: {arguments.backend} checks passed.")


if __name__ == "__main__":
    main()
