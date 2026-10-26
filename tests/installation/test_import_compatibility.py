"""Keep historical imports and saved geometry valid across package moves."""

import importlib
from pathlib import Path
import pickle
import subprocess
import sys
import textwrap

import numpy as np
from numpy.testing import assert_allclose
import pytest

import nurbspy
import nurbspy.numpy as numpy_backend


@pytest.mark.parametrize(
    "module_name, export",
    [
        ("nurbs_basis_functions", "compute_basis_polynomials"),
        ("nurbs_curve", "NurbsCurve"),
        ("nurbs_curve_circular_arc", "CircularArc"),
        ("nurbs_surface", "NurbsSurface"),
        ("nurbs_surface_bilinear", "NurbsSurfaceBilinear"),
        ("nurbs_surface_ruled", "NurbsSurfaceRuled"),
        ("nurbs_surface_extruded", "NurbsSurfaceExtruded"),
        ("nurbs_surface_revolution", "NurbsSurfaceRevolution"),
        ("nurbs_surface_coons", "NurbsSurfaceCoons"),
    ],
)
def test_legacy_modules_share_canonical_objects(module_name, export):
    """Resolve every old module path to the same NumPy module and public object."""
    legacy = importlib.import_module(f"nurbspy.{module_name}")
    canonical = importlib.import_module(f"nurbspy.numpy.{module_name}")
    assert legacy is canonical
    assert getattr(nurbspy, module_name) is canonical
    assert getattr(numpy_backend, module_name) is canonical
    assert getattr(nurbspy, export) is getattr(canonical, export)
    assert getattr(numpy_backend, export) is getattr(canonical, export)


@pytest.mark.parametrize(
    "first_import",
    [
        "import nurbspy",
        "import nurbspy.numpy.nurbs_curve",
        "from nurbspy.nurbs_curve import NurbsCurve",
    ],
    ids=["root", "explicit_numpy", "legacy_module"],
)
def test_import_order_and_wildcard_exports(first_import):
    """Check fresh import orders, module attributes, and historical star imports.

    Isolated Python still honors installed packages, including an editable
    installation, but cannot accidentally import through the working directory.
    """
    script = (
        first_import
        + "\n"
        + textwrap.dedent(
            """
        from pathlib import Path
        import sys
        import nurbspy
        import nurbspy.numpy as numpy_backend
        import nurbspy.nurbs_curve as legacy_curve
        from nurbspy import *

        assert Path(nurbspy.__file__).resolve() == Path(sys.argv[1])
        assert legacy_curve is numpy_backend.nurbs_curve
        assert nurbs_curve is legacy_curve
        assert NurbsCurve is legacy_curve.NurbsCurve
        assert NurbsSurface is numpy_backend.NurbsSurface
        assert CircularArc is numpy_backend.CircularArc
        assert make_offset_surface is numpy_backend.make_offset_surface
        assert merge_nurbs_curves is numpy_backend.merge_nurbs_curves
        assert set_plot_options is nurbspy.graphics.set_plot_options
        assert callable(print_banner) and callable(print_package_info)
        assert callable(minimal_example)
        assert PACKAGE_NAME == "nurbspy"
        assert "jax" not in sys.modules
        assert "nurbspy.jax" not in sys.modules
        """
        )
    )
    result = subprocess.run(
        [sys.executable, "-I", "-c", script, str(Path(nurbspy.__file__).resolve())],
        text=True,
        capture_output=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_shared_graphics_exports():
    """Keep all existing graphics exports available through the root interface."""
    for name, value in vars(nurbspy.graphics).items():
        if not name.startswith("_"):
            assert getattr(nurbspy, name) is value


def test_pre_migration_pickle_loads_geometry():
    """Load objects serialized before the move and check analytical geometry.

    The committed fixture was produced by the previous layout, so this catches
    missing legacy paths without rewriting current classes' module metadata.
    """
    fixture = Path(__file__).with_name("data") / "legacy_geometry.pickle"
    curve, surface = pickle.loads(fixture.read_bytes())
    assert type(curve) is numpy_backend.NurbsCurve
    assert type(surface) is numpy_backend.NurbsSurface
    assert_allclose(curve.get_value(0.25)[:, 0], [0.5, 0.375])
    assert_allclose(curve.get_derivative(0.25, 1)[:, 0], [2.0, 1.0])
    assert_allclose(surface.get_value(0.25, 0.75)[:, 0], [0.25, 0.75, 0.0])
    assert_allclose(surface.get_derivative(0.25, 0.75, 1, 0)[:, 0], [1.0, 0.0, 0.0])
    assert_allclose(surface.get_derivative(0.25, 0.75, 0, 1)[:, 0], [0.0, 1.0, 0.0])


@pytest.mark.parametrize("kind", ["curve", "surface"])
def test_new_pickle_uses_canonical_backend(kind):
    """Round-trip new geometry using its explicit NumPy class path and identity."""
    if kind == "curve":
        geometry = nurbspy.NurbsCurve(np.array([[0.0, 1.0], [0.0, 2.0]]))
        parameters = (0.3,)
    else:
        geometry = nurbspy.NurbsSurface(np.arange(12.0).reshape(3, 2, 2))
        parameters = (0.3, 0.7)
    assert type(geometry).__module__ == f"nurbspy.numpy.nurbs_{kind}"
    restored = pickle.loads(pickle.dumps(geometry))
    assert type(restored) is type(geometry)
    assert_allclose(restored.get_value(*parameters), geometry.get_value(*parameters))
