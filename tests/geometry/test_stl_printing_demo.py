"""Reference geometry and end-to-end printing example, including saved plots."""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image
import pytest
import trimesh


SCRIPT = Path(__file__).resolve().parents[2] / "demos" / "demos_surfaces" / "demo_stl_printing.py"
spec = importlib.util.spec_from_file_location("stl_printing_demo", SCRIPT)
demo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(demo)


def test_torus_is_exact_and_periodic():
    torus = demo.make_torus(30, 10)
    u, v = np.linspace(0, 1, 53), np.linspace(0, 1, 53)**1.3
    x, y, z = torus.get_value(u, v)
    np.testing.assert_allclose((np.hypot(x, y) - 30)**2 + z**2, 100, atol=1e-11)
    np.testing.assert_allclose(torus.get_value(0*u, v), torus.get_value(0*u+1, v), atol=1e-12)
    np.testing.assert_allclose(torus.get_value(u, 0*v), torus.get_value(u, 0*v+1), atol=1e-12)


def test_oloid_contacts_are_on_circles_and_rulings_have_correct_length():
    radius = 20
    patches = demo.make_oloid(radius, 61)
    assert len(patches) == 2
    for patch in patches:
        first, second = patch.P[:, :, 0], patch.P[:, :, 1]
        np.testing.assert_allclose((first[0]+radius/2)**2 + first[1]**2, radius**2, atol=1e-10)
        np.testing.assert_allclose((second[0]-radius/2)**2 + second[2]**2, radius**2, atol=1e-10)
        np.testing.assert_allclose(np.linalg.norm(second-first, axis=0), np.sqrt(3)*radius, atol=1e-12)


def test_torus_mesh_refinement_improves_analytic_volume(tmp_path):
    torus = demo.make_torus()
    exact = 2 * np.pi**2 * 30 * 10**2
    errors = []
    for quality in ("preview", "print"):
        nu, nv = demo.QUALITY[quality]["torus"]
        path = torus.export(tmp_path/f"{quality}.stl", nu=nu, nv=nv, require_closed=True)
        mesh = trimesh.load_mesh(path)
        assert mesh.is_watertight and mesh.is_volume
        errors.append(abs(mesh.volume-exact))
    assert errors[1] < errors[0] / 10


@pytest.mark.parametrize("major,minor", [(10, 10), (5, 10), (30, -1), (np.inf, 10)])
def test_only_ring_tori_are_constructed(major, minor):
    with pytest.raises(ValueError):
        demo.make_torus(major, minor)


@pytest.mark.parametrize("count", [3, 20, True])
def test_invalid_oloid_boundary_sampling_is_rejected(count):
    with pytest.raises(ValueError, match="odd integer"):
        demo.make_oloid(boundary_points=count)


def test_complete_cli_creates_closed_stls_and_plot_without_window(tmp_path):
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--quality", "preview", "--no-show", "--output-dir", str(tmp_path)],
        cwd=SCRIPT.parents[2], capture_output=True, text=True, timeout=60,
        env=dict(os.environ, MPLBACKEND="Agg", NUMBA_DISABLE_JIT="1"),
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "STL has no units" in result.stdout
    assert "Tight layout" not in result.stderr
    for name, dimensions, count in (("torus", [80, 80, 20], 1024), ("oloid", [60, 40, 40], 480)):
        mesh = trimesh.load_mesh(tmp_path/f"{name}_preview.stl")
        assert mesh.is_watertight and mesh.is_volume
        assert len(mesh.faces) == count
        # The sampled oloid can be slightly undersized; the rational torus is exact.
        tolerance = .01 if name == "oloid" else 1e-5
        np.testing.assert_allclose(np.ptp(mesh.bounds, axis=0), dimensions, atol=tolerance)
    with Image.open(tmp_path/"stl_printing_preview.png") as plot:
        assert plot.format == "PNG"
        assert plot.width > 1000 and plot.height > 1000
