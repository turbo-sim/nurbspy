"""Numbered teaching demos and read-only mesh annotations."""

from io import StringIO
from pathlib import Path
import importlib
import os
import subprocess
import sys

import numpy as np
import pytest
import nurbspy as nrb

ROOT = Path(__file__).resolve().parents[2]
TUTORIALS = [
    f"demos.{folder}.{path.stem}"
    for folder in ("01_curves", "02_surfaces")
    for path in sorted((ROOT / "demos" / folder).glob("[0-9]*.py"))
]


@pytest.mark.parametrize(
    "relative",
    [
        "00_basis_polynomials/01_basis_polynomials.py",
        "01_curves/01_bezier_planar.py",
        "02_surfaces/01_bilinear_planar.py",
    ],
)
def test_demo_runs_as_standalone_file_outside_checkout(relative, tmp_path):
    # Copy just the script, with no demos package or checkout on Python's path.
    script = tmp_path / Path(relative).name
    source = (ROOT / "demos" / relative).read_text(encoding="utf-8")
    basis = relative.startswith("00_basis_polynomials/")
    if not basis:
        source = source.replace("    main()", "    main(show=False)")
    script.write_text(source, encoding="utf-8")
    result = subprocess.run(
        [sys.executable, "-I", str(script)],
        cwd=tmp_path,
        env=dict(os.environ, MPLBACKEND="Agg"),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("name", TUTORIALS)
def test_tutorial_opens_one_editor_with_auto_quality_and_grid(
    name, monkeypatch, tmp_path
):
    module = importlib.import_module(name)
    if hasattr(module, "OUTPUT"):
        monkeypatch.setattr(module, "OUTPUT", tmp_path)
    opened = []
    monkeypatch.setattr(nrb.Figure, "edit", lambda figure: opened.append(figure))
    monkeypatch.setattr(
        nrb.Figure, "show", lambda figure: pytest.fail("Demo opened a viewer")
    )
    figure = module.main()
    source = Path(module.__file__).read_text(encoding="utf-8")
    assert "matplotlib" not in source and "sys.path" not in source
    assert isinstance(figure, nrb.Figure)
    assert opened == [figure]
    assert figure.resolution is None
    assert "gallery" not in source and "EXPORT_FIGURES" not in source
    assert ".savefig(" not in source and ".to_html(" not in source
    assert len(module.__doc__.splitlines()) >= 3
    restored = nrb.Figure.from_state(figure.to_state())
    assert restored.to_state() == figure.to_state()
    for object_name in figure.names:
        assert np.isfinite(figure.geometry(object_name).P).all()
        assert figure.settings(object_name)["grid"] is True
        assert figure.settings(object_name)["quality"] == "auto"
    if name == "demos.02_surfaces.13_oloid_ruled":
        for object_name in figure.names:
            assert figure.settings(object_name)["control_points"] is False
    else:
        assert figure.settings()["control_points"] is True
    assert figure.settings()["control_polygon"] is True


def test_demo_isocurves_are_linked_overlays_rather_than_editable_copies():
    surface = importlib.import_module("demos.02_surfaces.05_bspline")
    construction = surface.main(show=False)
    assert construction.settings()["isocurves"] is True
    assert not any(name.startswith(("u=", "v=")) for name in construction.names)
    scene = construction._payload()["scenes"][construction.active]
    curve = scene["diagnostics"]["isocurves"]["curves"][0]
    uv = np.asarray(scene["sampling"]["full"]["v"])
    np.testing.assert_allclose(
        np.asarray(curve["points"]).T,
        construction.geometry().get_value(uv * 0 + 0.08, uv),
        atol=1e-13,
    )
    assert any(name.startswith("Normal ") for name in construction.names)


def test_mesh_snapshot_round_trip_and_copy_isolation():
    curve = nrb.NurbsCurve([[0, 1], [0, 1], [0, 0]])
    vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    figure = nrb.Figure(curve, resolution=8).mesh(vertices, [[0, 1, 2]], wireframe=True)
    vertices[:] = 100
    restored = nrb.Figure.from_yaml(StringIO(figure.to_yaml()))
    assert restored.to_state() == figure.to_state()
    assert restored.to_state()["meshes"][0]["vertices"][0] == [0.0, 0.0, 0.0]
    assert restored.names == ("geometry",)


@pytest.mark.parametrize(
    "vertices,triangles",
    [
        ([[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[0, 1, 3]]),
        ([[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[0.0, 1.0, 2.0]]),
        ([[0, 0, float("nan")], [1, 0, 0], [0, 1, 0]], [[0, 1, 2]]),
    ],
)
def test_mesh_rejects_invalid_geometry(vertices, triangles):
    with pytest.raises(ValueError):
        nrb.Figure().mesh(vertices, triangles)


@pytest.mark.skipif(
    os.environ.get("NURBSPY_TEST_PNG") != "1",
    reason="Explicit headless browser integration check",
)
def test_batch_transitions_between_curve_surface_and_reloaded_mesh(tmp_path):
    from PIL import Image

    curve = nrb.NurbsCurve([[0, 1, 2], [0, 1, 0], [0, 0, 0]])
    surface = nrb.NurbsSurface([[[0, 0], [1, 1]], [[0, 1], [0, 1]], [[0, 0], [0, 1]]])
    a = nrb.Figure(curve, title="Curve", resolution=8, display=dict(view="xy"))
    b = nrb.Figure(surface, title="Surface", resolution=8)
    c = nrb.Figure(surface, title="Reloaded triangles", resolution=8).update(
        visible=False
    )
    c.mesh([[0, 0, 0], [1, 0, 0], [0, 1, 1]], [[0, 1, 2]], wireframe=True)
    paths = a.saveframes([a, b, c], tmp_path, width=800, height=600)
    pixels = []
    for path in paths:
        with Image.open(path) as image:
            assert image.size == (800, 600)
            rgb = np.asarray(image.convert("RGB"))
            assert np.count_nonzero(np.any(rgb < 240, axis=2)) > 500
            pixels.append(rgb)
    assert not np.array_equal(pixels[0], pixels[1])
    assert not np.array_equal(pixels[1], pixels[2])
