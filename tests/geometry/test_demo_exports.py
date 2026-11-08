"""Numbered demo discovery and matching offline HTML/PNG exports."""

import importlib.util
import os
from pathlib import Path
import shutil

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "demo_exports", ROOT / "docs" / "export_demo_figures.py"
)
exports = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exports)


def test_discovery_finds_each_single_example_and_resolves_cli_selections():
    scripts = exports.discover_scripts()
    assert len(scripts) == 35
    for folder in ("01_curves", "02_surfaces"):
        numbers = [int(path.name[:2]) for path in scripts if path.parent.name == folder]
        assert numbers == list(range(1, len(numbers) + 1))
    selected = exports.discover_scripts(
        ["01_curves/01_bezier_planar.py", "22_stl_oloid"]
    )
    assert [path.stem for path in selected] == ["01_bezier_planar", "22_stl_oloid"]
    with pytest.raises(ValueError, match="Expected one demo"):
        exports.discover_scripts(["not_a_demo"])


@pytest.mark.skipif(
    os.environ.get("NURBSPY_TEST_PNG") != "1", reason="Explicit browser export check"
)
@pytest.mark.parametrize(
    "relative", ["01_curves/01_bezier_planar.py", "02_surfaces/02_bilinear_spatial.py"]
)
def test_export_writes_offline_html_and_png_beside_the_script(
    relative, tmp_path, monkeypatch
):
    from PIL import Image
    from playwright.sync_api import sync_playwright
    import nurbspy as nrb
    from nurbspy.visualization.figure_hosts import _launch_browser

    # Run the actual teaching script outside the checkout, including the exporter.
    script = tmp_path / relative
    script.parent.mkdir(parents=True)
    shutil.copyfile(ROOT / "demos" / relative, script)
    cameras = []
    to_html = nrb.Figure.to_html

    def capture_html_camera(figure, *args, **kwargs):
        cameras.append(figure.camera)
        return to_html(figure, *args, **kwargs)

    monkeypatch.setattr(nrb.Figure, "to_html", capture_html_camera)
    monkeypatch.chdir(tmp_path)
    png, html = exports.export_script(script, width=800, height=600)
    assert png.parent == html.parent == script.parent / "output" / script.stem
    assert not (tmp_path / "output").exists()
    assert png.stem == html.stem == script.stem
    with Image.open(png) as image:
        assert image.size == (800, 600)
        assert (
            np.count_nonzero(np.any(np.asarray(image.convert("RGB")) < 240, axis=2))
            > 500
        )

    with sync_playwright() as playwright:
        browser = _launch_browser(playwright)
        page = browser.new_page()
        errors, network = [], []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.on(
            "request",
            lambda request: (
                network.append(request.url)
                if request.url.startswith(("http:", "https:"))
                else None
            ),
        )
        page.goto(html.as_uri())
        page.wait_for_function("window.nurbspyReportReady === true")
        page.evaluate("async()=>await window.nurbspyFigure.ready()")
        state = page.evaluate("window.nurbspyFigure.snapshot()")
        for key, value in cameras[0].items():
            if isinstance(value, str):
                assert state["camera"][key] == value
            else:
                np.testing.assert_allclose(
                    state["camera"][key], value, rtol=1e-12, atol=1e-12
                )
        assert all(obj["display"]["grid"] for obj in state["objects"].values())
        assert not page.get_by_role(
            "button", name="Open editing controls", exact=True
        ).count()
        page.get_by_role("button", name="Open display settings", exact=True).click()
        assert page.get_by_role(
            "checkbox", name="Reference grid", exact=True
        ).is_checked()
        assert not errors and not network
        browser.close()
