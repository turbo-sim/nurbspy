"""Gallery export caching, Sphinx assets, and on-demand embedded viewers."""

from io import StringIO
import importlib
import json
import os
from pathlib import Path
import shutil

import pytest

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"


@pytest.fixture
def gallery(monkeypatch):
    monkeypatch.syspath_prepend(str(DOCS))
    return importlib.import_module("demo_gallery")


@pytest.fixture
def cached_demo(gallery, tmp_path, monkeypatch):
    import nurbspy as nrb
    from PIL import Image

    script = tmp_path / "01_curves" / "01_line.py"
    script.parent.mkdir()
    script.write_text(
        """import nurbspy as nrb
def main(show=True):
    assert show is False
    return nrb.Figure(nrb.NurbsCurve([[0, 1], [0, 1], [0, 0]]))
""",
        encoding="utf-8",
    )
    calls = []

    def savefig(figure, destination, *, width, height):
        calls.append(destination)
        Image.new("RGB", (width, height), "white").save(destination)

    def to_html(figure, destination):
        Path(destination).write_text(
            "<!doctype html><title>Figure</title>", encoding="utf-8"
        )

    monkeypatch.setattr(nrb.Figure, "savefig", savefig)
    monkeypatch.setattr(nrb.Figure, "to_html", to_html)
    return gallery.exports, script, calls


def test_missing_exports_generate_once_then_reuse_and_invalidate(cached_demo):
    exports, script, calls = cached_demo
    renderer = ("renderer-one", 0)
    png, html = exports.ensure_export(script, width=80, height=60, renderer=renderer)
    assert len(calls) == 1 and png.exists() and html.exists()
    assert exports.ensure_export(script, width=80, height=60, renderer=renderer) == (
        png,
        html,
    )
    assert len(calls) == 1
    # A content change is detected even when timestamps are unchanged.
    modified = script.stat().st_mtime_ns
    script.write_text(
        script.read_text(encoding="utf-8") + "\n# Changed construction\n",
        encoding="utf-8",
    )
    os.utime(script, ns=(modified, modified))
    exports.ensure_export(script, width=80, height=60, renderer=renderer)
    assert len(calls) == 2
    exports.ensure_export(script, width=80, height=60, renderer=("renderer-two", 0))
    assert len(calls) == 3
    exports.ensure_export(script, width=160, height=120, renderer=("renderer-two", 0))
    assert len(calls) == 4
    from PIL import Image

    with Image.open(png) as image:
        assert image.size == (160, 120)


@pytest.mark.parametrize("damage", ["html", "png", "manifest"])
def test_incomplete_or_invalid_cached_exports_are_rebuilt(cached_demo, damage):
    exports, script, calls = cached_demo
    renderer = ("renderer", 0)
    png, html = exports.ensure_export(script, width=80, height=60, renderer=renderer)
    if damage == "manifest":
        (png.parent / ".figure-export.json").write_text("broken", encoding="utf-8")
    else:
        (html if damage == "html" else png).unlink()
    exports.ensure_export(script, width=80, height=60, renderer=renderer)
    assert len(calls) == 2


def test_existing_unstamped_exports_are_adopted_without_rendering(cached_demo):
    exports, script, calls = cached_demo
    png, html = exports.export_script(
        script, width=80, height=60, renderer=("renderer", 0)
    )
    manifest = png.parent / ".figure-export.json"
    manifest.unlink()
    exports.ensure_export(script, width=80, height=60, renderer=("renderer", 0))
    assert len(calls) == 1
    assert json.loads(manifest.read_text(encoding="utf-8"))["renderer"] == "renderer"
    manifest.unlink()
    older = script.stat().st_mtime_ns - 1_000_000
    os.utime(html, ns=(older, older))
    exports.ensure_export(script, width=80, height=60, renderer=("renderer", 0))
    assert len(calls) == 2


def test_renderer_signature_tracks_geometry_and_ignores_unrelated_backends(
    gallery, tmp_path, monkeypatch
):
    monkeypatch.setattr(gallery.exports, "PROJECT_ROOT", tmp_path)
    root = tmp_path / "src" / "nurbspy"
    for name in (
        "numpy/curve.py",
        "visualization/model.py",
        "web_assets/scene.js",
        "_basis_core.py",
        "fitting/fit.py",
        "jax/curve.py",
    ):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("original", encoding="utf-8")
    original = gallery.exports.renderer_signature()[0]
    for name in ("fitting/fit.py", "jax/curve.py"):
        (root / name).write_text("changed", encoding="utf-8")
    assert gallery.exports.renderer_signature()[0] == original
    (root / "_basis_core.py").write_text("changed", encoding="utf-8")
    assert gallery.exports.renderer_signature()[0] != original


@pytest.mark.skipif(
    os.environ.get("NURBSPY_TEST_PNG") != "1", reason="Sphinx/browser integration check"
)
def test_sphinx_gallery_preserves_outputs_and_loads_one_offline_viewer(
    gallery, tmp_path, monkeypatch
):
    from sphinx.application import Sphinx
    from playwright.sync_api import sync_playwright
    from nurbspy.visualization.figure_hosts import _launch_browser

    demos = tmp_path / "demos"
    chosen = ["01_curves/01_bezier_planar.py", "02_surfaces/02_bilinear_spatial.py"]
    for relative in chosen:
        original = ROOT / "demos" / relative
        target = demos / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, target)
        output = original.parent / "output" / original.stem
        if output.is_dir():
            shutil.copytree(output, target.parent / "output" / target.stem)
    # The gallery's scope excludes other demo categories, even when present.
    (demos / "03_viewer").mkdir()
    (demos / "03_viewer" / "01_other.py").write_text(
        "raise AssertionError('Outside gallery')", encoding="utf-8"
    )
    monkeypatch.setattr(gallery.exports, "DEMOS", demos)
    source = tmp_path / "source"
    source.mkdir()
    shutil.copyfile(DOCS / "source" / "gallery.md", source / "gallery.md")
    old_page = "# Existing examples\n\nThis page must stay unchanged.\n"
    (source / "old.md").write_text(old_page, encoding="utf-8")
    (source / "index.md").write_text(
        "# Docs\n\n```{toctree}\ngallery\nold\nfigures\n```\n", encoding="utf-8"
    )
    (source / "figures.md").write_text("# Figures\n", encoding="utf-8")
    (source / "conf.py").write_text(
        "extensions=['myst_parser', 'demo_gallery']\n"
        "source_suffix={'.md':'myst'}\nroot_doc='index'\n"
        f"html_static_path=[{str(DOCS / '_static')!r}]\n",
        encoding="utf-8",
    )
    out = tmp_path / "html"
    warnings = StringIO()
    app = Sphinx(
        str(source),
        str(source),
        str(out),
        str(tmp_path / "doctrees"),
        "html",
        status=StringIO(),
        warning=warnings,
        warningiserror=True,
        freshenv=True,
    )
    app.build(force_all=True)
    assert app.statuscode == 0, warnings.getvalue()
    assert not warnings.getvalue()
    assert (source / "old.md").read_text(encoding="utf-8") == old_page
    assert (out / "old.html").is_file()
    for relative in chosen:
        script = demos / relative
        original_output = script.parent / "output" / script.stem
        copied = out / "_gallery" / script.parent.name / script.stem
        for suffix in (".html", ".png"):
            assert (copied / (script.stem + suffix)).read_bytes() == (
                original_output / (script.stem + suffix)
            ).read_bytes()
        assert (copied / script.name).read_bytes() == script.read_bytes()
    with sync_playwright() as playwright:
        browser = _launch_browser(playwright)
        page = browser.new_page(viewport={"width": 1280, "height": 950})
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto((out / "gallery.html").as_uri())
        assert page.locator(".nrb-demo-card").count() == 2
        assert page.locator("iframe").count() == 0
        previews = page.locator(".nrb-demo-preview")
        previews.nth(0).locator("img").evaluate("image=>image.decode()")
        assert previews.nth(0).locator("img").evaluate("image=>image.naturalWidth>0")
        for index in (0, 1):
            previews.nth(index).click()
            assert page.locator("iframe").count() == 1
            frame = page.frame_locator(".nrb-demo-viewer iframe")
            frame.locator("canvas.nrb-renderer").wait_for()
            assert (
                frame.get_by_role(
                    "button", name="Open editing controls", exact=True
                ).count()
                == 0
            )
        page.get_by_role("button", name="Close interactive figure", exact=True).click()
        assert page.locator("iframe").count() == 0
        assert previews.nth(1).evaluate("element=>document.activeElement===element")
        page.set_viewport_size({"width": 390, "height": 844})
        previews.nth(0).click()
        page.frame_locator("iframe").locator("canvas.nrb-renderer").wait_for()
        assert page.evaluate("document.documentElement.scrollWidth<=innerWidth")
        assert not errors
        browser.close()
