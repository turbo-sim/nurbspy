"""Sphinx integration for the two numbered interactive geometry demo folders."""

import ast
from html import escape
from pathlib import Path
import posixpath
import shutil
from urllib.parse import quote

from sphinx.errors import ExtensionError
from sphinx.util import logging

import export_demo_figures as exports

LOGGER = logging.getLogger(__name__)
CATEGORIES = {"01_curves": "curves", "02_surfaces": "surfaces"}


def _title(script):
    words = script.stem.split("_")[1:]
    dimension = words.pop() if words[-1] in ("planar", "spatial") else None
    spelling = {
        "bezier": "Bézier",
        "bspline": "B-spline",
        "nurbs": "NURBS",
        "stl": "STL",
        "coons": "Coons",
        "oloid": "oloid",
    }
    title = " ".join(spelling.get(word, word) for word in words)
    title = title[0].upper() + title[1:]
    return title + (f" ({dimension})" if dimension else "")


def prepare_gallery(app):
    """Prepare assets outside the source tree, avoiding live-preview loops."""
    if app.builder.format != "html":
        return
    signature = exports.renderer_signature()
    entries = []
    for script in exports.discover_scripts():
        if script.parent.name not in CATEGORIES:
            continue
        try:
            png, html = exports.ensure_export(script, renderer=signature)
        except Exception as error:
            raise ExtensionError(
                f"Could not export gallery demo {script.name}. Run "
                f"'python docs/export_demo_figures.py --only {script.parent.name}/{script.stem}' "
                "to diagnose the export (a Playwright browser is required).",
                orig_exc=error,
            ) from error
        relative = Path("_gallery") / script.parent.name / script.stem
        destination = Path(app.outdir) / relative
        destination.mkdir(parents=True, exist_ok=True)
        for asset in (png, html, script):
            target = destination / asset.name
            if not target.exists() or target.read_bytes() != asset.read_bytes():
                shutil.copyfile(asset, target)
        description = (
            ast.get_docstring(ast.parse(script.read_text(encoding="utf-8")))
        ) or ""
        entries.append(
            dict(
                category=CATEGORIES[script.parent.name],
                number=script.name[:2],
                title=_title(script),
                description=description.splitlines()[0] if description else "",
                base=relative.as_posix(),
                stem=script.stem,
                source=script.name,
            )
        )
    app._nurbspy_gallery_entries = entries
    LOGGER.info("Interactive gallery: %s curve and surface figures ready", len(entries))


def render_cards(entries, prefix):
    """Render inert previews; JavaScript creates an iframe only on request."""
    cards = ['<div class="nrb-gallery-grid">']
    for entry in entries:
        base = posixpath.join(prefix, entry["base"])
        title = escape(entry["title"])
        png = escape(quote(f'{base}/{entry["stem"]}.png'), quote=True)
        html = escape(quote(f'{base}/{entry["stem"]}.html'), quote=True)
        source = escape(quote(f'{base}/{entry["source"]}'), quote=True)
        cards.append(f"""<article class="nrb-demo-card">
<h3><span class="nrb-demo-number">{entry['number']}</span> {title}</h3>
<p>{escape(entry['description'])}</p>
<button class="nrb-demo-preview" type="button" data-demo-html="{html}" data-demo-title="{title}" aria-label="Explore {title}">
<img src="{png}" alt="{title}" width="1600" height="1200" loading="lazy" decoding="async">
<span>Explore interactively</span></button>
<div class="nrb-demo-viewer" hidden></div>
<div class="nrb-demo-links"><a href="{html}" target="_blank" rel="noopener">Open full page</a>
<a href="{html}" download>Download HTML</a><a href="{source}" download>Python source</a></div>
</article>""")
    cards.append("</div>")
    return "\n".join(cards)


def gallery_source(app, docname, source):
    if docname != "gallery":
        return
    if app.builder.format != "html":
        return
    target = app.builder.get_target_uri(docname).split("#", 1)[0]
    prefix = posixpath.relpath(".", posixpath.dirname(target) or ".")
    entries = app._nurbspy_gallery_entries
    for category in CATEGORIES.values():
        cards = render_cards(
            [entry for entry in entries if entry["category"] == category], prefix
        )
        source[0] = source[0].replace(f"<!-- demo-gallery:{category} -->", cards)


def gallery_outdated(app, env, added, changed, removed):
    # The script list and captions can change without modifying gallery.md.
    return ["gallery"] if app.builder.format == "html" else []


def setup(app):
    app.connect("builder-inited", prepare_gallery)
    app.connect("source-read", gallery_source)
    app.connect("env-get-outdated", gallery_outdated)
    app.add_css_file("gallery.css")
    app.add_js_file("gallery.js", defer="defer")
    return {"version": "1", "parallel_read_safe": True, "parallel_write_safe": True}
