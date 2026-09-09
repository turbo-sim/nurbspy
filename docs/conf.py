"""Sphinx configuration and project-specific settings for build_docs.py."""
import sys
from pathlib import Path

DOCS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(DOCS_DIR.parent))
from nurbspy import __version__

project = "nurbspy"
author = "Roberto Agromayor"
copyright = "2026, Roberto Agromayor"
release = __version__

# All paths are relative to this file; build_docs.py is shared unchanged.
docs_build_config = {
    "source_dir": "source",
    "build_dir": "_build/html",
    "api_output_dir": "source/api",
    "src_dir": "../nurbspy",
    "exclude_modules": [],
    "sphinx_build_options": ["--fail-on-warning", "--keep-going"],
    "zotero_api_key_env": "ZOTERO_API_KEY",
    "zotero_group_id": "5252389",
    "bibliography_file": "source/references/bibliography.bib",
}

extensions = [
    "sphinx.ext.autodoc", "sphinx.ext.autosummary", "sphinx.ext.viewcode",
    "sphinx.ext.mathjax", "numpydoc", "sphinxcontrib.bibtex", "myst_parser",
    "sphinx_design",
]
source_suffix = {".rst": "restructuredtext", ".md": "myst"}
root_doc = "index"
myst_enable_extensions = ["amsmath", "colon_fence", "dollarmath"]
myst_heading_anchors = 3
autosummary_generate = True
autodoc_default_options = {"members": True, "show-inheritance": True}
numpydoc_class_members_toctree = False

# Bibliography paths are resolved relative to the configuration directory.
# Hand-maintained theory entries survive an optional Zotero export.
bibtex_bibfiles = [
    "source/references/bibliography.bib",
    "source/references/theory.bib",
]
bibtex_default_style = "alpha"
bibtex_reference_style = "author_year"


def _always_rebuild_bibliography(app, env, added, changed, removed):
    """Refresh collected citations when other notes change during live preview.

    Adding or removing citations elsewhere does not change the bibliography
    page itself. Mark it outdated so incremental builds refresh it without
    touching files and triggering another live-preview rebuild.
    """
    return ["references/bibliography"]


def setup(app):
    app.connect("env-get-outdated", _always_rebuild_bibliography)


exclude_patterns = ["Thumbs.db", ".DS_Store"]
html_theme = "sphinx_book_theme"
html_title = "nurbspy"
html_baseurl = "https://turbo-sim.github.io/nurbspy/"
html_theme_options = {
    "repository_url": "https://github.com/turbo-sim/nurbspy",
    "repository_branch": "main",
    "use_repository_button": True,
    "use_issues_button": True,
    "use_edit_page_button": True,
    "path_to_docs": "docs/source",
    "show_toc_level": 2,
}
