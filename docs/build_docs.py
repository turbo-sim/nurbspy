"""
build_docs.py
=============
Build and preview the Sphinx documentation for this project.

Run this script using Poetry. This ensures that Sphinx and all documentation
dependencies are loaded from the environment defined in pyproject.toml.

Most common command
-------------------
Start a local documentation server with automatic rebuilding:

    poetry run python docs/build_docs.py

This will:

1. Check that the script is running inside a Python virtual environment.
2. Delete the previous documentation build.
3. Generate the API documentation from Python docstrings.
4. Build the Sphinx documentation.
5. Open the documentation in your web browser.
6. Watch the documentation files and rebuild automatically when they change.

By default, the documentation is served locally at:

    http://127.0.0.1:8000

Press Ctrl+C in the terminal to stop the live-preview server.


Other useful commands
---------------------

Regenerate the bibliography from Zotero before building the documentation:

    poetry run python docs/build_docs.py --build-bib

Regenerate the bibliography from Zotero without building the documentation:

    poetry run python docs/build_docs.py --build-bib-only

Build the documentation once without starting the live-preview server:

    poetry run python docs/build_docs.py --no-autobuild


Start the live-preview server without automatically opening a browser tab:

    poetry run python docs/build_docs.py --no-open-browser


Delete the generated documentation and exit without rebuilding:

    poetry run python docs/build_docs.py --clean-only


Force regeneration of the automatically generated API pages:

    poetry run python docs/build_docs.py --force-apidoc


Use a different port for the live-preview server:

    poetry run python docs/build_docs.py --port 8080


Options can be combined. For example, regenerate the bibliography and then
perform a one-time HTML build:

    poetry run python docs/build_docs.py --build-bib --no-autobuild


Notes
-----
The script resolves all project paths relative to its own location, so it is
not affected by the current working directory.

The recommended workflow is nevertheless to run it from the project root
using Poetry:

    poetry run python docs/build_docs.py

Poetry provides the virtual environment. This script then launches Sphinx
from that same environment.

The bibliography page (source/references/bibliography.md) is always kept up to date
with citations added anywhere in the docs -- that is handled by an
``env-get-outdated`` hook in conf.py, not by this script. See conf.py for
details.

"""


import os
import sys
import time
import shutil
import argparse
import subprocess
from pathlib import Path

from conf import docs_build_config


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# Directory containing this script and conf.py
DOCS_DIR = Path(__file__).resolve().parent

# Repository root
PROJECT_ROOT = DOCS_DIR.parent

# Sphinx documentation sources
SOURCE_DIR = (DOCS_DIR / docs_build_config["source_dir"]).resolve()

# Generated Sphinx output
BUILD_DIR = (DOCS_DIR / docs_build_config["build_dir"]).resolve()

# Generated sphinx-apidoc .rst files
API_OUTPUT_DIR = (DOCS_DIR / docs_build_config["api_output_dir"]).resolve()

# Python source directory documented by sphinx-apidoc.
#
# This path is configured in conf.py so that this build script can be reused
# unchanged by projects with different source layouts.
SRC_DIR = (DOCS_DIR / docs_build_config["src_dir"]).resolve()

# Paths excluded from sphinx-apidoc.
#
# sphinx-apidoc matches these with fnmatch against each file's full path, so
# a bare directory path only matches that exact path -- it never matches the
# files inside it. A trailing wildcard is required to exclude the directory's
# contents.
EXCLUDE_MODULES = [
    str((DOCS_DIR / path).resolve())
    for path in docs_build_config["exclude_modules"]
]

# Extra command-line options used by one-shot Sphinx builds (including CI).
SPHINX_BUILD_OPTIONS = docs_build_config.get("sphinx_build_options", [])

# Zotero bibliography settings
API_KEY_ENV = docs_build_config.get("zotero_api_key_env")
API_KEY = os.environ.get(API_KEY_ENV) if API_KEY_ENV else None
GROUP_ID = docs_build_config.get("zotero_group_id")
OUTPUT_FILE = (
    DOCS_DIR / docs_build_config.get("bibliography_file", "bibliography.bib")
).resolve()


def venv_tool(name):
    """
    Resolve a console-script tool (e.g. "sphinx-autobuild") to the copy
    installed alongside the current Python interpreter, instead of letting
    subprocess search PATH.

    Calling these tools by bare name is unsafe on systems with multiple
    Python environments on PATH (e.g. Conda): if the tool isn't installed in
    the active virtual environment, PATH lookup can silently fall through to
    a same-named executable from a different, incompatible environment.
    """
    venv_bin = Path(sys.executable).parent
    candidate = venv_bin / (f"{name}.exe" if sys.platform == "win32" else name)

    if not candidate.exists():
        raise RuntimeError(
            f"'{name}' was not found in the active virtual environment "
            f"({venv_bin}).\n\n"
            "Install the documentation dependencies with:\n\n"
            "    poetry install --with dev\n"
        )

    return str(candidate)


def check_environment():
    """
    Verify that the script is running inside a Python virtual environment.

    The recommended way to launch this script is through Poetry:

        poetry run python docs/build_docs.py

    This check prevents the documentation tools from accidentally running
    against a global Python installation.
    """
    if sys.prefix == sys.base_prefix:
        raise RuntimeError(
            "\n"
            "The documentation builder is not running inside a virtual "
            "environment.\n\n"
            "Run it from the project root with:\n\n"
            "    poetry run python docs/build_docs.py\n"
        )

    print(f"Using Python environment: {sys.executable}")
    print()


def _rmtree_with_retry(path, attempts=5, delay=0.4):
    """
    Delete a directory tree, retrying briefly on a transient PermissionError.

    This project lives in a OneDrive-synced folder. OneDrive can briefly hold
    a lock on files just after they're written (e.g. Sphinx's doctree
    pickles) while it scans them for upload, which makes shutil.rmtree fail
    with "Access is denied" moments after a build completes -- most commonly
    when the live-preview server is stopped and restarted quickly. The lock
    clears on its own within a fraction of a second in practice, so a short
    retry loop rides it out instead of crashing the whole build.
    """
    path = Path(path).resolve()
    allowed_roots = (DOCS_DIR / "_build", DOCS_DIR / "source" / "api")
    if not any(path == root.resolve() or path.is_relative_to(root.resolve())
               for root in allowed_roots):
        raise ValueError(f"Refusing to delete outside generated documentation: {path}")

    for attempt in range(1, attempts + 1):
        try:
            shutil.rmtree(path)
            return
        except PermissionError:
            if attempt == attempts:
                raise
            time.sleep(delay)


def delete_build(build_dir=BUILD_DIR):
    """Delete a previously generated build directory."""
    build_dir = Path(build_dir)

    if build_dir.exists():
        _rmtree_with_retry(build_dir)
        print(f"Deleted build directory: {build_dir}")


def run_sphinx_apidoc(
    output_dir=API_OUTPUT_DIR,
    src_dir=SRC_DIR,
    exclude=None,
    force=False,
):
    """
    Run sphinx-apidoc to generate .rst API stubs from docstrings.

    Parameters
    ----------
    output_dir
        Directory that receives the generated .rst stubs.
    src_dir
        Root directory of the Python package.
    exclude
        Sub-paths to exclude from documentation.
    force
        Overwrite existing stubs when True.
    """
    if output_dir.exists():
        _rmtree_with_retry(output_dir)

    cmd = [
        venv_tool("sphinx-apidoc"),
        "-o",
        str(output_dir),
        str(src_dir),
    ]

    if isinstance(exclude, list):
        cmd.extend(str(path) for path in exclude)

    cmd += [
        "-e",        # one page per module
        "--no-toc",  # suppress modules.rst
        "-M",        # module first
    ]

    if force:
        cmd.append("-f")

    subprocess.check_call(cmd)
    print("sphinx-apidoc completed successfully.")


def run_sphinx_build(
    docs_dir=SOURCE_DIR,
    conf_dir=DOCS_DIR,
    build_dir=BUILD_DIR,
    builder="html",
):
    """
    Run sphinx-build for a one-shot documentation build.

    Parameters
    ----------
    docs_dir
        Directory containing the Sphinx source files.
    conf_dir
        Directory containing conf.py.
    build_dir
        Output directory.
    builder
        Sphinx builder name (html, latex, dirhtml ...).
    """
    cmd = [
        venv_tool("sphinx-build"),
        *SPHINX_BUILD_OPTIONS,
        "--conf-dir",
        str(conf_dir),
        "-b",
        builder,
        str(docs_dir),
        str(build_dir),
    ]

    subprocess.check_call(cmd)
    print(f"sphinx-build ({builder}) completed successfully.")


def run_sphinx_autobuild(
    docs_dir=SOURCE_DIR,
    conf_dir=DOCS_DIR,
    build_dir=BUILD_DIR,
    port=8000,
    open_browser=True,
):
    """
    Run sphinx-autobuild for live preview with hot-reload.

    Parameters
    ----------
    docs_dir
        Directory containing the Sphinx source files.
    conf_dir
        Directory containing conf.py.
    build_dir
        Output directory.
    port
        Local port to serve the HTML site.
    """
    cmd = [
        venv_tool("sphinx-autobuild"),
        "--conf-dir",
        str(conf_dir),
        str(docs_dir),
        str(build_dir),
        *SPHINX_BUILD_OPTIONS,
        *(["--open-browser"] if open_browser else []),
        "--port",
        str(port),
        "--watch",
        str(conf_dir),
        "--ignore",
        str(DOCS_DIR / "__pycache__" / "*"),
        "--ignore",
        str(DOCS_DIR / "_build" / "*"),
    ]

    subprocess.check_call(cmd)


def ensure_unique_keys(entries):
    """Guarantee that every BibTeX entry has a unique citation key."""
    seen = set()
    for entry in entries:
        base = entry.get("ID", "no_id")
        key  = base
        suffix = 1
        while key in seen:
            key = f"{base}_{suffix}"
            suffix += 1
        entry["ID"] = key
        seen.add(key)
    return entries


def prioritize_doi_over_url(entries):
    """Remove url field when a doi is present to keep references clean."""
    for entry in entries:
        if "doi" in entry and "url" in entry:
            del entry["url"]
    return entries


def replace_latex_commands(bibtex_str):
    """Replace LaTeX macros with Unicode equivalents."""
    replacements = {
        r"\textbar": "|",
    }
    for latex_cmd, unicode_char in replacements.items():
        bibtex_str = bibtex_str.replace(latex_cmd, unicode_char)
    return bibtex_str


def export_zotero_to_bibtex(group_id, api_key=None, output_file=OUTPUT_FILE):
    """
    Fetch the Zotero group library and write it to a BibTeX file.

    Parameters
    ----------
    group_id
        Zotero Group ID (visible in the group URL on zotero.org).
    api_key
        Zotero API key. None works for public groups.
    output_file
        Destination path for the generated .bib file.
    """
    if group_id is None:
        raise RuntimeError(
            "Set 'zotero_group_id' in docs_build_config in conf.py before "
            "building the bibliography."
        )

    # These optional dependencies are only needed when bibliography export is
    # requested; ordinary documentation builds do not import them.
    from pyzotero import zotero
    import bibtexparser
    from bibtexparser.bwriter import BibTexWriter

    zot = zotero.Zotero(group_id, "group", api_key)

    bibtex_db = zot.everything(zot.items(format="bibtex"))
    ensure_unique_keys(bibtex_db.entries)
    prioritize_doi_over_url(bibtex_db.entries)

    db = bibtexparser.bibdatabase.BibDatabase()
    db.entries = bibtex_db.entries

    writer = BibTexWriter()
    bibtex_str = bibtexparser.dumps(db, writer)
    bibtex_str = replace_latex_commands(bibtex_str)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(bibtex_str)


def build_bibliography():
    """Regenerate the configured bibliography from Zotero."""
    export_zotero_to_bibtex(GROUP_ID, API_KEY, OUTPUT_FILE)
    print(f"Bibliography written to: {OUTPUT_FILE}")


def main():
    """Parse command-line arguments and run the documentation workflow."""
    parser = argparse.ArgumentParser(
        description="Build or live-preview the Sphinx documentation."
    )

    parser.add_argument(
        "--no-autobuild",
        action="store_true",
        help="Use sphinx-build instead of sphinx-autobuild.",
    )

    parser.add_argument(
        "--no-open-browser",
        action="store_true",
        help="Start live preview without opening a browser.",
    )

    parser.add_argument(
        "--clean-only",
        action="store_true",
        help="Delete the build directory and exit.",
    )

    parser.add_argument(
        "--force-apidoc",
        action="store_true",
        help="Force sphinx-apidoc to overwrite existing stubs.",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for sphinx-autobuild (default: 8000).",
    )

    parser.add_argument(
        "--build-bib",
        action="store_true",
        help="Regenerate bibliography from Zotero before building.",
    )

    parser.add_argument(
        "--build-bib-only",
        action="store_true",
        help="Regenerate bibliography from Zotero and exit without building.",
    )

    args = parser.parse_args()

    # Verify the Python environment before launching any external tools.
    check_environment()

    if args.build_bib_only:
        build_bibliography()
        raise SystemExit(0)

    # Always start from a clean Sphinx build directory.
    delete_build()

    if args.clean_only:
        raise SystemExit(0)

    if args.build_bib:
        build_bibliography()

    run_sphinx_apidoc(
        output_dir=API_OUTPUT_DIR,
        src_dir=SRC_DIR,
        exclude=EXCLUDE_MODULES,
        force=args.force_apidoc,
    )

    if args.no_autobuild:
        run_sphinx_build(
            docs_dir=SOURCE_DIR,
            conf_dir=DOCS_DIR,
            build_dir=BUILD_DIR,
        )
    else:
        run_sphinx_autobuild(
            docs_dir=SOURCE_DIR,
            conf_dir=DOCS_DIR,
            build_dir=BUILD_DIR,
            port=args.port,
            open_browser=not args.no_open_browser,
        )


if __name__ == "__main__":
    main()