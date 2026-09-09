"""
generate_example_images.py
===========================
Regenerate the documentation example images from demos/documentation/*.py.

Each script under demos/documentation/ ends with plt.show() for interactive
use. This tool runs every script with a non-interactive Matplotlib backend,
captures the figure(s) it produces, and saves them as the PNGs referenced by
docs/source/examples/*.md -- so the committed images always reflect the
current behavior of the scripts that generate them, instead of drifting out
of sync silently.

Run this script using Poetry, from the project root.


Most common command
--------------------
Regenerate every image in docs/source/examples/images:

    poetry run python docs/generate_example_images.py


Other useful commands
----------------------

Check whether the committed images are still in sync with the demo scripts,
without overwriting anything. Exits with status 1 if any image is missing
or differs:

    poetry run python docs/generate_example_images.py --check

Regenerate (or check) a single example by script name:

    poetry run python docs/generate_example_images.py --only circular_arc
    poetry run python docs/generate_example_images.py --check --only circular_arc
"""

import argparse
import runpy
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DOCS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DOCS_DIR.parent
DEMOS_DIR = PROJECT_ROOT / "demos" / "documentation"
IMAGES_DIR = DOCS_DIR / "source" / "examples" / "images"

# Fixed rendering settings so every image is produced the same way,
# regardless of who runs this script or when.
SAVEFIG_KWARGS = dict(dpi=200, bbox_inches="tight", pad_inches=0.1)


def discover_scripts(only=None):
    """Return the demo scripts to process, sorted by name."""
    scripts = sorted(DEMOS_DIR.glob("*.py"))

    if only:
        wanted = set(only)
        scripts = [script for script in scripts if script.stem in wanted]
        missing = wanted - {script.stem for script in scripts}
        if missing:
            raise SystemExit(f"Unknown example(s): {', '.join(sorted(missing))}")

    return scripts


def render_script(script, out_dir):
    """
    Run one demo script and save the figure(s) it produces as PNGs.

    Scripts are executed with plt.show() replaced by a no-op, so their
    interactive behavior is unaffected when run normally. Multiple figures
    from the same script are numbered (name_0.png, name_1.png, ...); a
    script with a single figure keeps the plain "name.png".

    Returns the list of PNG paths written, in figure order.
    """
    plt.close("all")
    runpy.run_path(str(script), run_name="__main__")

    fignums = plt.get_fignums()
    if not fignums:
        raise RuntimeError(f"{script.name} did not produce any figure.")

    written = []
    for i, num in enumerate(fignums):
        suffix = "" if len(fignums) == 1 else f"_{i}"
        out_path = out_dir / f"{script.stem}{suffix}.png"
        plt.figure(num).savefig(out_path, **SAVEFIG_KWARGS)
        written.append(out_path)

    plt.close("all")
    return written


def images_differ(path_a, path_b):
    """True if two PNGs differ in size or pixel content."""
    a = np.asarray(Image.open(path_a).convert("RGB"))
    b = np.asarray(Image.open(path_b).convert("RGB"))
    return a.shape != b.shape or not np.array_equal(a, b)


def check(scripts):
    """Render every script to a temp dir and diff against the committed images."""
    stale = []

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        for script in scripts:
            print(f"Rendering {script.stem} ...")
            for rendered in render_script(script, tmp_dir):
                committed = IMAGES_DIR / rendered.name
                if not committed.exists():
                    print(f"  MISSING: {rendered.name} is not committed yet.")
                    stale.append(rendered.name)
                elif images_differ(rendered, committed):
                    print(f"  OUT OF DATE: {rendered.name}")
                    stale.append(rendered.name)

    if stale:
        print(f"\n{len(stale)} image(s) out of date:")
        for name in stale:
            print(f"  - {name}")
        print(
            "\nRun 'poetry run python docs/generate_example_images.py' "
            "to regenerate them."
        )
        raise SystemExit(1)

    print("\nAll example images are up to date.")


def generate(scripts):
    """Render every script and overwrite the committed images."""
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    for script in scripts:
        print(f"Rendering {script.stem} ...")
        for path in render_script(script, IMAGES_DIR):
            print(f"  Saved {path.relative_to(PROJECT_ROOT)}")


def main():
    parser = argparse.ArgumentParser(
        description="Regenerate documentation example images from demos/documentation/*.py."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Compare against the committed images instead of overwriting them; "
             "exits with status 1 if any are missing or out of date.",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        metavar="NAME",
        help="Only process these demo script(s) by stem, e.g. --only circular_arc.",
    )
    args = parser.parse_args()

    # Demo scripts are written for interactive use; replace plt.show() with a
    # no-op so runpy can execute them unattended.
    plt.show = lambda *args, **kwargs: None

    scripts = discover_scripts(args.only)

    if args.check:
        check(scripts)
    else:
        generate(scripts)


if __name__ == "__main__":
    main()
