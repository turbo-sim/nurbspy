"""Export the numbered curve/surface examples as matching PNG and offline HTML.

Run from any working directory with nurbspy installed:

    python docs/export_demo_figures.py
    python docs/export_demo_figures.py --only 01_curves/01_bezier_planar

Each demo stays focused on construction and editing. This separate command calls
main(show=False) and writes to <demo folder>/output/<script name>/. PNG capture
fits the camera first; HTML then preserves that same camera and figure state.
"""

import argparse
from pathlib import Path
import runpy

import nurbspy as nrb

DEMOS = Path(__file__).resolve().parents[1] / "demos"


def discover_scripts(only=None):
    """Select numbered teaching scripts by folder/stem or an unambiguous stem."""
    scripts = [
        path
        for folder in ("01_curves", "02_surfaces")
        for path in sorted((DEMOS / folder).glob("[0-9]*.py"))
    ]
    if only:
        selected = set()
        for value in only:
            key = value.replace("\\", "/").removesuffix(".py")
            matches = [
                path
                for path in scripts
                if key in (path.stem, f"{path.parent.name}/{path.stem}")
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"Expected one demo for {value!r}; use folder/script_name"
                )
            selected.add(matches[0])
        scripts = [path for path in scripts if path in selected]
    return scripts


def export_script(script, *, width=1600, height=1200):
    """Build a single example and save both formats beside its source folder."""
    script = Path(script).resolve()
    namespace = runpy.run_path(str(script))
    figure = namespace["main"](show=False)
    if not isinstance(figure, nrb.Figure):
        raise TypeError(f"{script.name}: main(show=False) must return one Figure")
    output = script.parent / "output" / script.stem
    output.mkdir(parents=True, exist_ok=True)
    png = output / f"{script.stem}.png"
    html = output / f"{script.stem}.html"
    figure.savefig(png, width=width, height=height)
    figure.to_html(html)
    return png, html


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="+", metavar="FOLDER/SCRIPT")
    # These are output image pixels. Geometry sampling remains the default Auto.
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=1200)
    args = parser.parse_args()
    try:
        scripts = discover_scripts(args.only)
    except ValueError as error:
        parser.error(str(error))
    for script in scripts:
        print(f"Exporting {script.parent.name}/{script.stem} ...", flush=True)
        for path in export_script(script, width=args.width, height=args.height):
            print(f"  Saved {path}", flush=True)


if __name__ == "__main__":
    main()
