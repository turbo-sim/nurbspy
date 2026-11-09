"""Export the numbered curve/surface examples as matching PNG and offline HTML.

Run from any working directory with nurbspy installed:

    python docs/export_demo_figures.py
    python docs/export_demo_figures.py --only 01_curves/01_bezier_planar

Each demo stays focused on construction and editing. This separate command calls
main(show=False) and writes to <demo folder>/output/<script name>/. PNG capture
fits the camera first; HTML then preserves that same camera and figure state.
"""

import argparse
import hashlib
import json
from pathlib import Path
import runpy
import struct

import nurbspy as nrb

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEMOS = PROJECT_ROOT / "demos"


def renderer_signature():
    """Fingerprint rendering and NumPy geometry sources once per gallery build."""
    root = PROJECT_ROOT / "src" / "nurbspy"
    # The gallery uses NumPy geometry; unrelated fitting and JAX development
    # should not force regeneration of every figure.
    files = list(root.glob("*.py"))
    for folder in ("visualization", "numpy"):
        files.extend((root / folder).rglob("*.py"))
    files.extend((root / "web_assets").rglob("*.js"))
    digest = hashlib.sha256()
    newest = 0
    for path in sorted(files):
        if not path.is_file():
            continue
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0" + path.read_bytes())
        newest = max(newest, path.stat().st_mtime_ns)
    return digest.hexdigest(), newest


def export_paths(script):
    script = Path(script).resolve()
    output = script.parent / "output" / script.stem
    return output / f"{script.stem}.png", output / f"{script.stem}.html"


def _export_stamp(script, width, height, renderer):
    return dict(
        version=1,
        script=hashlib.sha256(Path(script).read_bytes()).hexdigest(),
        renderer=renderer[0],
        exporter=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        width=width,
        height=height,
    )


def ensure_export(script, *, width=1600, height=1200, renderer=None):
    """Reuse current assets, adopting older exports once when their dates agree."""
    script = Path(script).resolve()
    renderer = renderer or renderer_signature()
    png, html = export_paths(script)
    manifest = png.parent / ".figure-export.json"
    expected = _export_stamp(script, width, height, renderer)
    present = all(path.is_file() and path.stat().st_size > 0 for path in (png, html))
    try:
        cached = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        cached = None
    if present and cached == expected:
        return png, html
    # Exports created before cache stamps were introduced can be reused if they
    # postdate their sources and have the requested image dimensions.
    if present and not manifest.exists():
        newest_source = max(script.stat().st_mtime_ns, renderer[1])
        with png.open("rb") as stream:
            header = stream.read(24)
        dimensions_match = (
            len(header) == 24
            and header[:8] == b"\x89PNG\r\n\x1a\n"
            and struct.unpack(">II", header[16:24]) == (width, height)
        )
        if (
            dimensions_match
            and min(png.stat().st_mtime_ns, html.stat().st_mtime_ns) >= newest_source
        ):
            manifest.write_text(json.dumps(expected, indent=2), encoding="utf-8")
            return png, html
    print(f"Generating gallery figure: {script.parent.name}/{script.stem}", flush=True)
    return export_script(script, width=width, height=height, renderer=renderer)


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


def export_script(script, *, width=1600, height=1200, renderer=None):
    """Build a single example and save both formats beside its source folder."""
    script = Path(script).resolve()
    stamp = _export_stamp(script, width, height, renderer or renderer_signature())
    namespace = runpy.run_path(str(script))
    figure = namespace["main"](show=False)
    if not isinstance(figure, nrb.Figure):
        raise TypeError(f"{script.name}: main(show=False) must return one Figure")
    png, html = export_paths(script)
    output = png.parent
    output.mkdir(parents=True, exist_ok=True)
    figure.savefig(png, width=width, height=height)
    figure.to_html(html)
    (output / ".figure-export.json").write_text(
        json.dumps(stamp, indent=2), encoding="utf-8"
    )
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
