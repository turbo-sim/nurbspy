"""Export a closed torus STL, then edit its original NURBS construction.

Reload and validate the STL using Trimesh. The editor opens one construction
with its control net and background grid. STL coordinates are millimetres;
the file records the initial geometry before interactive edits.
"""

import argparse
from pathlib import Path

import numpy as np
import trimesh
import nurbspy as nrb

QUALITY_LEVEL = "print"  # STL mesh sampling: "preview", "print", or "fine".
OUTPUT = Path(__file__).resolve().parent / "output" / Path(__file__).stem
# Samples in each STL parameter direction; these do not control display quality.
QUALITY = {"preview": (33, 17), "print": (129, 65), "fine": (257, 129)}


def make_torus(major_radius=30.0, minor_radius=10.0):
    """Exact ring torus: major radius is measured to the tube centreline [mm]."""
    if (
        not np.isfinite([major_radius, minor_radius]).all()
        or not major_radius > minor_radius > 0
    ):
        raise ValueError("Use finite radii with major_radius > minor_radius > 0")
    profile = nrb.CircularArc(
        nrb.Point(major_radius, 0, 0),
        np.array([1.0, 0, 0]),
        np.array([0.0, 0, 1]),
        minor_radius,
        0,
        2 * np.pi,
    ).NurbsCurve
    return nrb.NurbsSurfaceRevolution(
        profile,
        nrb.Point(0, 0, 0),
        np.array([0.0, 0, 1]),
        0,
        2 * np.pi,
    ).NurbsSurface


def run_demo(output_dir, quality="print", show=True):
    """Write one closed STL and return its editable NURBS figure."""
    settings = QUALITY[quality]
    patches = [make_torus()]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"torus_{quality}.stl"
    # The exporter merges coincident seam vertices with its scale-relative default.
    options = dict(nu=settings[0], nv=settings[1], require_closed=True)
    patches[0].export(path, **options)
    mesh = trimesh.load_mesh(path)
    if not mesh.is_watertight or not mesh.is_volume:
        raise RuntimeError(
            "The reloaded STL must be a closed, consistently oriented volume"
        )
    dimensions = np.ptp(mesh.bounds, axis=0)
    print(
        f"{path.name}: {len(mesh.faces):,} triangles; {path.stat().st_size / 1024:.1f} KiB; "
        f"size {dimensions.round(2)} mm; volume {mesh.volume:.1f} mm^3"
    )
    figure = nrb.Figure(
        patches[0],
        name="Torus patch 1",
        title="Torus: NURBS construction for STL export",
        display=dict(control_points=True, control_polygon=True, grid=True),
    )
    for index, patch in enumerate(patches[1:], start=2):
        figure.add(patch, name=f"Torus patch {index}")
    if show:
        figure.edit()
    print("STL has no units: select millimetres in your slicer.")
    return figure


def main(show=True):
    return run_demo(OUTPUT, quality=QUALITY_LEVEL, show=show)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quality", choices=QUALITY, default=QUALITY_LEVEL)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument(
        "--no-show", action="store_true", help="Export without opening the editor"
    )
    args = parser.parse_args()
    run_demo(args.output_dir, quality=args.quality, show=not args.no_show)
