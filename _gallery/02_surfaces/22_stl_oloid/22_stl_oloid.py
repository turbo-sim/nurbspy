"""Export a closed oloid STL, then edit its original NURBS construction.

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
QUALITY = {
    "preview": (61, 3, 61),
    "print": (181, 5, 181),
    "fine": (361, 9, 361),
}


def make_oloid(radius=20.0, boundary_points=181):
    """Two ruled NURBS patches with piecewise-linear sampled boundaries [mm]."""
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("radius must be finite and positive")
    if (
        isinstance(boundary_points, (bool, np.bool_))
        or not isinstance(boundary_points, (int, np.integer))
        or boundary_points < 5
        or boundary_points % 2 == 0
    ):
        raise ValueError("boundary_points must be an odd integer of at least five")
    # Equal-radius circles in perpendicular planes; centres one radius apart.
    # Cluster samples near the ends and keep a symmetric grid for matching seams.
    theta = np.pi - 2 * np.pi / 3 * np.cos(np.linspace(0, np.pi, boundary_points))
    c, s = np.cos(theta), np.sin(theta)
    cos_phi = np.clip(c / (1 - c), -1, 1)
    sin_phi = np.sqrt(1 - cos_phi**2)
    cos_phi[[0, -1]], sin_phi[[0, -1]] = 1.0, 0.0  # exact common seam endpoints
    first = nrb.NurbsCurve(
        [nrb.Point(radius * (x - 0.5), radius * y, 0) for x, y in zip(c, s)],
        degree=1,
    )
    patches = []
    for sign in (-1, 1):
        second = nrb.NurbsCurve(
            [
                nrb.Point(radius * (0.5 + x), 0, sign * radius * z)
                for x, z in zip(cos_phi, sin_phi)
            ],
            degree=1,
        )
        patches.append(nrb.NurbsSurfaceRuled(first, second).NurbsSurface)
    return patches


def run_demo(output_dir, quality="print", show=True):
    """Write one closed STL and return its editable NURBS figure."""
    settings = QUALITY[quality]
    patches = make_oloid(boundary_points=settings[2])
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"oloid_{quality}.stl"
    # The exporter merges coincident seam vertices with its scale-relative default.
    options = dict(nu=settings[0], nv=settings[1], require_closed=True)
    nrb.export_stl(patches, path, **options)
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
        name="Oloid patch 1",
        title="Oloid: NURBS construction for STL export",
        display=dict(control_points=True, control_polygon=True, grid=True),
    )
    for index, patch in enumerate(patches[1:], start=2):
        figure.add(patch, name=f"Oloid patch {index}")
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
