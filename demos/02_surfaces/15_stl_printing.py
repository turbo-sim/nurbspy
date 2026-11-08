"""Export a torus and an approximate oloid as closed STL files for 3D printing.

Run with --quality preview/print/fine. STL coordinates are intended as mm;
choose millimetres when importing them into a slicer. See the linked guide
for the distinction between geometry approximation and mesh resolution.
"""

import argparse
from pathlib import Path

import numpy as np
import trimesh
import nurbspy as nrb

QUALITY_LEVEL = "print"  # "preview", "print", or "fine" controls STL sampling.
OUTPUT = Path(__file__).resolve().parent / "output" / Path(__file__).stem


# (nu, nv) are regular samples per parameter direction, not triangle counts.
# Oloid boundary_points controls the geometry itself, independently of its mesh.
QUALITY = {
    "preview": {"torus": (33, 17), "oloid": (61, 3), "boundary_points": 61},
    "print": {"torus": (129, 65), "oloid": (181, 5), "boundary_points": 181},
    "fine": {"torus": (257, 129), "oloid": (361, 9), "boundary_points": 361},
}


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
    """Export closed STLs and compare NURBS against actual reloaded triangles."""
    settings = QUALITY[quality]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torus = make_torus()
    oloid = make_oloid(boundary_points=settings["boundary_points"])
    meshes = []
    for name, patches in (("torus", [torus]), ("oloid", oloid)):
        nu, nv = settings[name]
        path = output_dir / f"{name}_{quality}.stl"
        # None uses the scale-relative seam-merging default, about 8e-9 mm
        # for this torus. This is not a surface-accuracy tolerance.
        options = dict(nu=nu, nv=nv, require_closed=True, weld_tolerance=None)
        # Single-surface method and multi-patch helper share the same exporter.
        if name == "torus":
            torus.export(path, **options)
        else:
            nrb.export_stl(patches, path, **options)
        mesh = trimesh.load_mesh(path)
        if not mesh.is_watertight or not mesh.is_volume:
            raise RuntimeError(
                f"Reloaded {name} is not a closed, consistently oriented volume"
            )
        meshes.append(mesh)
        dimensions = np.ptp(mesh.bounds, axis=0)
        print(
            f"{path.name}: {len(mesh.faces):,} triangles; {path.stat().st_size / 1024:.1f} KiB; "
            f"size {dimensions.round(2)} mm; volume {mesh.volume:.1f} mm^3"
        )
    figures = {}
    for name, patches, mesh in (
        ("torus", [torus], meshes[0]),
        ("oloid", oloid, meshes[1]),
    ):
        # Show the actual reloaded triangles, including their wireframe.
        figure = nrb.Figure(
            patches[0], title=f"{name.title()}: reloaded STL", display=dict(grid=True)
        )
        figure.update(visible=False)
        figure.mesh(mesh.vertices, mesh.faces, wireframe=True)
        figures[name] = figure
    if show:
        for figure in figures.values():
            figure.show()
    print(
        "STL has no units: select millimetres in your slicer. Check orientation and supports there."
    )
    return figures


def main(show=True):
    return run_demo(OUTPUT, quality=QUALITY_LEVEL, show=show)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quality", choices=QUALITY, default=QUALITY_LEVEL)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Write and validate the STL files without opening a viewer",
    )
    args = parser.parse_args()
    run_demo(args.output_dir, quality=args.quality, show=not args.no_show)
