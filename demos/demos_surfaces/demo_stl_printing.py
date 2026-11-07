"""Create, plot, and export a torus and an approximate oloid for 3D printing.

Run with --quality preview/print/fine. STL coordinates are intended as mm;
choose millimetres when importing them into a slicer. See the linked guide
for the distinction between geometry approximation and mesh resolution.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import trimesh
import nurbspy as nrb


# (nu, nv) are regular samples per parameter direction, not triangle counts.
# Oloid boundary_points controls the geometry itself, independently of its mesh.
QUALITY = {
    "preview": {"torus": (33, 17), "oloid": (61, 3), "boundary_points": 61},
    "print": {"torus": (129, 65), "oloid": (181, 5), "boundary_points": 181},
    "fine": {"torus": (257, 129), "oloid": (361, 9), "boundary_points": 361},
}


def make_torus(major_radius=30., minor_radius=10.):
    """Exact ring torus: major radius is measured to the tube centreline [mm]."""
    if not np.isfinite([major_radius, minor_radius]).all() or not major_radius > minor_radius > 0:
        raise ValueError("Use finite radii with major_radius > minor_radius > 0")
    profile = nrb.CircularArc(
        nrb.Point(major_radius, 0, 0), np.array([1., 0, 0]),
        np.array([0., 0, 1]), minor_radius, 0, 2 * np.pi,
    ).NurbsCurve
    return nrb.NurbsSurfaceRevolution(
        profile, nrb.Point(0, 0, 0), np.array([0., 0, 1]), 0, 2 * np.pi,
    ).NurbsSurface


def make_oloid(radius=20., boundary_points=181):
    """Two ruled NURBS patches with piecewise-linear sampled boundaries [mm]."""
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("radius must be finite and positive")
    if isinstance(boundary_points, (bool, np.bool_)) or not isinstance(boundary_points, (int, np.integer)) or boundary_points < 5 or boundary_points % 2 == 0:
        raise ValueError("boundary_points must be an odd integer of at least five")
    # Equal-radius circles in perpendicular planes; centres one radius apart.
    # Cluster samples near the ends and keep a symmetric grid for matching seams.
    theta = np.pi - 2 * np.pi / 3 * np.cos(np.linspace(0, np.pi, boundary_points))
    c, s = np.cos(theta), np.sin(theta)
    cos_phi = np.clip(c / (1 - c), -1, 1)
    sin_phi = np.sqrt(1 - cos_phi**2)
    cos_phi[[0, -1]], sin_phi[[0, -1]] = 1., 0.  # exact common seam endpoints
    first = nrb.NurbsCurve(
        [nrb.Point(radius * (x - .5), radius * y, 0) for x, y in zip(c, s)], degree=1,
    )
    patches = []
    for sign in (-1, 1):
        second = nrb.NurbsCurve(
            [nrb.Point(radius * (.5 + x), 0, sign * radius * z)
             for x, z in zip(cos_phi, sin_phi)], degree=1,
        )
        patches.append(nrb.NurbsSurfaceRuled(first, second).NurbsSurface)
    return patches


def plot_comparison(fig, row, name, patches, mesh):
    """Native package plotting beside the triangles actually read from the STL."""
    left = fig.add_subplot(2, 2, 2 * row + 1, projection="3d")
    right = fig.add_subplot(2, 2, 2 * row + 2, projection="3d")
    for patch in patches:
        patch.plot_surface(fig, left, color="#278c93", alpha=.92, Nu=73, Nv=41)
        patch.plot_isocurve_u(fig, left, np.linspace(0, 1, 13), color="#24545f", linewidth=.45)
        if name == "Torus":
            patch.plot_isocurve_v(fig, left, np.linspace(0, 1, 9), color="#24545f", linewidth=.45)
    right.plot_trisurf(*mesh.vertices.T, triangles=mesh.faces, color="#bad0da",
                      edgecolor="#5d7482", linewidth=.12, antialiased=True)
    left.set_title(f"{name} | " + ("exact NURBS" if name == "Torus" else "sampled ruled patches"), loc="left", fontsize=11, weight="bold")
    right.set_title(f"Reloaded STL | {len(mesh.faces):,} triangles", loc="left", fontsize=11, weight="bold")
    center = mesh.bounds.mean(axis=0)
    half_span = np.ptp(mesh.bounds, axis=0).max() * .55
    for ax in (left, right):
        ax.set(xlim=center[0] + np.array([-half_span, half_span]),
               ylim=center[1] + np.array([-half_span, half_span]),
               zlim=center[2] + np.array([-half_span, half_span]))
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=26, azim=-55)
        ax.set_xlabel("x [mm]", fontsize=9)
        ax.set_ylabel("y [mm]", fontsize=9)
        ax.set_zlabel("z [mm]", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(False)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.pane.fill = False


def run_demo(output_dir, quality="print", show=True):
    """Export both objects, validate their reloaded meshes, and save one figure."""
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
            raise RuntimeError(f"Reloaded {name} is not a closed, consistently oriented volume")
        meshes.append(mesh)
        dimensions = np.ptp(mesh.bounds, axis=0)
        print(f"{path.name}: {len(mesh.faces):,} triangles; {path.stat().st_size / 1024:.1f} KiB; "
              f"size {dimensions.round(2)} mm; volume {mesh.volume:.1f} mm^3")
    with plt.rc_context({"font.family": "DejaVu Sans", "figure.facecolor": "white"}):
        fig = plt.figure(figsize=(12, 9), layout="constrained")
        fig.suptitle(f"NURBS to STL | {quality} quality", fontsize=16, weight="bold")
        plot_comparison(fig, 0, "Torus", [torus], meshes[0])
        plot_comparison(fig, 1, "Oloid", oloid, meshes[1])
        figure_path = output_dir / f"stl_printing_{quality}.png"
        fig.savefig(figure_path, dpi=180)
        if show:
            plt.show()
        else:
            plt.close(fig)
    print(f"Plot: {figure_path.resolve()}")
    print("STL has no units: select millimetres in your slicer. Check orientation and supports there.")
    return figure_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quality", choices=QUALITY, default="print")
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parents[2] / "output" / "stl_printing")
    parser.add_argument("--no-show", action="store_true", help="Save the plot without opening a window")
    args = parser.parse_args()
    run_demo(args.output_dir, quality=args.quality, show=not args.no_show)
