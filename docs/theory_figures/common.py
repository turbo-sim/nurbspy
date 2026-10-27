"""Shared geometries and publication settings; no replacement geometry kernels."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import nurbspy as nrb
from nurbspy.graphics import style_3d_axes


BLUE, ORANGE, GREEN, PURPLE = "#0072B2", "#D55E00", "#009E73", "#7E2F8E"


def configure():
    """Start with the repository's plotting defaults, then size for a page."""
    nrb.set_plot_options(fontsize=12, linewidth=1.6, color_order="matlab")
    plt.rcParams.update({
        # These bundled fonts make the output portable across Windows and CI.
        "font.serif": ["STIXGeneral", "DejaVu Serif"],
        "axes.labelpad": 5, "axes.titlepad": 10,
        "grid.alpha": 0.22, "grid.linewidth": 0.5,
        "legend.frameon": False, "legend.fontsize": 10,
        "svg.fonttype": "none", "svg.hashsalt": "nurbspy-theory",
        "pdf.fonttype": 42, "savefig.facecolor": "white",
    })


def save(fig, output_dir, name):
    """Keep a high-resolution preview and vector originals of every figure."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "svg", "pdf"):
        metadata = {"Date": None} if extension == "svg" else None
        if extension == "pdf":
            metadata = {"CreationDate": None, "ModDate": None}
        path = output_dir / f"{name}.{extension}"
        fig.savefig(path, dpi=300,
                    bbox_inches="tight", pad_inches=0.12, metadata=metadata)
        if extension == "svg":
            # Matplotlib leaves spaces before SVG path line breaks. Preserve
            # the drawing while keeping generated files clean in Git diffs.
            lines = path.read_text(encoding="utf-8").splitlines()
            path.write_text("\n".join(line.rstrip() for line in lines) + "\n",
                            encoding="utf-8", newline="\n")
    plt.close(fig)


def axis3d(ax, points, *, elev=26, azim=-58):
    """Use physical axis proportions and a restrained, readable 3D frame."""
    style_3d_axes(ax, elev=elev, azim=azim)
    points = np.asarray(points).reshape(3, -1)
    for index, setter in enumerate((ax.set_xlim, ax.set_ylim, ax.set_zlim)):
        lo, hi = points[index].min(), points[index].max()
        pad = 0.08 * max(hi - lo, 0.5)
        setter(lo - pad, hi + pad)
    ax.set_box_aspect(np.maximum(np.ptp(points, axis=1), 0.5))
    ax.set(xlabel="$x$", ylabel="$y$", zlabel="$z$")
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.labelpad = 3
        axis.set_major_locator(MaxNLocator(nbins=3, prune="both"))
    ax.tick_params(pad=1, labelsize=9)


def engineering_curve(backend=nrb):
    """A regular, x-monotone rational cubic with two nonzero knot spans."""
    points = np.array([[0., 0.6, 2., 3.5, 4.],
                       [0., 1.8, -0.5, 2.2, 1.]])
    return backend.NurbsCurve(points, np.array([1., 0.8, 1.6, 0.9, 1.]), degree=3)


def paraboloid(backend=nrb):
    """Exact quadratic graph z=(x²+y²)/2 over [-1,1]², in NURBS form."""
    line, square = np.array([-1., 0., 1.]), np.array([1., -1., 1.])
    x, y = np.meshgrid(line, line, indexing="ij")
    z = (square[:, None] + square[None, :]) / 2
    return backend.NurbsSurface(np.stack((x, y, z)), u_degree=2, v_degree=2)


def surface_grid(surface, count=81):
    u, v = np.meshgrid(np.linspace(0., 1., count), np.linspace(0., 1., count),
                       indexing="ij")
    xyz = np.asarray(surface.get_value(u.ravel(), v.ravel())).reshape(3, count, count)
    return u, v, xyz


def torus():
    """An exact rational torus made with the package's circular/revolution APIs."""
    profile = nrb.CircularArc(
        O=np.array([2., 0., 0.]), X=np.array([1., 0., 0.]),
        Y=np.array([0., 0., 1.]), R=0.6,
        theta_start=0., theta_end=2 * np.pi,
    ).NurbsCurve
    return nrb.NurbsSurfaceRevolution(
        generatrix=profile, axis_point=np.zeros(3),
        axis_direction=np.array([0., 0., 1.]),
        angle_start=0., angle_end=2 * np.pi,
    ).NurbsSurface
