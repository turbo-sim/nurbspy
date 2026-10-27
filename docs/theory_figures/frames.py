"""Frenet frames and torsion of an exact spatial cubic Bézier curve."""

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import nurbspy as nrb

from .common import BLUE, ORANGE, axis3d, save


def generate(output_dir):
    # Exact power curve C(u)=(2u, 2u², 1.5u³), represented in Bernstein form.
    curve = nrb.NurbsCurve(np.array([[0., 2/3, 4/3, 2.],
                                     [0., 0., 2/3, 2.],
                                     [0., 0., 0., 1.5]]))
    u = np.linspace(0., 1., 500)
    points = curve.get_value(u)
    fig = plt.figure(figsize=(10.4, 4.3), layout="constrained")
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    curve.plot_curve(fig, ax, color="0.2", linewidth=2)
    # Use the repository's plotting implementation, including its frame scale.
    curve.plot_frenet_serret(fig, ax, frame_number=5, frame_scale=0.085)
    frame_u = np.linspace(0., 1., 5)
    anchors = curve.get_value(frame_u)
    frame_scale = 0.085 * curve.get_arclength()
    arrow_tips = [anchors + frame_scale * method(frame_u) for method in
                  (curve.get_tangent, curve.get_normal, curve.get_binormal)]
    axis3d(ax, np.column_stack([points, *arrow_tips]), elev=24, azim=-58)
    ax.set_title("(a) Frames computed along a spatial Bézier curve")
    ax.legend(handles=[Line2D([], [], color=c, label=l) for c, l in
                       (("red", "$T$"), ("blue", "$N$"), ("green", "$B$"))],
              loc="upper left", ncol=3, handlelength=1)
    ax2 = fig.add_subplot(1, 2, 2)
    curvature, torsion = curve.get_curvature(u), curve.get_torsion(u)
    ax2.plot(u, curvature, color=BLUE, label="Curvature $\\kappa$")
    ax2.plot(u, torsion, color=ORANGE, linestyle="--", label="Torsion $\\tau$")
    ax2.set(xlabel="Parameter $u$", ylabel="Inverse length",
            title="(b) Bending and twisting are different quantities", ylim=(0, None))
    ax2.legend()
    save(fig, output_dir, "curve_frames")
    tangent, normal, binormal = (curve.get_tangent(u), curve.get_normal(u), curve.get_binormal(u))
    basis = np.stack((tangent, normal, binormal), axis=1)
    gram = np.einsum("din,djn->ijn", basis, basis)
    frame_error = np.max(abs(gram - np.eye(3)[:, :, None]))
    expected = np.vstack((2*u, 2*u**2, 1.5*u**3))
    np.testing.assert_allclose(points, expected, atol=2e-14)
    assert frame_error < 1e-12
    # Third derivative gives the numerator (C' x C'') . C''' = 72.
    first = np.vstack((2*np.ones_like(u), 4*u, 4.5*u**2))
    second = np.vstack((np.zeros_like(u), 4*np.ones_like(u), 9*u))
    cross = np.cross(first, second, axis=0)
    expected_torsion = 72 / np.sum(cross**2, axis=0)
    np.testing.assert_allclose(torsion, expected_torsion, rtol=2e-13)
    return {"frame_orthonormality_error": float(frame_error),
            "torsion_reference_error": float(np.max(abs(torsion - expected_torsion)))}
