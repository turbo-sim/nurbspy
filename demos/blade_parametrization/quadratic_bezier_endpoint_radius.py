"""Build a quadratic rational Bezier arc with a prescribed endpoint radius of curvature.

Two endpoints and their tangent directions fix the control polygon of a
quadratic Bezier curve. The middle control point's weight can then be tuned,
without moving any control point, to give the curve a prescribed radius of
curvature at one endpoint. This is the closed-form endpoint-curvature formula
for rational Bezier curves derived on the NURBS theory page, specialized to
degree n=2.
"""
import numpy as np
import matplotlib.pyplot as plt
import nurbspy as nrb

nrb.set_plot_options()


def cross2d(a, b):
    """Return the 2D scalar cross product."""
    return a[0] * b[1] - a[1] * b[0]


def compute_intersection_control_point(x0, y0, x2, y2, theta_0, theta_2):
    """
    Compute the middle control point P1 as the intersection of:
    - the ray starting at P0 with angle theta_0
    - the ray ending at P2 with tangent angle theta_2
    """
    A = np.array([[np.cos(theta_0), np.cos(theta_2)],
                  [np.sin(theta_0), np.sin(theta_2)]])
    b = np.array([x2 - x0, y2 - y0])
    d1, d2 = np.linalg.solve(A, b)

    x1 = x0 + d1 * np.cos(theta_0)
    y1 = y0 + d1 * np.sin(theta_0)
    x1_check = x2 - d2 * np.cos(theta_2)
    y1_check = y2 - d2 * np.sin(theta_2)
    if not np.allclose([x1, y1], [x1_check, y1_check]):
        raise ValueError("The two constructions of P1 do not match.")

    return x1, y1


def build_control_points(x0, y0, x2, y2, theta_0, theta_2):
    """Build the quadratic Bezier control points P0, P1, P2."""
    x1, y1 = compute_intersection_control_point(x0, y0, x2, y2, theta_0, theta_2)
    return np.asarray([[x0, x1, x2], [y0, y1, y2]])


def compute_middle_weight_for_endpoint_radius(P, R):
    """
    Compute the middle weight w1 for a quadratic rational Bezier curve with
    weights [1, w1, 1] so that the radius of curvature at u=1 equals R.
    """
    D20 = P[:, 2] - P[:, 0]  # P2 - P0
    D21 = P[:, 2] - P[:, 1]  # P2 - P1
    return np.sqrt(0.5 * R * abs(cross2d(D20, D21)) / np.linalg.norm(D21) ** 3)


def compute_endpoint_curvature(P, W):
    """Analytic curvature at u=1 for a quadratic (degree n=2) rational Bezier curve."""
    n = 2
    D20 = P[:, 2] - P[:, 0]  # P2 - P0
    D21 = P[:, 2] - P[:, 1]  # P2 - P1
    return (n - 1) / n * (W[2] * W[0]) / W[1] ** 2 * abs(cross2d(D20, D21)) / np.linalg.norm(D21) ** 3


def create_bezier_with_endpoint_radius(P, R):
    """Quadratic rational Bezier curve with prescribed radius of curvature at u=1."""
    w1 = compute_middle_weight_for_endpoint_radius(P, R)
    W = np.asarray([1.0, w1, 1.0])
    return nrb.NurbsCurve(control_points=P, weights=W), W


# -------------------------------------------------------------------------------------------------------------------- #
# Demo: same control polygon, several prescribed endpoint radii
# -------------------------------------------------------------------------------------------------------------------- #
x0, y0, theta_0 = 0.00, 0.00, np.radians(90.0)
x2, y2, theta_2 = 0.20, 0.20, np.radians(0.0)
target_radii = [0.5, 1.0, 1.5]

P = build_control_points(x0, y0, x2, y2, theta_0, theta_2)
u = np.linspace(0.0, 1.0, 400)

fig, ax = plt.subplots(figsize=(6, 5), layout="constrained")
ax.plot(P[0, :], P[1, :], "k--", marker="o", label="Control polygon")
for R, color in zip(target_radii, nrb.COLORS_MATLAB):
    bezier, W = create_bezier_with_endpoint_radius(P, R)
    C = bezier.get_value(u)
    R_check = 1.0 / compute_endpoint_curvature(P, W)
    print(f"Target R = {R:.4f}  ->  w1 = {W[1]:.6f}, checked R = {R_check:.6f}")
    ax.plot(C[0, :], C[1, :], color=color, label=f"R = {R:.1f}")

ax.set(xlabel="x", ylabel="y", title="Quadratic rational Bezier with prescribed endpoint radius")
ax.axis("equal")
ax.grid(alpha=0.2)
ax.legend()
plt.show()
