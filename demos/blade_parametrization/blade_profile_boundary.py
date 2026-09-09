"""Chain four quadratic Bezier arcs into a blade profile boundary.

Five anchor points, from the leading edge to the trailing edge, and a
prescribed tangent angle at each of them define four quadratic Bezier
segments. Sharing a control point and a tangent angle at each interior
join gives position (G0) and tangent (G1) continuity by construction.

The leading-edge segment additionally has its weight tuned, with the
endpoint-radius technique from `quadratic_bezier_endpoint_radius.py`, to
match the curvature *magnitude* of its neighbor at their shared point.
The checks below verify that this also matches the curvature *vector*
(magnitude and direction, as required for true G2 continuity — see the
G2 continuity theory page) at that join, and contrast it with the two
other interior joins, which keep plain unit weights and are therefore
only G0- and G1-continuous.
"""
import numpy as np
import matplotlib.pyplot as plt
import nurbspy as nrb

nrb.set_plot_options()


def cross2d(a, b):
    """Return the 2D scalar cross product."""
    return a[0] * b[1] - a[1] * b[0]


def compute_intersection_control_point(x0, y0, x2, y2, theta_0, theta_2):
    """Middle control point P1 where the tangent rays at P0 and P2 cross."""
    A = np.array([[np.cos(theta_0), np.cos(theta_2)],
                  [np.sin(theta_0), np.sin(theta_2)]])
    b = np.array([x2 - x0, y2 - y0])
    d1, _ = np.linalg.solve(A, b)
    return x0 + d1 * np.cos(theta_0), y0 + d1 * np.sin(theta_0)


def create_bezier_with_endpoint_radius(P, R):
    """Quadratic rational Bezier curve with prescribed radius of curvature at u=1."""
    D20, D21 = P[:, 2] - P[:, 0], P[:, 2] - P[:, 1]
    w1 = np.sqrt(0.5 * R * abs(cross2d(D20, D21)) / np.linalg.norm(D21) ** 3)
    return nrb.NurbsCurve(control_points=P, weights=np.asarray([1.0, w1, 1.0]))


def endpoint_geometry(curve, u):
    """Unit tangent and curvature vector of `curve` at parameter `u`."""
    velocity = curve.get_derivative(u, 1)[:, 0]
    acceleration = curve.get_derivative(u, 2)[:, 0]
    speed = np.linalg.norm(velocity)
    tangent = velocity / speed
    curvature_vector = (acceleration - tangent * np.dot(tangent, acceleration)) / speed**2
    return tangent, curvature_vector


# -------------------------------------------------------------------------------------------------------------------- #
# Anchor points and tangent angles, leading edge (index 0) to trailing edge (index 4)
# -------------------------------------------------------------------------------------------------------------------- #
x = [0.00, 0.10, 0.30, 0.99, 1.00]
y = [0.00, 0.10, 0.125, 0.01, 0.00]
theta = np.radians([90.0, 25.0, 0.0, -15.0, 90.0])

middle = [compute_intersection_control_point(x[i], y[i], x[i + 1], y[i + 1], theta[i], theta[i + 1])
          for i in range(4)]
control_polygons = [np.asarray([[x[i], mx, x[i + 1]], [y[i], my, y[i + 1]]])
                     for i, (mx, my) in enumerate(middle)]
P_a, P_b, P_c, P_d = control_polygons

# -------------------------------------------------------------------------------------------------------------------- #
# Build the segments; try to match the leading-edge curvature to its neighbor
# -------------------------------------------------------------------------------------------------------------------- #
segments = {
    "b": nrb.NurbsCurve(control_points=P_b),
    "c": nrb.NurbsCurve(control_points=P_c),
    "d": nrb.NurbsCurve(control_points=P_d),
}
target_curvature = np.linalg.norm(endpoint_geometry(segments["b"], 0.0)[1])
segments["a"] = create_bezier_with_endpoint_radius(P_a, R=1.0 / target_curvature)

# -------------------------------------------------------------------------------------------------------------------- #
# Check continuity at the three interior joins
# -------------------------------------------------------------------------------------------------------------------- #
for left, right in [("a", "b"), ("b", "c"), ("c", "d")]:
    t_left, k_left = endpoint_geometry(segments[left], 1.0)
    t_right, k_right = endpoint_geometry(segments[right], 0.0)
    print(f"Join {left}-{right}:")
    print(f"  tangent mismatch:          {np.linalg.norm(t_left - t_right):.2e}")
    print(f"  curvature magnitude:       {np.linalg.norm(k_left):.4f} vs {np.linalg.norm(k_right):.4f}")
    print(f"  curvature-vector mismatch: {np.linalg.norm(k_left - k_right):.2e}")

# -------------------------------------------------------------------------------------------------------------------- #
# Plot the boundary and the curvature along it
# -------------------------------------------------------------------------------------------------------------------- #
u = np.linspace(0.0, 1.0, 200)
fig1, ax1 = plt.subplots(figsize=(8, 4), layout="constrained")
fig2, ax2 = plt.subplots(figsize=(8, 4), layout="constrained")
for name, color in zip("abcd", nrb.COLORS_MATLAB):
    curve = segments[name]
    C = curve.get_value(u)
    curvature = curve.get_curvature(u)
    curve.plot_control_points(fig1, ax1, color=color, linewidth=1)
    ax1.lines[-1].set_alpha(0.4)
    ax1.plot(C[0, :], C[1, :], color=color, label=f"Segment {name}")
    ax2.plot(C[0, :], curvature, color=color, label=f"Segment {name}")

ax1.set(xlabel="x", ylabel="y", title="Blade profile boundary")
ax1.axis("equal")
ax1.grid(alpha=0.2)
ax1.legend()

ax2.set(xlabel="x", ylabel="curvature", title="Curvature along the boundary")
ax2.grid(alpha=0.2)
ax2.legend()

plt.show()
