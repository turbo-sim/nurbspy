"""Example showing how to create quadratic rational Bézier curves with a prescribed endpoint radius."""

# -------------------------------------------------------------------------------------------------------------------- #
# Importing packages
# -------------------------------------------------------------------------------------------------------------------- #
import numpy as np
import nurbspy.jax as nrb
import matplotlib.pyplot as plt


from nurbspy import set_plot_options

set_plot_options()



def cross2d(a, b):
    """Return the 2D scalar cross product."""
    return a[0] * b[1] - a[1] * b[0]


def compute_intersection_control_point(x0, y0, x2, y2, theta_0, theta_2):
    """
    Compute the middle control point P1 as the intersection of:
    - the ray starting at P0 with angle theta_0
    - the ray ending at P2 with tangent angle theta_2
    """
    A = np.array(
        [
            [np.cos(theta_0), np.cos(theta_2)],
            [np.sin(theta_0), np.sin(theta_2)],
        ]
    )
    b = np.array([x2 - x0, y2 - y0])

    d1, d2 = np.linalg.solve(A, b)

    x1 = x0 + d1 * np.cos(theta_0)
    y1 = y0 + d1 * np.sin(theta_0)

    x1_bis = x2 - d2 * np.cos(theta_2)
    y1_bis = y2 - d2 * np.sin(theta_2)

    # Optional consistency check
    if not np.allclose([x1, y1], [x1_bis, y1_bis]):
        raise ValueError("The two constructions of P1 do not match.")

    return x1, y1


def compute_middle_weight_for_endpoint_radius(P, R):
    """
    Compute the middle weight w1 for a quadratic rational Bézier curve
    with weights [1, w1, 1] so that the endpoint radius at u = 1 is R.
    """
    D20 = P[:, 2] - P[:, 0]  # P2 - P0
    D21 = P[:, 2] - P[:, 1]  # P2 - P1

    w1 = np.sqrt(0.5 * R * abs(cross2d(D20, D21)) / np.linalg.norm(D21) ** 3)
    return w1


def compute_endpoint_curvature(P, W):
    """
    Compute the analytic endpoint curvature at u = 1 for a quadratic rational Bézier curve.
    """
    p = 2
    D20 = P[:, 2] - P[:, 0]  # P2 - P0
    D21 = P[:, 2] - P[:, 1]  # P2 - P1

    curvature = (
        (p - 1)
        / p
        * (W[2] * W[0])
        / W[1] ** 2
        * abs(cross2d(D20, D21))
        / np.linalg.norm(D21) ** 3
    )
    return curvature


def compute_endpoint_radius(P, W):
    """Compute the analytic endpoint radius at u = 1."""
    return 1.0 / compute_endpoint_curvature(P, W)


def create_bezier_with_endpoint_radius(P, R):
    """
    Create a quadratic rational Bézier curve with prescribed endpoint radius at u = 1.
    """
    w1 = compute_middle_weight_for_endpoint_radius(P, R)
    W = np.asarray([1.0, w1, 1.0])
    bezier = nrb.NurbsCurve(control_points=P, weights=W)
    return bezier, W


def evaluate_bezier_curve(bezier, num=300):
    """Sample a Bézier/NURBS curve for plotting."""
    u = np.linspace(0.0, 1.0, num)
    C = bezier.get_value(u)
    return C


# -------------------------------------------------------------------------------------------------------------------- #
# Input data
# -------------------------------------------------------------------------------------------------------------------- #
x0 = 0.00
y0 = 0.00
theta_0 = np.radians(90.0)

x2 = 0.1
y2 = 0.1
theta_2 = np.radians(25.0)

x4 = 0.3
y4 = 0.125
theta_4 = np.radians(0.0)

x6 = 0.99
y6 = 0.01
theta_6 = np.radians(-15.0)

x8 = 1.00
y8 = 0.00
theta_8 = np.radians(90.0)


x1, y1 = compute_intersection_control_point(x0, y0, x2, y2, theta_0, theta_2)
x3, y3 = compute_intersection_control_point(x2, y2, x4, y4, theta_2, theta_4)
x5, y5 = compute_intersection_control_point(x4, y4, x6, y6, theta_4, theta_6)
x7, y7 = compute_intersection_control_point(x6, y6, x8, y8, theta_6, theta_8)


P_a = np.asarray([[x0, y0], [x1, y1], [x2, y2]]).T
P_b = np.asarray([[x2, y2], [x3, y3], [x4, y4]]).T
P_c = np.asarray([[x4, y4], [x5, y5], [x6, y6]]).T
P_d = np.asarray([[x6, y6], [x7, y7], [x8, y8]]).T




# B_a = nrb.NurbsCurve(control_points=P_a, weights=[1.0, 1.0, 1.0])
B_b = nrb.NurbsCurve(control_points=P_b, weights=[1.0, 1.0, 1.0])
k_end = np.squeeze(B_b.get_curvature(0.0))


# This does not create a bezier curve with the specified leading edge radius of curvature
B_a, W_a = create_bezier_with_endpoint_radius(P_a, R=1.0 / k_end)


B_c = nrb.NurbsCurve(control_points=P_c, weights=[1.0, 1.0, 1.0])
B_d = nrb.NurbsCurve(control_points=P_d, weights=[1.0, 1.0, 1.0])


P_total = np.hstack([P_a, P_b, P_c, P_d])
W_total = np.hstack([B_a.W, B_b.W, B_c.W, B_d.W])
B_total = nrb.NurbsCurve(control_points=P_total, weights=W_total, degree=4)

print(f"Start point curvature: {B_total.get_curvature(0.0)}")

# -------------------------------------------------------------------------------------------------------------------- #
# Create curves and plot
# -------------------------------------------------------------------------------------------------------------------- #
fig, ax = plt.subplots(figsize=(8, 6))


B_a.plot(control_points=True, fig=fig, ax=ax)
B_b.plot(control_points=True, fig=fig, ax=ax)
B_c.plot(control_points=True, fig=fig, ax=ax)
B_d.plot(control_points=True, fig=fig, ax=ax)

B_total.plot(control_points=False, fig=fig, ax=ax)


u = np.linspace(0.0, 1.0, 100)

C_a = B_a.get_value(u)
C_b = B_b.get_value(u)
C_c = B_c.get_value(u)
C_d = B_d.get_value(u)

k_a = B_a.get_curvature(u)
k_b = B_b.get_curvature(u)
k_c = B_c.get_curvature(u)
k_d = B_d.get_curvature(u)

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(C_a[0, :], C_a[1, :], label="B_a")
ax.plot(C_b[0, :], C_b[1, :], label="B_b")
ax.plot(C_c[0, :], C_c[1, :], label="B_c")
ax.plot(C_d[0, :], C_d[1, :], label="B_d")
ax.legend()
ax.axis("equal")
ax.grid(True)
# ax.legend()
fig.tight_layout(pad=1)


fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(C_a[0, :], k_a, label="B_a")
ax.plot(C_b[0, :], k_b, label="B_b")
ax.plot(C_c[0, :], k_c, label="B_c")
ax.plot(C_d[0, :], k_d, label="B_d")
ax.legend()

# ax.legend()
fig.tight_layout(pad=1)

# ax.plot(
#     [x0, x1, x2, x3, x4, x5, x6, x7, x8],
#     [y0, y1, y2, y3, y4, y5, y6, y7, y8],
#     "k--",
#     marker="o",
#     label="Control polygon",
# )

# # Plot control polygon once
# ax.plot(P[0, :], P[1, :], "k--", marker="o", label="Control polygon")

# for R in target_radii:
#     bezier, W = create_bezier_with_endpoint_radius(P, R)
#     C = evaluate_bezier_curve(bezier, num=400)

#     R_check = compute_endpoint_radius(P, W)

#     print(f"Input R:   {R:.4f}")
#     print(f"Weight w1: {W[1]:.6f}")
#     print(f"Checked R: {R_check:.6f}")
#     print()

#     ax.plot(C[0, :], C[1, :], label=f"R = {R:.1f}")


# # -------------------------------------------------------------------------------------------------------------------- #
# # Plot formatting
# # -------------------------------------------------------------------------------------------------------------------- #
# ax.set_xlabel("x")
# ax.set_ylabel("y")
# ax.set_title("Quadratic rational Bézier curves with prescribed endpoint radius")

plt.show()
