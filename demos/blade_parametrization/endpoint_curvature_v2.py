"""Example showing how to create quadratic rational Bézier curves with a prescribed endpoint radius."""

# -------------------------------------------------------------------------------------------------------------------- #
# Importing packages
# -------------------------------------------------------------------------------------------------------------------- #
import numpy as np
import nurbspy.jax as nrb
import matplotlib.pyplot as plt


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

    return np.array([x1, y1])


def build_control_points(x0, y0, x2, y2, theta_0, theta_2):
    """Build the quadratic Bézier control points P0, P1, P2."""
    p1 = compute_intersection_control_point(x0, y0, x2, y2, theta_0, theta_2)
    P = np.asarray(
        [
            [x0, p1[0], x2],
            [y0, p1[1], y2],
        ]
    )
    return P


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

x2 = 0.2
y2 = 0.2

theta_0 = np.radians(90.0)
theta_2 = np.radians(00.0)

target_radii = [0.5, 1.0, 1.5]


# -------------------------------------------------------------------------------------------------------------------- #
# Build common control polygon
# -------------------------------------------------------------------------------------------------------------------- #
P = build_control_points(x0, y0, x2, y2, theta_0, theta_2)


# -------------------------------------------------------------------------------------------------------------------- #
# Create curves and plot
# -------------------------------------------------------------------------------------------------------------------- #
fig, ax = plt.subplots(figsize=(8, 6))

# Plot control polygon once
ax.plot(P[0, :], P[1, :], "k--", marker="o", label="Control polygon")

for R in target_radii:
    bezier, W = create_bezier_with_endpoint_radius(P, R)
    C = evaluate_bezier_curve(bezier, num=400)

    R_check = compute_endpoint_radius(P, W)

    print(f"Input R:   {R:.4f}")
    print(f"Weight w1: {W[1]:.6f}")
    print(f"Checked R: {R_check:.6f}")
    print()

    ax.plot(C[0, :], C[1, :], label=f"R = {R:.1f}")


# -------------------------------------------------------------------------------------------------------------------- #
# Plot formatting
# -------------------------------------------------------------------------------------------------------------------- #
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Quadratic rational Bézier curves with prescribed endpoint radius")
ax.axis("equal")
ax.grid(True)
ax.legend()

plt.show()
