"""Join two cubic Bezier segments with G2 continuity and different speeds."""
import numpy as np
import matplotlib.pyplot as plt
import nurbspy as nrb

nrb.set_plot_options()

PA = np.array([[-2., -1., -1., 0.],
               [ 1.,  1.,  0., 0.]])
left = nrb.NurbsCurve(control_points=PA)
join = left.get_value(1.)[:, 0]
d1 = left.get_derivative(1., 1)[:, 0]
d2 = left.get_derivative(1., 2)[:, 0]
speed_ratio, mu = 0.6, 0.4

# For cubic Bezier: C'(0)=3(P1-P0), C''(0)=6(P2-2P1+P0).
PB = np.empty((2, 4))
PB[:, 0] = join
PB[:, 1] = join + speed_ratio * d1 / 3
PB[:, 2] = (speed_ratio**2 * d2 + mu * d1) / 6 + 2 * PB[:, 1] - PB[:, 0]
PB[:, 3] = [2., 1.]
right = nrb.NurbsCurve(control_points=PB)


def endpoint_geometry(curve, u):
    velocity = curve.get_derivative(u, 1)[:, 0]
    acceleration = curve.get_derivative(u, 2)[:, 0]
    speed = np.linalg.norm(velocity)
    tangent = velocity / speed
    curvature_vector = (acceleration - tangent * np.dot(tangent, acceleration)) / speed**2
    return speed, tangent, curvature_vector


sa, ta, ka = endpoint_geometry(left, 1.)
sb, tb, kb = endpoint_geometry(right, 0.)
print(f"Endpoint speeds: {sa:.6f}, {sb:.6f}")
print("Curvature vectors:", ka.round(6), kb.round(6))
print(f"Tangent mismatch: {np.linalg.norm(ta - tb):.2e}")
print(f"Curvature-vector mismatch: {np.linalg.norm(ka - kb):.2e}")

fig, ax = plt.subplots(figsize=(8, 3.5), layout="constrained")
for curve, name, color in [(left, "Segment A", nrb.COLORS_MATLAB[0]),
                            (right, "Segment B", nrb.COLORS_MATLAB[1])]:
    curve.plot_curve(fig, ax, color=color, linewidth=2)
    ax.lines[-1].set_label(name)
    curve.plot_control_points(fig, ax, color=color, linewidth=1)
    ax.lines[-1].set_alpha(0.4)
ax.scatter(*join, color="black", zorder=5, label="G2 join")
ax.set(xlabel="x", ylabel="y", title="Curvature continuity with different parameter speeds")
ax.set_aspect("equal", adjustable="box")
ax.grid(alpha=0.2)
ax.legend()
plt.show()
