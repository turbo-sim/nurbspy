"""Example showing how to create a thickness distribution"""

# -------------------------------------------------------------------------------------------------------------------- #
# Importing packages
# -------------------------------------------------------------------------------------------------------------------- #
import numpy as np
import nurbspy.jax as nrb
import matplotlib.pyplot as plt


def cross2d(a, b):
    return a[0] * b[1] - a[1] * b[0]


x0 = 0.00
y0 = 0.00

x2 = 0.2
y2 = 0.2

theta_0 = np.radians(90.0)
theta_2 = np.radians(10.0)

R = 0.5

A = np.array([[np.cos(theta_0), np.cos(theta_2)], [np.sin(theta_0), np.sin(theta_2)]])
b = np.array([x2 - x0, y2 - y0])
d1, d2 = np.linalg.solve(A, b)


x1 = x0 + d1 * np.cos(theta_0)
y1 = y0 + d1 * np.sin(theta_0)

x1_bis = x2 - d2 * np.cos(theta_2)
y1_bis = y2 - d2 * np.sin(theta_2)


P = np.asarray([[x0, y0], [x1, y1], [x2, y2]]).T

D20 = P[:, 2] - P[:, 0]  # P2 - P0
D21 = P[:, 2] - P[:, 1]  # P2 - P1


w1 = np.sqrt((0.5 * R * abs(cross2d(D20, D21)) / np.linalg.norm(D21) ** 3))

print(f"Weight w1: {w1}")


W = np.asarray([1, w1, 1])
bezier = nrb.NurbsCurve(control_points=P, weights=W)
fig, ax = bezier.plot(frenet_serret=False, control_points=True)

R_check = (
    (bezier.p - 1)
    / bezier.p
    * (bezier.W[2] * bezier.W[0])
    / bezier.W[1] ** 2
    * abs(cross2d(D20, D21))
    / np.linalg.norm(D21) ** 3
) ** -1

print(f"Input R: {R}")
print(f"Checked R: {R_check}")


plt.show()
