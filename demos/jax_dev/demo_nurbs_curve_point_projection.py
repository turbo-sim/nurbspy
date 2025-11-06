# -------------------------------------------------------------------------------------------------------------------- #
# Importing packages
# -------------------------------------------------------------------------------------------------------------------- #
import jax.numpy as jnp
import matplotlib.pyplot as plt
import nurbspy.jax as nrb


# -------------------------------------------------------------------------------------------------------------------- #
# 2D NURBS curve example
# -------------------------------------------------------------------------------------------------------------------- #
# Define the array of control points (shape: 2 × 5)
P = jnp.array([
    [0.20, 0.40, 0.80, 0.60, 0.40],
    [0.50, 0.70, 0.60, 0.20, 0.20]
])

# Create the NURBS curve
nurbs2D = nrb.NurbsCurve(control_points=P, degree=3)
point = nurbs2D.get_value(u=0.2).squeeze()

# Define multiple points to project (each column is one point)
Q_all = jnp.array([
    [0.50, 0.70, 0.30, 0.50, 0.1, point[0]],
    [0.50, 0.50, 0.40, 0.30, 0.1, point[1]]
])  # shape (2, 4)

# Compute projected parameters for all points (vectorized)
u_all = nurbs2D.project_points(Q_all)

# Evaluate the projected coordinates on the curve
C_all = nurbs2D.get_value(u_all)

# Plot the NURBS curve and projection results
fig, ax = nurbs2D.plot()

for i in range(Q_all.shape[1]):
    Qx, Qy = Q_all[:, i]
    Cx, Cy = C_all[:, i]
    ax.plot(
        [Qx, Cx],
        [Qy, Cy],
        linestyle='--',
        color='b',
        marker='o',
        markeredgecolor='b',
        markerfacecolor='w'
    )

plt.show()
