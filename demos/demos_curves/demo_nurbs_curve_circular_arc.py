"""Example showing how to represent circular arcs using NURBS curves and verify geometric properties."""

# -------------------------------------------------------------------------------------------------------------------- #
# Imports
# -------------------------------------------------------------------------------------------------------------------- #
import numpy as np
import matplotlib.pyplot as plt
import nurbspy.jax as nrb


# -------------------------------------------------------------------------------------------------------------------- #
# 2D circular arc example
# -------------------------------------------------------------------------------------------------------------------- #
O = np.array([0.00, 1.00])             # Circle center
X = np.array([1.00, 0.00])             # X-direction in circle plane
Y = np.array([0.00, 1.00])             # Y-direction in circle plane
R = 0.5                                # Radius
theta_start = 1/6 * np.pi
theta_end = 3/2 * np.pi - 1/6 * np.pi

# Create and plot the circular arc
circ2D = nrb.CircularArc(O, X, Y, R, theta_start, theta_end)
circ2D.plot()

# Evaluate along the curve
u = np.linspace(0, 1, 100)
curv = circ2D.NurbsCurve.get_curvature(u)
tors = circ2D.NurbsCurve.get_torsion(u)
arc_len = circ2D.NurbsCurve.get_arclength()

# Analytical values
theta_span = np.abs(theta_end - theta_start)
curv_exact = 1 / R
tors_exact = 0.0
arc_exact = R * theta_span

# Report results
print("\n=== 2D circular arc check ===")
print(f"Expected curvature          : {curv_exact:.6f}")
print(f"Curvature RMS error         : {np.sqrt(np.mean((curv - curv_exact)**2)):.3e}")
print(f"Torsion RMS error           : {np.sqrt(np.mean((tors - tors_exact)**2)):.3e}")
print(f"Arc length (computed)       : {arc_len:.6f}")
print(f"Arc length (analytical)     : {arc_exact:.6f}")
print(f"Arc length absolute error   : {np.abs(arc_len - arc_exact):.3e}")


# -------------------------------------------------------------------------------------------------------------------- #
# 3D circular arc example
# -------------------------------------------------------------------------------------------------------------------- #
O = np.array([0.00, 0.00, 0.50])       # Circle center
X = np.array([3.00, 0.00, 0.00])       # X-direction (in plane)
Y = np.array([0.00, 1.00, 0.00])       # Y-direction (in plane)
R = 0.5
theta_start = 1/6 * np.pi
theta_end = np.pi

# Create and plot the circular arc
circ3D = nrb.CircularArc(O, X, Y, R, theta_start, theta_end)
circ3D.plot()

# Evaluate along the curve
u = np.linspace(0, 1, 100)
curv = circ3D.NurbsCurve.get_curvature(u)
tors = circ3D.NurbsCurve.get_torsion(u)
arc_len = circ3D.NurbsCurve.get_arclength()

# Analytical values
theta_span = np.abs(theta_end - theta_start)
curv_exact = 1 / R
tors_exact = 0.0
arc_exact = R * theta_span

# Report results
print("\n=== 3D circular arc check ===")
print(f"Expected curvature          : {curv_exact:.6f}")
print(f"Curvature RMS error         : {np.sqrt(np.mean((curv - curv_exact)**2)):.3e}")
print(f"Torsion RMS error           : {np.sqrt(np.mean((tors - tors_exact)**2)):.3e}")
print(f"Arc length (computed)       : {arc_len:.6f}")
print(f"Arc length (analytical)     : {arc_exact:.6f}")
print(f"Arc length absolute error   : {np.abs(arc_len - arc_exact):.3e}")

# -------------------------------------------------------------------------------------------------------------------- #
# Show figures
# -------------------------------------------------------------------------------------------------------------------- #
plt.show()
