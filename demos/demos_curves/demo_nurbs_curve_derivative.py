
""" Example showing how to compute the derivatives of a NURBS curve and validation against finite differences """


# -------------------------------------------------------------------------------------------------------------------- #
# Importing packages
# -------------------------------------------------------------------------------------------------------------------- #
import numpy as np
import nurbspy.jax as nrb
import matplotlib.pyplot as plt


# -------------------------------------------------------------------------------------------------------------------- #
# 3D NURBS curve example
# -------------------------------------------------------------------------------------------------------------------- #
# Define the array of control points
P = np.zeros((3, 5))
P[:, 0] = [0.00, 0.00, 0.00]
P[:, 1] = [0.00, 0.30, 0.05]
P[:, 2] = [0.25, 0.30, 0.30]
P[:, 3] = [0.50, 0.30, -0.05]
P[:, 4] = [0.50, 0.10, 0.10]

# Maximum index of the control points (counting from zero)
n = np.shape(P)[1] - 1

# Define the array of control point weights
W = np.asarray([1, 1, 3, 1, 1])

# Define the order of the basis polynomials
p = 3

# Define the knot vector (clamped spline)
# p+1 zeros, n-p equispaced points between 0 and 1, and p+1 ones. In total r+1 points where r=n+p+1
U = np.concatenate((np.zeros(p), np.linspace(0, 1, n - p + 2), np.ones(p)))

# Create and plot the NURBS curve
my_nurbs = nrb.NurbsCurve(control_points=P, weights=W, degree=p, knots=U)
my_nurbs.plot(frenet_serret=True)

# -------------------------------------------------------------------------------------------------------------------- #
# Check the analytic derivatives against a finite difference aproximation
# -------------------------------------------------------------------------------------------------------------------- #
# Define a u-parametrization suitable for finite differences
h = 1e-4
hh = 2*(h + h**2)
Nu = 1000
u = np.linspace(0.00+hh, 1.00-hh, Nu)

# Compute the NURBS derivatives analytically
dC_exact = my_nurbs.get_derivative(u, order=1)
ddC_exact = my_nurbs.get_derivative(u, order=2)
dddC_exact = my_nurbs.get_derivative(u, order=3)

# Approximate the NURBS derivatives by central finite differences
dC_fd = (my_nurbs.get_value(u + h) - my_nurbs.get_value(u - h)) / (2 * h)
ddC_fd = (my_nurbs.get_value(u + h) - 2 * my_nurbs.get_value(u) + my_nurbs.get_value(u - h)) / (h ** 2)
dddC_fd = (- 1 / 2 * my_nurbs.get_value(u - 2 * h) + my_nurbs.get_value(u - h) - my_nurbs.get_value(u + h) + 1 / 2 * my_nurbs.get_value(u + 2 * h)) / (h ** 3)

# Print the results
print('The two-norm of the first derivative error is   :  ', np.sum((dC_fd - dC_exact) ** 2) ** (1 / 2) / len(u))
print('The two-norm of the second derivative error is  :  ', np.sum((ddC_fd - ddC_exact) ** 2) ** (1 / 2) / len(u))
print('The two-norm of the third derivative error is   :  ', np.sum((dddC_fd - dddC_exact) ** 2) ** (1 / 2) / len(u))

plt.show()

