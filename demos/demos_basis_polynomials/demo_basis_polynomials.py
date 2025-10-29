
""" Example showing how to compute a family of basis polynomials """


# -------------------------------------------------------------------------------------------------------------------- #
# Importing packages
# -------------------------------------------------------------------------------------------------------------------- #
import numpy as np
import nurbspy.jax as nrb
import matplotlib.pyplot as plt


# -------------------------------------------------------------------------------------------------------------------- #
# Basis polynomials and derivatives example
# -------------------------------------------------------------------------------------------------------------------- #
# Maximum index of the basis polynomials (counting from zero)
n = 4

# Define the order of the basis polynomials
p = 3

# Define the knot vector (clamped spline)
# p+1 zeros, n-p equispaced points between 0 and 1, and p+1 ones. In total r+1 points where r=n+p+1
U = np.concatenate((np.zeros(p), np.linspace(0, 1, n - p + 2), np.ones(p)))

# Define a new u-parametrization suitable for finite differences
u = np.linspace(0.00, 1.00, 1001)       # Make sure that the limits [0, 1] also work when making changes

# Compute the basis polynomials and derivatives
N_basis   = nrb.compute_basis_polynomials(n, p, U, u)
dN_basis  = nrb.compute_basis_polynomials_derivatives(n, p, U, u, derivative_order=1)
ddN_basis = nrb.compute_basis_polynomials_derivatives(n, p, U, u, derivative_order=2)


# -------------------------------------------------------------------------------------------------------------------- #
# Plot the basis polynomials
# -------------------------------------------------------------------------------------------------------------------- #
# Create the figure
fig = plt.figure(figsize=(15, 4.5))

# Plot the basis polynomials
ax1 = fig.add_subplot(131)
ax1.set_title('Zeroth derivative', fontsize=12, color='k', pad=12)
ax1.set_xlabel('$u$ parameter', fontsize=12, color='k', labelpad=12)
ax1.set_ylabel('Function value', fontsize=12, color='k', labelpad=12)
for i in range(n+1):
    line, = ax1.plot(u, N_basis[i, :])
    line.set_linewidth(1.25)
    line.set_linestyle("-")
    # line.set_color("k")
    line.set_marker(" ")
    line.set_markersize(3.5)
    line.set_markeredgewidth(1)
    line.set_markeredgecolor("k")
    line.set_markerfacecolor("w")
    line.set_label('index ' + str(i))


# Plot the first derivative
ax2 = fig.add_subplot(132)
ax2.set_title('First derivative', fontsize=12, color='k', pad=12)
ax2.set_xlabel('$u$ parameter', fontsize=12, color='k', labelpad=12)
ax2.set_ylabel('Function value', fontsize=12, color='k', labelpad=12)
for i in range(n+1):
    line, = ax2.plot(u, dN_basis[i, :])
    line.set_linewidth(1.25)
    line.set_linestyle("-")
    # line.set_color("k")
    line.set_marker(" ")
    line.set_markersize(3.5)
    line.set_markeredgewidth(1)
    line.set_markeredgecolor("k")
    line.set_markerfacecolor("w")
    line.set_label('index ' + str(i))


# Plot the second derivative
ax3 = fig.add_subplot(133)
ax3.set_title('Second derivative', fontsize=12, color='k', pad=12)
ax3.set_xlabel('$u$ parameter', fontsize=12, color='k', labelpad=12)
ax3.set_ylabel('Function value', fontsize=12, color='k', labelpad=12)
for i in range(n+1):
    line, = ax3.plot(u, ddN_basis[i, :])
    line.set_linewidth(1.25)
    line.set_linestyle("-")
    # line.set_color("k")
    line.set_marker(" ")
    line.set_markersize(3.5)
    line.set_markeredgewidth(1)
    line.set_markeredgecolor("k")
    line.set_markerfacecolor("w")
    line.set_label('index ' + str(i))


# Create legend
ax3.legend(ncol=1, loc='right', bbox_to_anchor=(1.60, 0.50), fontsize=10, edgecolor='k', framealpha=1.0)

# Adjust pad
plt.tight_layout(pad=1.)

# Show the figure
plt.show()

