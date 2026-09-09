"""Plot cubic B-spline basis functions and their first two derivatives."""
import numpy as np
import matplotlib.pyplot as plt
import nurbspy as nrb

nrb.set_plot_options()

n, p = 4, 3  # Five basis functions, each of degree three.
U = np.array([0., 0., 0., 0., 0.5, 1., 1., 1., 1.])
u = np.linspace(0., 1., 501)
N = nrb.compute_basis_polynomials(n, p, U, u)
dN = nrb.compute_basis_polynomials_derivatives(n, p, U, u, 1)
ddN = nrb.compute_basis_polynomials_derivatives(n, p, U, u, 2)

print("Basis array shape:", N.shape)
print("Basis at u=0.5:", N[:, 250])
print(f"Maximum partition-of-unity error: {np.max(np.abs(N.sum(axis=0) - 1)):.2e}")
print(f"Maximum first-derivative sum: {np.max(np.abs(dN.sum(axis=0))):.2e}")
print(f"Maximum second-derivative sum: {np.max(np.abs(ddN.sum(axis=0))):.2e}")

fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), layout="constrained")
for ax, values, title in zip(axes, [N, dN, ddN],
                             ["Basis functions", "First derivatives", "Second derivatives"]):
    for i, row in enumerate(values):
        ax.plot(u, row, label=f"i={i}")
    ax.axvline(0.5, color="0.7", linestyle=":", linewidth=1)
    ax.set(xlabel="u", ylabel="Value", title=title)
    ax.grid(alpha=0.2)
axes[0].legend(fontsize=10)
plt.show()
