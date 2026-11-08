"""Illustrate the basis functions used to build B-spline curves and surfaces.

Define five cubic basis functions using a clamped knot vector. Three subplots
compare their values, first derivatives (slopes), and second derivatives
(changes in slope) along the parameter u. Change the degree, number of basis
functions, or knots below to explore how the basis changes.
"""

import numpy as np
import matplotlib.pyplot as plt
import nurbspy as nrb


def main(show=True):
    """Create the three-panel plot and return its Matplotlib figure."""
    degree = 3  # Cubic basis functions.
    control_count = 5  # Number of basis functions; at least degree + 1.
    n = control_count - 1  # The basis API uses the maximum index, starting at zero.

    # Clamped knots repeat each endpoint degree + 1 times.
    knots = np.concatenate(
        (np.zeros(degree), np.linspace(0, 1, n - degree + 2), np.ones(degree))
    )
    u = np.linspace(0, 1, 1001)  # Parameter samples for drawing smooth lines.

    basis = nrb.compute_basis_polynomials(n, degree, knots, u)
    first = nrb.compute_basis_polynomials_derivatives(n, degree, knots, u, 1)
    second = nrb.compute_basis_polynomials_derivatives(n, degree, knots, u, 2)

    nrb.set_plot_options(fontsize=12)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    titles = (
        "Basis function values",
        "Basis function first derivative",
        "Basis function second derivative",
    )
    for ax, values, title in zip(axes, (basis, first, second), titles):
        for i, row in enumerate(values):
            ax.plot(u, row, label=f"index {i}")
        ax.set(
            title=title, xlabel="$u$ parameter", ylabel="Function value", xlim=(0, 1)
        )

    # The same colours identify each basis function in all three panels.
    axes[-1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout(pad=1.0)
    if show:
        plt.show()
    return fig


if __name__ == "__main__":
    main()
