"""Verify actual Numba compilation, including complex-step inputs."""

import importlib.util

import numpy as np
from numpy.testing import assert_allclose
import pytest


@pytest.mark.parametrize("complex_step", [False, True], ids=["real", "complex_step"])
def test_numba_extra_compiles_basis_functions(complex_step):
    """Compile basis values and derivatives for real and complex parameters."""
    if importlib.util.find_spec("numba") is None:
        pytest.skip("Install nurbspy[numba] to check compiled acceleration.")
    import numba
    import nurbspy.numpy as nrb

    knots = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    u = np.array([0.2, 0.7]) + 1e-30j if complex_step else np.array([0.0, 0.25, 1.0])
    assert_allclose(
        nrb.compute_basis_polynomials(2, 2, knots, u),
        np.array([(1 - u) ** 2, 2 * u * (1 - u), u**2]),
    )
    assert_allclose(
        nrb.compute_basis_polynomials_derivatives(2, 2, knots, u, 1),
        np.array([-2 * (1 - u), 2 - 4 * u, 2 * u]),
    )

    # Matching values alone would also pass through the plain NumPy fallback.
    # Dispatchers and compiled signatures prove that this extra is active.
    for function in (
        nrb.compute_basis_polynomials,
        nrb.compute_basis_polynomials_derivatives,
    ):
        assert isinstance(function, numba.core.registry.CPUDispatcher)
        assert function.nopython_signatures
