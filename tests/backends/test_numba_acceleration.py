"""Verify actual Numba compilation, including complex-step inputs."""

import importlib.util
import os
import subprocess
import sys

import numpy as np
from numpy.testing import assert_allclose
import pytest


def test_numba_basis_derivative_cache_reloads_in_another_process(tmp_path_factory):
    """A warm disk cache must not retain invalid recursive call addresses."""
    if importlib.util.find_spec("numba") is None:
        pytest.skip("Install nurbspy[numba] to check compiled cache reuse.")
    script = """
import numpy as np
from numpy.testing import assert_allclose
from nurbspy.numpy import compute_basis_polynomials_derivatives as derivative
knots = np.array([0., 0., 0., 1., 1., 1.])
u = np.array([0., .25, 1.])
assert_allclose(derivative(2, 2, knots, u, 1),
                np.array([-2*(1-u), 2-4*u, 2*u]))
assert_allclose(derivative(2, 2, knots, u, 3), 0, atol=0)
assert_allclose(derivative(1, 0, np.array([0., .4, 1.]), .3, 1),
                np.zeros((2, 1)), atol=0)
"""
    # Numba adds module hashes and long function names. Keep this prefix short
    # enough for Windows installations with the legacy path-length limit.
    cache_path = tmp_path_factory.mktemp("nb")
    environment = dict(os.environ, NUMBA_DISABLE_JIT="0", NUMBA_CACHE_DIR=str(cache_path))
    for _ in range(2):
        result = subprocess.run(
            [sys.executable, "-c", script], env=environment,
            capture_output=True, text=True, timeout=90,
        )
        assert result.returncode == 0, result.stdout+result.stderr


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
