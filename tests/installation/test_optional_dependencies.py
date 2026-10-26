"""Check that the NumPy API works when optional dependencies are unavailable."""

from pathlib import Path
import subprocess
import sys
import textwrap

import pytest


@pytest.mark.parametrize("numba_state", ["missing", "incompatible"])
def test_numpy_backend_without_usable_numba(numba_state):
    """Use plain NumPy when Numba is missing or incompatible, without JAX."""
    # A fresh interpreter ensures other tests cannot leave Numba or nurbspy
    # cached in sys.modules and accidentally bypass the fallback.
    script = textwrap.dedent(
        """
        import importlib.abc
        from pathlib import Path
        import sys
        import warnings

        import numpy as np
        from numpy.testing import assert_allclose

        class BlockOptionalDependencies(importlib.abc.MetaPathFinder):
            '''Keep this subprocess independent of installed optional packages.'''

            def find_spec(self, fullname, path=None, target=None):
                '''Simulate unavailable optional dependencies in a fresh process.'''
                if fullname == "jax":
                    raise ModuleNotFoundError("No module named 'jax'", name="jax")
                if fullname == "numba":
                    if sys.argv[1] == "missing":
                        raise ModuleNotFoundError("No module named 'numba'", name="numba")
                    raise ImportError("Numba needs an older NumPy version")

        sys.meta_path.insert(0, BlockOptionalDependencies())
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", RuntimeWarning)
            import nurbspy as nrb

        assert Path(nrb.__file__).resolve().parent == Path.cwd() / "nurbspy"

        runtime_warnings = [w for w in caught if issubclass(w.category, RuntimeWarning)]
        assert len(runtime_warnings) == (sys.argv[1] == "incompatible"), runtime_warnings
        if runtime_warnings:
            assert "numba" in str(runtime_warnings[0].message).lower()
        assert "numba" not in sys.modules
        assert "jax" not in sys.modules
        assert not hasattr(nrb.compute_basis_polynomials, "py_func")

        knots = np.array([0., 0., 0., 1., 1., 1.])
        for u in (0, 0.25, np.array([0., 0.3, 1.]), np.array([0.2, 0.7]) + 1e-30j):
            t = np.atleast_1d(u)
            expected = np.array([(1-t)**2, 2*t*(1-t), t**2])
            derivative = np.array([-2*(1-t), 2-4*t, 2*t])
            assert_allclose(nrb.compute_basis_polynomials(2, 2, knots, u), expected)
            assert_allclose(nrb.compute_basis_polynomials_derivatives(2, 2, knots, u, 1), derivative)

        points = np.array([[0., 1., 2.], [0., 1., 0.]])
        curve = nrb.NurbsCurve(control_points=points)
        assert_allclose(curve.get_value(0.25)[:, 0], [0.5, 0.375])
        assert_allclose(curve.get_derivative(0.25, 1)[:, 0], [2., 1.])
        h = 1e-30
        assert_allclose(curve.get_value(0.25 + h*1j).imag[:, 0] / h, [2., 1.])

        corners = np.array([[[0., 0.], [1., 1.]], [[0., 1.], [0., 1.]], [[0., 0.], [0., 0.]]])
        surface = nrb.NurbsSurface(control_points=corners)
        assert_allclose(surface.get_value(0.25, 0.75)[:, 0], [0.25, 0.75, 0.])
    """
    )
    result = subprocess.run(
        [sys.executable, "-c", script, numba_state],
        cwd=Path(__file__).resolve().parents[2],
        text=True,
        capture_output=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
