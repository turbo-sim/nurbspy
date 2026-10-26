"""Select supported backends while keeping geometry tests implementation-neutral.

Numba accelerates the NumPy backend when installed; it is not a separate
geometry API. Shared geometry checks exercise NumPy and optional JAX.
"""

import importlib
import importlib.util

import pytest


def pytest_addoption(parser):
    """Allow CI to require an optional backend instead of accepting skips."""
    parser.addoption(
        "--require-backend",
        action="append",
        choices=("numba", "jax"),
        default=[],
        help="Fail early if the requested optional backend cannot be imported.",
    )


def pytest_sessionstart(session):
    """Verify explicitly requested dependencies before collecting test modules."""
    for backend in session.config.getoption("--require-backend"):
        module = "nurbspy.jax" if backend == "jax" else "numba"
        try:
            importlib.import_module(module)
        except ImportError as error:
            raise pytest.UsageError(
                f"Required backend {backend!r} could not import: {error}"
            ) from error


def _load_geometry_backend(module_name):
    """Skip an absent JAX extra, but expose errors in an installed backend."""
    if module_name == "nurbspy.jax" and importlib.util.find_spec("jax") is None:
        pytest.skip("Install nurbspy[jax] to run the JAX geometry cases.")
    return importlib.import_module(module_name)


@pytest.fixture(scope="module", params=["nurbspy", "nurbspy.jax"], ids=["numpy", "jax"])
def basis_backend(request):
    """Run common basis-function checks against NumPy and available JAX."""
    return _load_geometry_backend(request.param)


@pytest.fixture(scope="module", params=["nurbspy", "nurbspy.jax"], ids=["numpy", "jax"])
def curve_backend(request):
    """Run common curve checks against NumPy and available JAX."""
    return _load_geometry_backend(request.param)


@pytest.fixture(scope="module", params=["nurbspy", "nurbspy.jax"], ids=["numpy", "jax"])
def surface_backend(request):
    """Run common surface checks against NumPy and available JAX."""
    return _load_geometry_backend(request.param)


@pytest.fixture(scope="module")
def numpy_backend():
    """Provide NumPy/Numba for complex-step and other specific guarantees."""
    return importlib.import_module("nurbspy")


@pytest.fixture(scope="module")
def jax_backend():
    """Provide JAX for functionality that has no NumPy equivalent."""
    return _load_geometry_backend("nurbspy.jax")
