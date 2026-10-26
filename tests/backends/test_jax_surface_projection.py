"""Converged surface projection and its implicit forward/reverse derivatives."""

from importlib.util import find_spec

import numpy as np
from numpy.testing import assert_allclose
import pytest
from scipy.interpolate import BSpline
from scipy.optimize import root

if find_spec("jax") is None:
    pytest.skip(
        "Install nurbspy[jax] for differentiable projection checks.",
        allow_module_level=True,
    )

import jax
import jax.numpy as jnp
import nurbspy.jax as nrb
from tests.cases import paraboloid_control_points


@pytest.fixture(scope="module")
def paraboloid_projection_case():
    """Provide a strictly regular off-surface minimum with known parameters."""
    points = paraboloid_control_points() * np.array([1.0, 1.0, 0.2])[:, None, None]
    uv = np.array([0.31, 0.63])
    normal = np.array([-0.4 * uv[0], -0.4 * uv[1], 1.0])
    normal /= np.linalg.norm(normal)
    position = np.array([*uv, 0.2 * np.sum(uv**2)])
    return points, uv, position + 0.15 * normal


@pytest.mark.parametrize("mode", [jax.jacfwd, jax.jacrev], ids=["forward", "reverse"])
def test_curved_projection_query_derivatives(paraboloid_projection_case, mode):
    """Check the exact implicit Jacobian including the residual-curvature term."""
    points, uv, query = paraboloid_projection_case
    surface = nrb.NurbsSurface(points)

    def project(point):
        """Expose the query while retaining tight solver tolerances."""
        return jnp.stack(
            surface.project_point_to_surface(point, ftol=1e-14, gtol=1e-12)
        )

    result = jax.jit(project)(jnp.asarray(query))
    assert_allclose(result, uv, atol=2e-8)
    jacobian = np.array([[1.0, 0.0], [0.0, 1.0], [0.4 * uv[0], 0.4 * uv[1]]])
    residual_z = 0.2 * np.sum(uv**2) - query[2]
    hessian = jacobian.T @ jacobian + 0.4 * residual_z * np.eye(2)
    expected = np.linalg.solve(hessian, jacobian.T)
    assert_allclose(
        jax.jit(mode(project))(jnp.asarray(query)), expected, rtol=1e-7, atol=1e-8
    )


@pytest.mark.parametrize(
    "query,diagonal",
    [
        ([0.3, 0.6, 1.0], [1.0, 1.0]),
        ([-0.2, 0.6, 1.0], [0.0, 1.0]),
        ([1.2, 1.3, 1.0], [0.0, 0.0]),
    ],
)
def test_projection_active_bound_derivatives(query, diagonal):
    """Interior, edge and corner solutions have the appropriate free derivatives."""
    points = jnp.array(
        [[[0.0, 0.0], [1.0, 1.0]], [[0.0, 1.0], [0.0, 1.0]], [[0.0, 0.0], [0.0, 0.0]]]
    )
    surface = nrb.NurbsSurface(points)

    def project(point):
        """Return the clipped plane parameters as a single array."""
        return jnp.stack(
            surface.project_point_to_surface(point, ftol=1e-14, gtol=1e-12)
        )

    expected = np.column_stack((np.diag(diagonal), np.zeros(2)))
    assert_allclose(
        jax.jit(jax.jacfwd(project))(jnp.asarray(query)), expected, atol=1e-11
    )
    assert_allclose(
        jax.jit(jax.jacrev(project))(jnp.asarray(query)), expected, atol=1e-11
    )

    def squared_distance(point):
        """Differentiate the complete projection objective without stopping gradients."""
        uv = project(point)
        return jnp.sum((surface.get_value(*uv)[:, 0] - point) ** 2)

    closest = np.r_[np.clip(query[:2], 0, 1), 0.0]
    assert_allclose(
        jax.jit(jax.grad(squared_distance))(jnp.asarray(query)),
        2 * (query - closest),
        atol=1e-10,
    )


@pytest.mark.parametrize("argument", ["points", "weights"])
def test_projection_geometry_gradients(paraboloid_projection_case, argument):
    """Check projection-parameter shape gradients against an independent SciPy solve."""
    points, uv, query = paraboloid_projection_case
    weights = np.ones((3, 3))
    value = points if argument == "points" else weights
    direction = np.linspace(-0.2, 0.3, value.size).reshape(value.shape)

    def project_loss(variable):
        """Expose points or weights through the converged projection parameters."""
        surface = nrb.NurbsSurface(
            variable if argument == "points" else points,
            variable if argument == "weights" else weights,
        )
        result = jnp.stack(
            surface.project_point_to_surface(jnp.asarray(query), ftol=1e-14, gtol=1e-12)
        )
        return jnp.dot(result, jnp.array([0.7, 1.3]))

    basis = BSpline([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], np.eye(3), 2)

    def reference(variable):
        """Minimize an independently evaluated rational tensor-product patch."""
        p = variable if argument == "points" else points
        w = variable if argument == "weights" else weights

        def objective(parameters):
            """Supply the squared-distance objective and its analytic gradient."""
            bu, bv = basis(parameters[0]), basis(parameters[1])
            tensor = np.outer(bu, bv)
            denominator = np.sum(w * tensor)
            position = np.einsum("dij,ij->d", p * w, tensor) / denominator
            residual = position - query
            gradient = []
            for partial in [
                np.outer(basis(parameters[0], nu=1), bv),
                np.outer(bu, basis(parameters[1], nu=1)),
            ]:
                tangent = (
                    np.einsum("dij,ij->d", p * w, partial)
                    - position * np.sum(w * partial)
                ) / denominator
                gradient.append(np.dot(residual, tangent))
            return 0.5 * np.sum(residual**2), np.asarray(gradient)

        # This reference minimum is strictly interior. Solving stationarity
        # avoids objective-based stopping masking tiny geometric perturbations.
        result = root(lambda parameters: objective(parameters)[1], uv, tol=1e-11)
        assert_allclose(result.fun, 0.0, atol=1e-11)
        assert np.all((result.x > 0) & (result.x < 1))
        return np.dot(result.x, [0.7, 1.3])

    step = 1e-4
    expected = (
        reference(value + step * direction) - reference(value - step * direction)
    ) / (2 * step)
    _, tangent = jax.jit(lambda x, dx: jax.jvp(project_loss, (x,), (dx,)))(
        jnp.asarray(value), jnp.asarray(direction)
    )
    reverse = jnp.sum(jax.jit(jax.grad(project_loss))(jnp.asarray(value)) * direction)
    assert_allclose([tangent, reverse], expected, rtol=2e-5, atol=2e-7)


def test_batched_projection(paraboloid_projection_case):
    """Batch query points through the compiled public API with a (2,N) result."""
    points, uv, query = paraboloid_projection_case
    surface = nrb.NurbsSurface(points)
    queries = jnp.stack((jnp.asarray(query), surface.get_value(0.8, 0.2)[:, 0]), axis=1)
    result = jax.jit(lambda q: surface.project_points(q, ftol=1e-14, gtol=1e-12))(
        queries
    )
    assert result.shape == (2, 2)
    assert_allclose(result, np.column_stack((uv, [0.8, 0.2])), atol=2e-8)
