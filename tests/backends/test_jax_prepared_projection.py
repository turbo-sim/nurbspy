"""Prepared initialization must preserve projection and its implicit derivatives."""

from importlib.util import find_spec

import numpy as np
from numpy.testing import assert_allclose
import pytest

if find_spec("jax") is None:
    pytest.skip("Install nurbspy[jax] for prepared projection checks.", allow_module_level=True)

import jax
import jax.numpy as jnp
import nurbspy.jax as nrb


def _paraboloid_case():
    """Represent z=.2*(u^2+v^2) exactly by a quadratic Bezier patch."""
    t = np.array([0., .5, 1.])
    x, y = np.meshgrid(t, t, indexing="ij")
    square_coefficients = np.array([0., 0., 1.])
    points = np.stack([x, y, .2 * (square_coefficients[:, None] + square_coefficients[None])])
    uv = np.array([.31, .63])
    normal = np.array([-.4 * uv[0], -.4 * uv[1], 1.])
    normal /= np.linalg.norm(normal)
    query = np.r_[uv, .2 * np.sum(uv**2)] + .15 * normal
    return points, uv, query


@pytest.mark.parametrize("mode", [jax.jacfwd, jax.jacrev], ids=["forward", "reverse"])
def test_prepared_projection_query_derivative_matches_exact_surface(mode):
    """Initialization cannot change the exact residual-curvature Hessian."""
    points, uv, query = _paraboloid_case()
    prepared = nrb.prepare_surface_projection(nrb.NurbsSurface(points))

    def project(point):
        return prepared.project_point(point, ftol=1e-14, gtol=1e-12)

    assert_allclose(jax.jit(project)(query), uv, atol=2e-8)
    jacobian = np.array([[1., 0.], [0., 1.], [.4 * uv[0], .4 * uv[1]]])
    residual_z = .2 * np.sum(uv**2) - query[2]
    hessian = jacobian.T @ jacobian + .4 * residual_z * np.eye(2)
    expected = np.linalg.solve(hessian, jacobian.T)
    assert_allclose(jax.jit(mode(project))(query), expected, rtol=1e-7, atol=1e-8)


def test_prepared_projection_batches_active_bound_solutions_and_derivatives():
    """A plane gives independent interior, edge and corner solutions."""
    points = jnp.array([[[0., 0.], [1., 1.]], [[0., 1.], [0., 1.]], [[0., 0.], [0., 0.]]])
    surface = nrb.NurbsSurface(points)
    prepared = nrb.prepare_surface_projection(surface)
    queries = jnp.array([[.3, -.2, 1.2], [.6, .6, 1.3], [1., 1., 1.]])
    expected = np.array([[.3, 0., 1.], [.6, .6, 1.]])
    actual = prepared.project_points(queries, ftol=1e-14, gtol=1e-12)
    assert_allclose(actual, expected, atol=1e-11)
    assert_allclose(actual, surface.project_points(queries, ftol=1e-14, gtol=1e-12), atol=1e-11)

    def project(point):
        return prepared.project_point(point, ftol=1e-14, gtol=1e-12)

    derivatives = jax.jit(jax.vmap(jax.jacrev(project)))(queries.T)
    expected_jacobians = np.array([
        [[1., 0., 0.], [0., 1., 0.]],
        [[0., 0., 0.], [0., 1., 0.]],
        [[0., 0., 0.], [0., 0., 0.]],
    ])
    assert_allclose(derivatives, expected_jacobians, atol=1e-11)


@pytest.mark.parametrize("mode", [jax.jacfwd, jax.jacrev], ids=["forward", "reverse"])
def test_projection_factory_inside_ad_preserves_geometry_sensitivity(mode):
    """A vertical patch translation has an independent implicit derivative."""
    points, uv, query = _paraboloid_case()

    def project_translation(offset):
        shifted = jnp.asarray(points).at[2].add(offset)
        prepared = nrb.prepare_surface_projection(nrb.NurbsSurface(shifted))
        return prepared.project_point(query, ftol=1e-14, gtol=1e-12)

    jacobian = np.array([[1., 0.], [0., 1.], [.4 * uv[0], .4 * uv[1]]])
    residual_z = .2 * np.sum(uv**2) - query[2]
    hessian = jacobian.T @ jacobian + .4 * residual_z * np.eye(2)
    expected = np.linalg.solve(hessian, -jacobian[2])
    actual = jax.jit(mode(project_translation))(jnp.array(0.))
    assert_allclose(actual, expected, rtol=1e-7, atol=1e-8)


def test_prepared_projection_snapshot_rebuild_and_argument_validation():
    """Changed geometry gets a new explicit snapshot rather than stale samples."""
    points, _, query = _paraboloid_case()
    original_surface = nrb.NurbsSurface(points)
    original = nrb.prepare_surface_projection(original_surface)
    shifted_surface = nrb.NurbsSurface(jnp.asarray(points).at[2].add(.2))
    rebuilt = nrb.prepare_surface_projection(shifted_surface)
    expected = jnp.stack(shifted_surface.project_point_to_surface(query, ftol=1e-14, gtol=1e-12))
    assert_allclose(rebuilt.project_point(query, ftol=1e-14, gtol=1e-12), expected, atol=2e-8)
    assert np.max(np.abs(np.asarray(original.sampled_points - rebuilt.sampled_points))) > .19
    assert original.surface is original_surface
    with pytest.raises(ValueError, match="surface dimension"):
        original.project_point(jnp.zeros(2))
    with pytest.raises(ValueError, match="shape"):
        original.project_points(jnp.zeros((2, 3)))
    with pytest.raises(ValueError, match="nonnegative"):
        original.project_point(query, maxiter=-1)
    with pytest.raises(ValueError, match="nonempty"):
        nrb.prepare_surface_projection(nrb.NurbsSurface())
