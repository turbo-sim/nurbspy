"""Sampled mesh geometry, topology indexing, and NumPy/JAX agreement."""
import numpy as np
import pytest

from nurbspy.mesh import tessellate_surfaces


def patch(backend, P=None):
    if P is None:
        x, y = np.meshgrid([0., 1.], [0., 2.], indexing="ij")
        P = np.stack((x, y, np.zeros_like(x)))
    return backend.NurbsSurface(P)


def test_planar_area_orientation_and_non_square_grid(surface_backend):
    surface = patch(surface_backend)
    vertices, faces = surface.tessellate(nu=5, nv=3)
    assert isinstance(vertices, np.ndarray) and vertices.shape == (15, 3)
    assert faces.shape == (16, 3) and np.issubdtype(faces.dtype, np.integer)
    triangles = vertices[faces]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    np.testing.assert_allclose(normals[:, :2], 0)
    assert np.all(normals[:, 2] > 0)
    assert np.isclose(normals[:, 2].sum() / 2, 2)


def test_actual_parameter_domain_and_knot_breaks(surface_backend):
    x, y = np.meshgrid([0., .3, .6, 1.], [0., 1.], indexing="ij")
    surface = surface_backend.NurbsSurface(np.stack((x, y, x * y)), u_degree=2, v_degree=1,
        u_knots=np.array([2, 2, 2, 2.7, 5, 5, 5]), v_knots=np.array([-3, -3, -1, -1]))
    vertices, faces = surface.tessellate(3, 2)
    assert len(vertices) == 8
    expected = np.asarray(surface.get_value(np.array([2.7, 2.7]), np.array([-3., -1.]))).T
    assert all(np.min(np.linalg.norm(vertices - point, axis=1)) < 1e-12 for point in expected)
    assert faces.min() >= 0 and faces.max() < len(vertices)


def test_patch_collection_offsets_and_no_implicit_welding(surface_backend):
    surface = patch(surface_backend)
    vertices, faces = tessellate_surfaces(iter([surface, surface]), 3, 2)
    assert len(vertices) == 12 and len(faces) == 8
    assert faces[:4].max() < 6 and faces[4:].min() >= 6
    np.testing.assert_array_equal(vertices[:6], vertices[6:])


def test_collapsed_boundary_has_no_zero_area_faces(surface_backend):
    P = np.array([[[0, 0], [1, 0]], [[0, 0], [0, 1]], [[0, 0], [0, 0]]], dtype=float)
    vertices, faces = patch(surface_backend, P).tessellate(5, 3)
    tri = vertices[faces]
    assert np.all(np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1) > 0)


def test_curved_surface_area_converges(surface_backend):
    x, y = np.meshgrid([0., .5, 1.], [0., 1.], indexing="ij")
    surface = surface_backend.NurbsSurface(np.stack((x, y, np.stack(([0., 0.], [0., 0.], [1., 1.])))))
    exact = np.sqrt(5) / 2 + np.arcsinh(2) / 4  # area of z=x**2 extruded over unit y
    errors = []
    for count in (3, 8, 24):
        vertices, faces = surface.tessellate(count, 2)
        tri = vertices[faces]
        area = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1).sum() / 2
        errors.append(abs(area - exact))
    assert errors[2] < errors[1] < errors[0]


@pytest.mark.parametrize("nu,nv", [(1, 3), (True, 3), (3., 3), (3, 0), (3, False)])
def test_invalid_sample_counts(numpy_backend, nu, nv):
    with pytest.raises(ValueError):
        patch(numpy_backend).tessellate(nu, nv)


def test_invalid_inputs_are_explicit(numpy_backend):
    with pytest.raises(ValueError, match="At least one"):
        tessellate_surfaces([])
    with pytest.raises(TypeError):
        tessellate_surfaces(None)
    with pytest.raises(ValueError, match="three-dimensional"):
        numpy_backend.NurbsSurface(np.zeros((2, 2, 2))).tessellate()
    for value in (np.nan, np.inf, -1, 0):
        surface = patch(numpy_backend)
        surface.W[0, 0] = value
        with pytest.raises(ValueError):
            surface.tessellate(2, 2)
    surface = patch(numpy_backend)
    surface.P = surface.P.astype(complex)
    with pytest.raises(ValueError, match="real"):
        surface.tessellate(2, 2)
    with pytest.raises(ValueError, match="extent"):
        patch(numpy_backend, np.zeros((3, 2, 2))).tessellate(2, 2)


def test_discontinuous_patch_is_not_bridged(numpy_backend):
    x, y = np.meshgrid([0., 1., 3., 4.], [0., 1.], indexing="ij")
    surface = numpy_backend.NurbsSurface(np.stack((x, y, 0*x)), u_degree=1, v_degree=1,
        u_knots=np.array([0., 0., .5, .5, 1., 1.]), v_knots=np.array([0., 0., 1., 1.]))
    with pytest.raises(ValueError, match="Discontinuous"):
        surface.tessellate(4, 3)


def test_numpy_jax_samples_agree(numpy_backend, jax_backend):
    vertices, faces = patch(numpy_backend).tessellate(7, 4)
    jax_vertices, jax_faces = patch(jax_backend).tessellate(7, 4)
    np.testing.assert_allclose(jax_vertices, vertices)
    np.testing.assert_array_equal(jax_faces, faces)


@pytest.mark.parametrize("knots", [np.array([0., 0., 0., 0.]), np.array([0., .5, .2, 1.]), np.array([0., 0., 1.]), np.array([0., 0., np.nan, 1.])])
def test_bad_knots_are_rejected(numpy_backend, knots):
    surface = patch(numpy_backend)
    surface.U = knots
    with pytest.raises(ValueError, match="knot"):
        surface.tessellate(3, 3)


def test_sampling_does_not_mutate_native_model(surface_backend):
    surface = patch(surface_backend)
    before = surface.to_yaml()
    surface.tessellate(6, 4)
    assert surface.to_yaml() == before
