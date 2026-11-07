"""STL exchange uses Trimesh but is checked independently at the binary boundary."""
import builtins
import struct

import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")
pytest.importorskip("networkx")

from nurbspy.mesh import export_stl


def plane(backend):
    x, y = np.meshgrid([0., 1.], [0., 2.], indexing="ij")
    return backend.NurbsSurface(np.stack((x, y, 0*x)))


def cube(backend, scale=1., offset=0.):
    patches = []
    for axis in range(3):
        other = [d for d in range(3) if d != axis]
        for side in (0., 1.):
            P = np.zeros((3, 2, 2))
            P[axis] = side
            P[other[0]] = [[0, 0], [1, 1]]
            P[other[1]] = [[0, 1], [0, 1]]
            patches.append(backend.NurbsSurface(scale * P + offset))
    return patches


def raw_stl(path):
    data = path.read_bytes()
    count = struct.unpack_from("<I", data, 80)[0]
    assert len(data) == 84 + 50 * count
    dtype = np.dtype([("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attribute", "<u2")])
    return np.frombuffer(data, dtype=dtype, offset=84)


def test_surface_method_exports_current_geometry_and_preserves_input(surface_backend, tmp_path):
    surface = plane(surface_backend)
    snapshot = surface.to_yaml()
    path = tmp_path / "patch.STL"
    assert surface.export(path, nu=4, nv=3) == path
    records = raw_stl(path)
    assert len(records) == 12
    np.testing.assert_allclose(records["normal"], np.tile([0, 0, 1], (12, 1)))
    assert np.isfinite(records["vertices"]).all()
    reloaded = trimesh.load_mesh(path)
    assert not reloaded.is_watertight
    assert reloaded.area == pytest.approx(2)
    assert surface.to_yaml() == snapshot


def test_multiple_patches_form_closed_outward_cube(surface_backend, tmp_path):
    path = tmp_path / "cube.stl"
    export_stl(cube(surface_backend), path, 4, 4, require_closed=True)
    records = raw_stl(path)
    points = records["vertices"].astype(float)
    cross = np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0])
    np.testing.assert_allclose(records["normal"], cross / np.linalg.norm(cross, axis=1)[:, None], atol=1e-6)
    reloaded = trimesh.load_mesh(path)
    assert reloaded.is_watertight and reloaded.is_winding_consistent and reloaded.is_volume
    assert reloaded.volume == pytest.approx(1)
    assert reloaded.area == pytest.approx(6)


def test_closed_requirement_rejects_open_surface_without_overwriting(numpy_backend, tmp_path):
    path = tmp_path / "existing.stl"
    path.write_bytes(b"original")
    with pytest.raises(ValueError, match="not closed"):
        plane(numpy_backend).export(path, require_closed=True)
    assert path.read_bytes() == b"original"


def test_incompatible_patch_sampling_is_not_silently_filled(numpy_backend, tmp_path):
    path = tmp_path / "gap.stl"
    with pytest.raises(ValueError, match="not closed"):
        export_stl(cube(numpy_backend), path, 4, 3, require_closed=True)
    assert not path.exists()


@pytest.mark.parametrize("offset", [.25, 2.])
def test_nested_or_disconnected_shells_are_explicitly_unsupported(numpy_backend, tmp_path, offset):
    surfaces = cube(numpy_backend) + cube(numpy_backend, scale=.5, offset=offset)
    with pytest.raises(ValueError, match="one connected shell"):
        export_stl(surfaces, tmp_path / "shells.stl", 3, 3, require_closed=True)


@pytest.mark.parametrize("scale,offset", [(1., 1e8), (1e39, 0.), (1e-50, 0.)])
def test_float32_precision_failures_do_not_write(numpy_backend, tmp_path, scale, offset):
    path = tmp_path / "precision.stl"
    with pytest.raises(ValueError, match="float32"):
        export_stl(cube(numpy_backend, scale=scale, offset=offset), path, 2, 2, require_closed=True)
    assert not path.exists()


@pytest.mark.parametrize("tolerance", [0., -1., np.nan, np.inf, True, "invalid", 1e-30, 5.])
def test_invalid_merge_tolerance(numpy_backend, tmp_path, tolerance):
    with pytest.raises(ValueError, match="weld_tolerance"):
        plane(numpy_backend).export(tmp_path / "bad.stl", weld_tolerance=tolerance)


def test_unsupported_format_and_closed_option(numpy_backend, tmp_path):
    surface = plane(numpy_backend)
    with pytest.raises(ValueError, match="Only .stl"):
        surface.export(tmp_path / "patch.obj")
    with pytest.raises(ValueError, match="boolean"):
        surface.export(tmp_path / "patch.stl", require_closed="yes")


def test_missing_optional_dependency_has_actionable_error(numpy_backend, monkeypatch, tmp_path):
    real_import = builtins.__import__
    def without_trimesh(name, *args, **kwargs):
        if name == "trimesh":
            raise ImportError("not installed")
        return real_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", without_trimesh)
    with pytest.raises(ImportError, match=r"nurbspy\[mesh\]"):
        plane(numpy_backend).export(tmp_path / "patch.stl")


def test_oloid_seams_and_pinched_corners(numpy_backend, tmp_path):
    nrb = numpy_backend
    theta = np.pi - 2 * np.pi / 3 * np.cos(np.linspace(0, np.pi, 31))
    cosine = np.cos(theta)
    cos_phi = np.clip(cosine / (1 - cosine), -1, 1)
    sin_phi = np.sqrt(1 - cos_phi**2)
    cos_phi[[0, -1]], sin_phi[[0, -1]] = 1, 0
    first = nrb.NurbsCurve(np.stack((cosine - .5, np.sin(theta), np.zeros_like(theta))), degree=1)
    surfaces = [nrb.NurbsSurfaceRuled(first, nrb.NurbsCurve(
        np.stack((.5 + cos_phi, np.zeros_like(theta), sign * sin_phi)), degree=1)).NurbsSurface
        for sign in (-1, 1)]
    path = tmp_path / "oloid.stl"
    export_stl(surfaces, path, 20, 8, require_closed=True)
    reloaded = trimesh.load_mesh(path)
    assert reloaded.is_watertight and reloaded.is_winding_consistent and reloaded.is_volume
    assert 2.9 < reloaded.volume < 3.1


def test_optional_export_is_shared_between_backends(numpy_backend, jax_backend):
    assert numpy_backend.NurbsSurface.export is jax_backend.NurbsSurface.export


def test_nonmanifold_edges_are_rejected(numpy_backend, tmp_path):
    a, b = np.array([0., 0., 0.]), np.array([1., 0., 0.])
    patches = []
    for c in ([0., 1., 0.], [0., 0., 1.], [0., -1., 0.]):
        P = np.stack((a, c, b, c)).T.reshape(3, 2, 2)
        patches.append(numpy_backend.NurbsSurface(P))
    with pytest.raises(ValueError, match="non-manifold"):
        export_stl(patches, tmp_path / "nonmanifold.stl", 2, 2)


def test_mesh_library_normal_precision_does_not_silently_write_zeros(numpy_backend, tmp_path):
    surface = plane(numpy_backend)
    surface.P *= 1e-9
    path = tmp_path / "tiny.stl"
    with pytest.raises(ValueError, match="normals.*rescale"):
        surface.export(path, nu=2, nv=2)
    assert not path.exists()
