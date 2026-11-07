"""In-memory exchange uses the main external STL processing/writing path."""
from io import BytesIO
import numpy as np
import pytest
import trimesh

from nurbspy import export_stl_bytes


def test_stl_bytes_roundtrip_for_current_backend_geometry(surface_backend):
    x, y = np.meshgrid([0., 1.], [0., 2.], indexing="ij")
    surface = surface_backend.NurbsSurface(np.stack((x, y, 0.5 * x)))
    before = surface.to_yaml()
    data = export_stl_bytes(surface, nu=4, nv=3)
    loaded = trimesh.load_mesh(BytesIO(data), file_type="stl")
    assert isinstance(data, bytes)
    assert loaded.area == pytest.approx(2 * np.sqrt(1.25))
    np.testing.assert_allclose(loaded.bounds, [[0, 0, 0], [1, 2, .5]])
    assert surface.to_yaml() == before


def test_in_memory_export_retains_external_closure_validation(numpy_backend):
    surface = numpy_backend.NurbsSurface(np.array([[[0, 0], [1, 1]], [[0, 1], [0, 1]], [[0, 0], [0, 0.]]]))
    with pytest.raises(ValueError, match="closed"):
        export_stl_bytes(surface, require_closed=True)
