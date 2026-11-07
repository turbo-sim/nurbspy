"""NURBS-specific surface sampling, independent of mesh libraries and viewers.

Sampling is count-based, includes interior knot breaks, and has no geometric
error guarantee. Vertices on different patches are not welded here.
"""
import numpy as np


__all__ = ["tessellate_surfaces", "export_stl", "export_stl_bytes"]


def _parameter_grid(knots, degree, count, control_count):
    knots = np.asarray(knots)
    if isinstance(degree, bool) or not isinstance(degree, (int, np.integer)) or not 0 <= degree < control_count:
        raise ValueError("Surface degrees must be integers smaller than their control-point counts")
    if (knots.dtype.kind not in "iuf" or knots.ndim != 1 or
            knots.size != control_count + degree + 1 or not np.all(np.isfinite(knots))):
        raise ValueError("Surface knot vectors must have finite real values and compatible lengths")
    knots = knots.astype(float)
    start, end = knots[degree], knots[-degree - 1]
    if np.any(np.diff(knots) < 0) or start >= end:
        raise ValueError("Surface knots must be ordered with a positive parameter domain")
    values, multiplicity = np.unique(knots, return_counts=True)
    interior = (values > start) & (values < end)
    if np.any(multiplicity[interior] > degree):
        raise ValueError("Discontinuous interior knots require separate surface patches for tessellation")
    return np.unique(np.concatenate((np.linspace(start, end, count), values[interior])))


def _tessellate_patch(surface, nu, nv):
    try:
        P, W = np.asarray(surface.P), np.asarray(surface.W)
    except AttributeError as error:
        raise TypeError("Expected a nurbspy surface or a sequence of surfaces") from error
    if P.ndim != 3 or P.shape[0] != 3 or min(P.shape) < 1 or W.shape != P.shape[1:]:
        raise ValueError("Tessellation requires a three-dimensional surface with compatible weights")
    if (any(array.dtype.kind not in "iuf" for array in (P, W)) or
            not np.all(np.isfinite(P)) or not np.all(np.isfinite(W)) or np.any(W <= 0)):
        raise ValueError("Tessellation requires finite real coordinates and positive finite weights")
    u = _parameter_grid(surface.U, surface.p, nu, P.shape[1])
    v = _parameter_grid(surface.V, surface.q, nv, P.shape[2])
    uu, vv = np.meshgrid(u, v, indexing="ij")
    vertices = np.asarray(surface.get_value(uu.ravel(), vv.ravel()))
    if vertices.shape != (3, uu.size) or np.iscomplexobj(vertices) or not np.all(np.isfinite(vertices)):
        raise ValueError("Surface evaluation must return finite real coordinates of shape (3, N)")
    vertices = vertices.T.astype(float)
    with np.errstate(over="ignore", invalid="ignore"):
        extent = float(np.ptp(vertices, axis=0).max())
    if not np.isfinite(extent) or extent <= 0:
        raise ValueError("Surface extent must be finite and nonzero")
    local = (vertices - vertices[0]) / extent
    grid = np.arange(uu.size).reshape(uu.shape)
    a, b = grid[:-1, :-1].ravel(), grid[1:, :-1].ravel()
    c, d = grid[1:, 1:].ravel(), grid[:-1, 1:].ravel()
    first = np.stack((np.stack((a, b, c), axis=1), np.stack((a, c, d), axis=1)), axis=1)
    second = np.stack((np.stack((a, b, d), axis=1), np.stack((b, c, d), axis=1)), axis=1)

    def area(faces):
        triangles = local[faces]
        return np.linalg.norm(np.cross(triangles[..., 1, :] - triangles[..., 0, :],
                                       triangles[..., 2, :] - triangles[..., 0, :]), axis=-1)

    # Prefer the healthier diagonal at pinched or highly skewed patch corners.
    first_quality, second_quality = area(first).min(axis=1), area(second).min(axis=1)
    tolerance = 32 * np.finfo(float).eps * np.maximum(first_quality, second_quality)
    use_second = second_quality > first_quality + tolerance
    first[use_second] = second[use_second]
    faces = first.reshape(-1, 3)
    faces = faces[area(faces) > 0]
    if not len(faces):
        raise ValueError("Surface tessellation contains no nondegenerate triangles")
    used, indices = np.unique(faces, return_inverse=True)
    return vertices[used], indices.reshape(-1, 3)


def tessellate_surfaces(surfaces, nu=80, nv=30):
    """Return host NumPy ``(vertices, faces)`` from one or several surfaces.

    Vertices have shape ``(N, 3)`` and integer triangle indices ``(M, 3)``.
    ``nu``/``nv`` are regular sample counts (at least two); knot breaks are added
    and may increase the resulting counts. Each patch retains its u-cross-v
    orientation. Zero-area faces are omitted, but seams are not welded and mesh
    topology is not repaired. Discontinuous interior knots are rejected.

    Requires finite real 3D geometry with positive weights, independently of
    the numerical backend. JAX samples transfer to the host; this is outside
    JIT/autodiff. Count-based sampling does not guarantee geometric accuracy,
    watertightness, or absence of self-intersections.
    """
    for name, count in (("nu", nu), ("nv", nv)):
        if isinstance(count, bool) or not isinstance(count, (int, np.integer)) or count < 2:
            raise ValueError(f"{name} must be an integer of at least two")
    if hasattr(surfaces, "get_value"):
        surfaces = [surfaces]
    else:
        try:
            surfaces = list(surfaces)
        except TypeError as error:
            raise TypeError("Expected a surface or iterable of surfaces") from error
    if not surfaces:
        raise ValueError("At least one surface is required")
    vertices, faces, offset = [], [], 0
    for surface in surfaces:
        patch_vertices, patch_faces = _tessellate_patch(surface, nu, nv)
        vertices.append(patch_vertices)
        faces.append(patch_faces + offset)
        offset += len(patch_vertices)
    return np.concatenate(vertices), np.concatenate(faces)


class SurfaceMeshMixin:
    """Shared NumPy/JAX host-side mesh interface."""

    def tessellate(self, nu=80, nv=30):
        """Sample this surface as NumPy vertices/triangles; see tessellate_surfaces."""
        return tessellate_surfaces(self, nu=nu, nv=nv)

    def export(self, path, *, nu=80, nv=30, require_closed=False, weld_tolerance=None):
        """Write .stl through Trimesh; see export_stl for sampling/solid limits."""
        return export_stl(self, path, nu=nu, nv=nv, require_closed=require_closed,
                          weld_tolerance=weld_tolerance)


def _mesh_library():
    try:
        import trimesh
        import networkx  # Required by Trimesh's winding repair; imported lazily.
    except ImportError as error:
        raise ImportError("STL dependencies are unavailable; install trimesh and networkx or reinstall nurbspy") from error
    return trimesh


def export_stl(surfaces, path, nu=80, nv=30, *, require_closed=False, weld_tolerance=None):
    """Export one or several surfaces using Trimesh processing/writing.

    ``require_closed=True`` checks edge closure, winding, and positive volume
    for ONE face-connected shell. Disconnected/nested shells are rejected in
    this mode; cavity interpretation and self-intersection testing are outside
    this exporter. No holes are filled. Open surface export is the default.

    Seam merging uses Trimesh's decimal rounding on scale-normalized vertices.
    ``weld_tolerance`` is an approximate absolute rounding scale, defaulting to
    1e-10 times the largest extent. Compatible boundary samples are required;
    it cannot stitch different parameterizations or repair arbitrary geometry.
    STL coordinates are unitless float32. Precision collapse is rejected before
    writing. Returns the output Path. This host operation is outside JIT.
    """
    from pathlib import Path
    path = Path(path)
    if path.suffix.lower() != ".stl":
        raise ValueError("Only .stl export is supported; use to_yaml() for native geometry")
    path.write_bytes(export_stl_bytes(surfaces, nu=nu, nv=nv,
                                    require_closed=require_closed, weld_tolerance=weld_tolerance))
    return path


def export_stl_bytes(surfaces, nu=80, nv=30, *, require_closed=False, weld_tolerance=None):
    """Return binary STL using the same external mesh pipeline as export_stl.

    Suitable for downloads and in-memory exchange. Sampling, processing,
    validation, and units are identical to the file export operation.
    """
    mesh = _prepare_stl_mesh(surfaces, nu, nv, require_closed=require_closed,
                             weld_tolerance=weld_tolerance)
    return mesh.export(file_type="stl")


def _prepare_stl_mesh(surfaces, nu=80, nv=30, *, require_closed=False, weld_tolerance=None):
    """Return the externally processed/validated mesh shared by files and viewers."""
    if not isinstance(require_closed, (bool, np.bool_)):
        raise ValueError("require_closed must be a boolean")
    trimesh = _mesh_library()
    vertices, faces = tessellate_surfaces(surfaces, nu=nu, nv=nv)
    extent = float(np.ptp(vertices, axis=0).max())
    if not np.isfinite(extent) or extent <= 0:
        raise ValueError("Combined surface extent must be finite and positive")
    origin = vertices.min(axis=0)
    if weld_tolerance is None:
        relative_tolerance = 1e-10
    else:
        if isinstance(weld_tolerance, (bool, np.bool_)) or not np.isscalar(weld_tolerance):
            raise ValueError("weld_tolerance must be a positive finite length")
        try:
            relative_tolerance = float(weld_tolerance) / extent
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("weld_tolerance must be a positive finite length") from error
    if not np.isfinite(relative_tolerance) or not 1e-15 <= relative_tolerance < 1:
        raise ValueError("weld_tolerance must be between 1e-15 times the extent and the extent")
    digits = int(np.ceil(-np.log10(relative_tolerance)))
    mesh = trimesh.Trimesh(vertices=(vertices - origin) / extent, faces=faces, process=False)
    mesh.merge_vertices(digits_vertex=digits)
    mesh.update_faces(mesh.unique_faces())
    mesh.update_faces(mesh.nondegenerate_faces(height=0))
    mesh.remove_unreferenced_vertices()
    if not len(mesh.faces):
        raise ValueError("No triangles remain after mesh processing; reduce weld_tolerance")
    if np.any(np.bincount(mesh.edges_unique_inverse) > 2):
        raise ValueError("Mesh contains non-manifold edges; export patches separately or revise the geometry")
    trimesh.repair.fix_winding(mesh)
    if not mesh.is_winding_consistent:
        raise ValueError("Mesh winding cannot be made consistent")
    if require_closed:
        if not mesh.is_watertight:
            raise ValueError("Mesh is not closed; boundary samples must match and all openings must be covered")
        labels = trimesh.graph.connected_component_labels(mesh.face_adjacency, node_count=len(mesh.faces))
        if len(np.unique(labels)) != 1:
            raise ValueError("Closed export supports one connected shell; disconnected or nested cavity shells are unsupported")
        trimesh.repair.fix_normals(mesh, multibody=False)
        if not mesh.is_volume:
            raise ValueError("Mesh must enclose a finite positive volume")
    positions = np.asarray(mesh.vertices) * extent + origin
    if np.max(np.abs(positions)) > np.finfo(np.float32).max:
        raise ValueError("Coordinates exceed STL float32 range; rescale the geometry")
    encoded = positions.astype(np.float32)
    if len(np.unique(encoded, axis=0)) != len(positions):
        raise ValueError("STL float32 precision merges vertices; recenter/rescale or reduce sampling")
    # Normalize encoded positions for a stable collapse check at tiny/large scales.
    check = (encoded.astype(float) - origin) / extent
    triangles = check[np.asarray(mesh.faces)]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    if np.any(np.linalg.norm(normals, axis=1) == 0):
        raise ValueError("STL float32 precision collapses triangles; recenter/rescale the geometry")
    encoded_mesh = trimesh.Trimesh(vertices=encoded.astype(float), faces=mesh.faces, process=False)
    if not np.allclose(np.linalg.norm(encoded_mesh.face_normals, axis=1), 1.0):
        raise ValueError("Coordinate scale prevents reliable mesh-library normals; rescale the geometry")
    return encoded_mesh
