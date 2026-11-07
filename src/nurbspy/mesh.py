"""NURBS-specific surface sampling, independent of mesh libraries and viewers.

Sampling is count-based, includes interior knot breaks, and has no geometric
error guarantee. Vertices on different patches are not welded here.
"""
import numpy as np


__all__ = ["tessellate_surfaces"]


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
