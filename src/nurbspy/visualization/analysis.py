"""Native Python quantities and export previews for the geometry viewer."""

from functools import lru_cache
import base64
import re

import numpy as np
from scipy.integrate import cumulative_trapezoid

CURVE_FIELDS = {
    "u": ("Parameter u", ""),
    "curvature": ("Curvature", "1/length"),
    "torsion": ("Torsion", "1/length"),
    "arc_length": ("Cumulative arc length", "length"),
    "speed": ("Parameter speed", "length/parameter"),
}
SURFACE_FIELDS = {
    "u": ("Parameter u", ""),
    "v": ("Parameter v", ""),
    "mean_curvature": ("Mean curvature", "1/length"),
    "gaussian_curvature": ("Gaussian curvature", "1/length²"),
    "principal_min": ("Minimum principal curvature", "1/length"),
    "principal_max": ("Maximum principal curvature", "1/length"),
}


def display_options(options, surface):
    """Validate optional initial UI settings; they never alter model geometry."""
    defaults = dict(
        view="3d",
        axes=False,
        grid=True,
        quantity="solid",
        colormap="viridis",
        geometry_color="#0072bd" if surface else "#000000",
        control_color="#ff7f0e",
        selected_color="#2ca02c",
        opacity=1.0,
        curve_depth_test=True,
        mesh="off",
        mesh_u=40,
        mesh_v=40,
        frames=False,
        frame_count=8,
        frame_scale=0.1,
        control_points=True,
        control_polygon=True,
        boundary=True,
        reverse=False,
        limits=None,
    )
    if options is not None:
        if not isinstance(options, dict) or set(options) - set(defaults):
            raise ValueError("Unknown initial display options")
        defaults.update(options)
    choices = dict(
        view=("3d", "xy", "yz", "zx"),
        quantity=("solid", *(SURFACE_FIELDS if surface else CURVE_FIELDS)),
        colormap=tuple(colormap_tables()),
        mesh=("off", "display", "export") if surface else ("off",),
    )
    for name, values in choices.items():
        if defaults[name] not in values:
            raise ValueError(f"Invalid display {name}")
    for name in ("axes", "grid", "frames", "control_points", "control_polygon", "boundary", "reverse", "curve_depth_test"):
        if not isinstance(defaults[name], bool):
            raise ValueError(f"Display {name} must be a boolean")
    limits = defaults["limits"]
    opacity = defaults["opacity"]
    if isinstance(opacity, bool) or not isinstance(opacity, (int, float)) or not np.isfinite(opacity) or not 0 <= opacity <= 1:
        raise ValueError("Display opacity must be between zero and one")
    if limits is not None:
        if (not isinstance(limits, (list, tuple)) or len(limits) != 2
                or any(isinstance(x, bool) or not isinstance(x, (int, float)) or not np.isfinite(x) for x in limits)
                or limits[0] >= limits[1]):
            raise ValueError("Display limits must be two finite increasing numbers or None")
        defaults["limits"] = list(limits)
    if surface and defaults["frames"]:
        raise ValueError("Frenet frames require a curve")
    for name in ("geometry_color", "control_color", "selected_color"):
        if not isinstance(defaults[name], str) or not re.fullmatch(
            r"#[0-9a-fA-F]{6}", defaults[name]
        ):
            raise ValueError(f"Display {name} must be a six-digit hex colour")
    for name, low, high in (
        ("mesh_u", 2, 160),
        ("mesh_v", 2, 160),
        ("frame_count", 2, 30),
    ):
        value = defaults[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or not low <= value <= high
        ):
            raise ValueError(f"Display {name} must be an integer from {low} to {high}")
    value = defaults["frame_scale"]
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not np.isfinite(value)
        or not 0.005 <= value <= 0.5
    ):
        raise ValueError("Display frame_scale must be from 0.005 to 0.5")
    return defaults


@lru_cache(maxsize=1)
def colormap_tables():
    """Small display lookup tables; no Matplotlib window or pyplot is needed."""
    from matplotlib import colormaps

    names = ("viridis", "plasma", "magma", "inferno", "cividis", "coolwarm", "RdBu")
    return {
        name: np.round(colormaps[name](np.linspace(0, 1, 128))[:, :3], 6).tolist()
        for name in names
    }


def _regular_samples(geometry, u, order, v=None):
    """Mark knot joins without the differentiability needed by a diagnostic."""
    valid = np.ones(np.shape(u), dtype=bool)
    directions = [(geometry.U, geometry.p, u)]
    if v is not None:
        directions.append((geometry.V, geometry.q, v))
    for knots, degree, values in directions:
        start, end = knots[degree], knots[-degree - 1]
        unique, counts = np.unique(knots, return_counts=True)
        for knot, multiplicity in zip(unique, counts):
            if start < knot < end and degree - multiplicity < order:
                valid &= ~np.isclose(values, knot, rtol=0, atol=(end - start) * 1e-12)
    return valid


def scalar_field(geometry, u, v, name):
    """Evaluate a selected field with native nurbspy derivatives and geometry.

    Lengths use the model's coordinate units. Arc length is cumulative numerical
    quadrature on a dense, knot-aware grid. Nonregular/undefined values are null
    in the JSON result, not invented zero values.
    """
    fields = SURFACE_FIELDS if v is not None else CURVE_FIELDS
    if name not in fields:
        raise ValueError("Unknown quantity for this geometry")
    order = 0
    with np.errstate(all="ignore"):
        if name == "u":
            values = u
        elif name == "v":
            values = v
        elif name == "speed":
            order = 1
            values = np.linalg.norm(geometry.get_derivative(u, 1), axis=0)
        elif name == "arc_length":
            grid, distance = _arc_grid(geometry)
            values = np.interp(u, grid, distance)
        elif name == "curvature":
            order = 2
            values = geometry.get_curvature(u)
        elif name == "torsion":
            order = 3
            values = geometry.get_torsion(u)
        else:
            order = 2
            mean, gaussian = geometry.get_curvature(u, v)
            if name == "mean_curvature":
                values = mean
            elif name == "gaussian_curvature":
                values = gaussian
            else:
                root = np.sqrt(np.maximum(mean * mean - gaussian, 0))
                values = mean - root if name == "principal_min" else mean + root
    values = np.asarray(values, dtype=float).ravel()
    valid = np.isfinite(values) & _regular_samples(geometry, u, order, v)
    label, unit = fields[name]
    finite = values[valid]
    return {
        "name": name,
        "label": label,
        "unit": unit,
        "values": [float(value) if ok else None for value, ok in zip(values, valid)],
        "range": [float(finite.min()), float(finite.max())] if finite.size else None,
        "undefined": int((~valid).sum()),
    }


def _arc_grid(geometry):
    knots = np.asarray(geometry.U)
    start, end = knots[geometry.p], knots[-geometry.p - 1]
    grid = np.unique(
        np.r_[np.linspace(start, end, 1001), knots[(knots >= start) & (knots <= end)]]
    )
    speed = np.linalg.norm(geometry.get_derivative(grid, 1), axis=0)
    distance = cumulative_trapezoid(speed, grid, initial=0)
    return grid, distance


def frenet_frames(geometry, count=8, scale=0.1):
    """Sample native T/N/B frames at approximately equal arc-length intervals."""
    if isinstance(count, bool) or not isinstance(count, int) or not 2 <= count <= 30:
        raise ValueError("Frame count must be an integer from 2 to 30")
    if (
        isinstance(scale, bool)
        or not isinstance(scale, (float, int))
        or not np.isfinite(scale)
        or not 0.005 <= scale <= 0.5
    ):
        raise ValueError("Frame scale must be from 0.005 to 0.5")
    grid, distance = _arc_grid(geometry)
    if not np.all(np.isfinite(distance)) or distance[-1] <= 0:
        return {"frames": [], "omitted": count}
    u = np.interp(np.linspace(0, distance[-1], count), distance, grid)
    with np.errstate(all="ignore"):
        tangent = np.asarray(geometry.get_tangent(u))
        normal = np.asarray(geometry.get_normal(u))
        binormal = np.asarray(geometry.get_binormal(u))
    positions = np.asarray(geometry.get_value(u))
    valid = _regular_samples(geometry, u, 2)
    for values in (tangent, normal, binormal):
        valid &= np.all(np.isfinite(values), axis=0)
    frames = [
        {
            "origin": positions[:, i].tolist(),
            "tangent": tangent[:, i].tolist(),
            "normal": normal[:, i].tolist(),
            "binormal": binormal[:, i].tolist(),
        }
        for i in np.flatnonzero(valid)
    ]
    return {
        "frames": frames,
        "length": float(distance[-1] * scale),
        "omitted": int((~valid).sum()),
    }


def control_point_normal(geometry, index):
    """Native normal at the selected control point's fixed Greville location.

    This associates a parameter with a control point; it does not assert that
    the point lies on the geometry. Undefined normals are explicit nulls.
    """
    shape = geometry.W.shape
    count = int(np.prod(shape))
    if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < count:
        raise ValueError("Control-point index is out of range")
    surface = len(shape) == 2
    indices = np.unravel_index(index, shape)
    directions = [(geometry.U, geometry.p)]
    if surface:
        directions.append((geometry.V, geometry.q))
    parameters = []
    for i, (knots, degree) in zip(indices, directions):
        if degree == 0:
            return dict(
                direction=None,
                reason="A degree-zero direction has no geometric normal.",
            )
        parameters.append(
            float(
                np.clip(
                    np.mean(knots[i + 1 : i + degree + 1]),
                    knots[degree],
                    knots[-degree - 1],
                )
            )
        )
    u = np.array([parameters[0]])
    v = np.array([parameters[1]]) if surface else None
    order = 1 if surface else 2
    if not _regular_samples(geometry, u, order, v)[0]:
        return dict(
            direction=None,
            parameters=parameters,
            reason="The geometry is not sufficiently smooth at this parameter location.",
        )
    with np.errstate(all="ignore"):
        direction = np.asarray(
            geometry.get_normals(u, v) if surface else geometry.get_normal(u)
        )[:, 0]
    length = float(np.linalg.norm(direction))
    if not np.all(np.isfinite(direction)) or length < 1e-12:
        return dict(
            direction=None,
            parameters=parameters,
            reason="The normal is undefined here (zero curvature or a singular geometry).",
        )
    return dict(
        direction=(direction / length).tolist(), parameters=parameters, reason=""
    )


def export_preview(geometry, nu=40, nv=40):
    """Generate one export mesh and encode the STL from that exact mesh."""
    from ..mesh import _prepare_stl_mesh

    for count in (nu, nv):
        if (
            isinstance(count, bool)
            or not isinstance(count, int)
            or not 2 <= count <= 160
        ):
            raise ValueError("Export mesh samples must be integers from 2 to 160")
    mesh = _prepare_stl_mesh(geometry, nu=nu, nv=nv)
    vertices, triangles = np.asarray(mesh.vertices), np.asarray(mesh.faces)
    encoded = mesh.export(file_type="stl")
    return {
        "vertices": vertices.tolist(),
        "triangles": triangles.tolist(),
        "stl": base64.b64encode(encoded).decode("ascii"),
        "nu": nu,
        "nv": nv,
    }
