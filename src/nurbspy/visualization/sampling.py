"""Bounded, geometry-aware display sampling; independent of camera and transport.

Refinement probes positions in batches and retains a structured parameter grid.
These are visual quality heuristics, not certified geometric error bounds.
"""

import numpy as np

PRESETS = {"fast": (0.003, 24.0), "auto": (0.0007, 12.0), "fine": (0.00018, 6.0)}
MAX_AXIS = 128
MAX_VERTICES = 16384


def validate_quality(quality):
    if not isinstance(quality, str) or quality not in PRESETS:
        raise ValueError("quality must be 'auto', 'fast', or 'fine'")
    return quality


def validate_preview_quality(quality, preview_resolution=None):
    if not isinstance(quality, str) or quality not in ("coarse", "full"):
        raise ValueError("preview_quality must be 'coarse' or 'full'")
    if quality == "full" and preview_resolution is not None:
        raise ValueError("preview_resolution cannot be combined with preview_quality='full'")
    return quality


def _axis(knots, degree, minimum=8):
    lo, hi = knots[degree], knots[-degree - 1]
    values = np.unique(
        np.r_[np.linspace(lo, hi, minimum), knots[(knots >= lo) & (knots <= hi)]]
    )
    if len(values) > MAX_AXIS:
        values = values[np.linspace(0, len(values) - 1, MAX_AXIS).round().astype(int)]
    return values


def _errors(points):
    """Deviation of quarter/midpoints from chords, plus change of direction."""
    start, end = points[:, :, 0], points[:, :, -1]
    t = np.linspace(0, 1, 5)[None, None, :]
    chord = start[:, :, None] * (1 - t) + end[:, :, None] * t
    distance = np.max(np.linalg.norm(points - chord, axis=0), axis=1)
    tangents = np.diff(points, axis=2)
    lengths = np.linalg.norm(tangents, axis=0)
    unit = tangents / np.maximum(lengths[None], np.finfo(float).tiny)
    cosine = np.sum(unit[:, :, :-1] * unit[:, :, 1:], axis=0)
    valid = (lengths[:, :-1] > 1e-14) & (lengths[:, 1:] > 1e-14)
    angle = np.max(np.where(valid, np.arccos(np.clip(cosine, -1, 1)), 0), axis=1)
    return distance, angle


def automatic_axes(geometry, quality="auto"):
    """Choose independent u/v densities with bounded vectorized probing."""
    tolerance, angle = PRESETS[validate_quality(quality)]
    surface = geometry.P.ndim == 3
    u = _axis(np.asarray(geometry.U), geometry.p)
    v = _axis(np.asarray(geometry.V), geometry.q) if surface else None
    scale = float(
        np.linalg.norm(
            np.ptp(np.asarray(geometry.P).reshape(geometry.P.shape[0], -1), axis=1)
        )
    )
    tolerance *= max(scale, np.finfo(float).tiny)
    angle = np.deg2rad(angle)
    capped = len(np.unique(geometry.U)) > MAX_AXIS or (
        surface and len(np.unique(geometry.V)) > MAX_AXIS
    )
    # A fixed iteration bound also limits expensive geometry evaluations.
    for _ in range(5):
        refinements = []
        for direction, axis in enumerate((u, v) if surface else (u,)):
            parameters = (
                axis[:-1, None] + np.diff(axis)[:, None] * np.linspace(0, 1, 5)
            ).ravel()
            if surface:
                other = v if direction == 0 else u
                probes = np.unique(np.r_[other, (other[:-1] + other[1:]) / 2])
                probes = probes[
                    np.linspace(0, len(probes) - 1, min(len(probes), 9))
                    .round()
                    .astype(int)
                ]
                a, b = np.meshgrid(parameters, probes, indexing="ij")
                values = (
                    geometry.get_value(a.ravel(), b.ravel())
                    if direction == 0
                    else geometry.get_value(b.ravel(), a.ravel())
                )
                values = np.asarray(values).reshape(
                    geometry.P.shape[0], len(axis) - 1, 5, len(probes)
                )
                distances, angles = _errors(
                    values.transpose(0, 1, 3, 2).reshape(geometry.P.shape[0], -1, 5)
                )
                distances = distances.reshape(-1, len(probes)).max(axis=1)
                angles = angles.reshape(-1, len(probes)).max(axis=1)
            else:
                distances, angles = _errors(
                    np.asarray(geometry.get_value(parameters)).reshape(
                        geometry.P.shape[0], -1, 5
                    )
                )
            refinements.append(np.maximum(distances / tolerance, angles / angle))
        if surface:
            # Straight parameter lines can still form a twisted bilinear patch.
            # Probe coarse cells to detect their diagonal triangulation error.
            a = u[np.linspace(0, len(u) - 1, min(len(u), 9)).round().astype(int)]
            b = v[np.linspace(0, len(v) - 1, min(len(v), 9)).round().astype(int)]
            x, y = np.meshgrid(a, b, indexing="ij")
            corners = np.asarray(geometry.get_value(x.ravel(), y.ravel())).reshape(
                geometry.P.shape[0], len(a), len(b)
            )
            mixed = (
                np.linalg.norm(
                    corners[:, :-1, :-1]
                    - corners[:, :-1, 1:]
                    - corners[:, 1:, :-1]
                    + corners[:, 1:, 1:],
                    axis=0,
                )
                / 4
            )
            # Scale the coarse-cell estimate down to the current interval sizes.
            for direction, axis in enumerate((u, v)):
                coarse = a if direction == 0 else b
                per_interval = mixed.max(axis=1 if direction == 0 else 0)
                indices = np.clip(
                    np.searchsorted(coarse, (axis[:-1] + axis[1:]) / 2) - 1,
                    0,
                    len(coarse) - 2,
                )
                ratio = np.diff(axis) / np.diff(coarse)[indices]
                refinements[direction] = np.maximum(
                    refinements[direction], per_interval[indices] * ratio**2 / tolerance
                )
        changed = False
        for direction, axis in enumerate((u, v) if surface else (u,)):
            scores = refinements[direction]
            selected = np.flatnonzero(scores > 1)
            room = MAX_AXIS - len(axis)
            if len(selected) > room:
                capped = True
                selected = (
                    selected[np.argsort(scores[selected])[-room:]] if room else []
                )
            if len(selected):
                axis = np.sort(
                    np.r_[
                        axis,
                        (axis[np.asarray(selected)] + axis[np.asarray(selected) + 1])
                        / 2,
                    ]
                )
                if direction == 0:
                    u = axis
                else:
                    v = axis
                changed = True
        if not changed:
            break
    else:
        capped = True
    return u, v, capped


def preview_axes(u, v, count=20):
    def thin(axis):
        return axis[
            np.linspace(0, len(axis) - 1, min(len(axis), count)).round().astype(int)
        ]

    return thin(u), thin(v) if v is not None else None


def grid(u, v=None):
    """Flatten a rectangular grid and build its triangles in row-major order."""
    u = np.asarray(u, dtype=float)
    if v is None:
        return u, None, []
    v = np.asarray(v, dtype=float)
    a, b = np.meshgrid(u, v, indexing="ij")
    indices = np.arange(a.size).reshape(a.shape)
    x, y, z, w = (
        indices[:-1, :-1].ravel(),
        indices[1:, :-1].ravel(),
        indices[1:, 1:].ravel(),
        indices[:-1, 1:].ravel(),
    )
    triangles = (
        np.stack((np.stack((x, y, z), axis=1), np.stack((x, z, w), axis=1)), axis=1)
        .reshape(-1, 3)
        .tolist()
    )
    return a.ravel(), b.ravel(), triangles


def descriptor(u, v, capped=False):
    pu, pv = preview_axes(u, v)
    return {
        "full": {"u": u.tolist(), "v": v.tolist() if v is not None else None},
        "preview": {"u": pu.tolist(), "v": pv.tolist() if pv is not None else None},
        "capped": bool(capped),
    }


def sampling_from_descriptor(description, geometry):
    result = {}
    for quality in ("full", "preview"):
        axes = []
        for name, knots, degree in [
            ("u", geometry.U, geometry.p),
            *([("v", geometry.V, geometry.q)] if geometry.P.ndim == 3 else []),
        ]:
            axis = np.asarray(description[quality][name])
            if (
                axis.dtype.kind not in "iuf"
                or axis.ndim != 1
                or not 2 <= len(axis) <= MAX_AXIS
                or not np.isfinite(axis).all()
                or np.any(np.diff(axis) <= 0)
                or axis[0] != knots[degree]
                or axis[-1] != knots[-degree - 1]
            ):
                raise ValueError("Invalid display sampling axes")
            axes.append(axis)
        result[quality] = grid(*axes)
    return result


def topologies(description):
    result = {}
    for quality in ("full", "preview"):
        axes = description[quality]
        u, v, triangles = grid(axes["u"], axes["v"])
        result[quality] = dict(
            count=len(u),
            triangles=triangles,
            shape=[len(axes["u"]), len(axes["v"])] if v is not None else [len(u)],
        )
    return result
