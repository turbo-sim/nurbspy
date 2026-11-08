"""Transport-neutral geometry validation, sampling, and native reconstruction.

No GUI framework, HTTP application, or mutable browser session is owned here.
The same native Python evaluation path serves desktop and embedded editors.
"""

from copy import deepcopy
import re
import base64
import hashlib
from threading import RLock
from collections import OrderedDict
from .sampling import (
    validate_quality,
    validate_preview_quality,
    automatic_axes,
    descriptor,
    sampling_from_descriptor,
    topologies,
)
import numpy as np


class _GeometryModel:
    """Independent geometry snapshot and validated display evaluation."""

    def __init__(
        self,
        geometry,
        *,
        id,
        title=None,
        resolution=None,
        quality="auto",
        height="520px",
        preview_resolution=None,
        preview_quality="coarse",
        display=None,
    ):
        if not isinstance(id, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", id):
            raise ValueError(
                "id must start with a letter and contain letters, digits, '_' or '-'"
            )
        if resolution is not None and (
            isinstance(resolution, bool)
            or not isinstance(resolution, (int, np.integer))
            or not 8 <= resolution <= 100
        ):
            raise ValueError("resolution must be None or an integer between 8 and 100")
        if isinstance(height, (int, float)) and not isinstance(height, bool):
            if not np.isfinite(height) or height <= 0:
                raise ValueError("height must be positive")
            height = f"{height}px"
        if not isinstance(height, str) or not height.strip():
            raise ValueError(
                "height must be a CSS length or a positive number of pixels"
            )
        try:
            points, weights = np.asarray(geometry.P), np.asarray(geometry.W)
        except AttributeError as error:
            raise TypeError("Expected a nurbspy curve or surface") from error
        if points.ndim not in (2, 3) or points.shape[0] not in (2, 3):
            raise ValueError(
                "The explorer requires a curve or surface in two or three dimensions"
            )
        self.preview_quality = validate_preview_quality(preview_quality, preview_resolution)
        if preview_resolution is None and resolution is not None:
            preview_resolution = (resolution if preview_quality == "full" else
                                  min(resolution, 20 if points.ndim == 3 else 40))
        if preview_resolution is not None and (
            isinstance(preview_resolution, bool)
            or not isinstance(preview_resolution, (int, np.integer))
            or not 8 <= preview_resolution <= (resolution or 128)
        ):
            raise ValueError(
                "preview_resolution must be an integer from 8 to resolution"
            )
        # Viewing uses one spatial model; planar inputs are promoted only here.
        points = np.pad(
            points, ((0, 3 - points.shape[0]), *((0, 0),) * (points.ndim - 1))
        )
        self.preview_resolution = preview_resolution
        self.id, self.resolution, self.height = id, resolution, height
        self.quality = validate_quality(quality)
        self._sample_cache = OrderedDict()
        self._sample_lock = RLock()
        self.surface = points.ndim == 3
        from .analysis import display_options

        self.display = display_options(display, self.surface)
        if display is None or "quality" not in display:
            self.display["quality"] = self.quality
        self.quality = self.display["quality"]
        self.dimension = points.shape[0]
        self.title = str(
            title or ("Surface explorer" if self.surface else "Curve explorer")
        )
        self.viewport_id, self.edit_id = f"{id}-viewport", f"{id}-edits"
        self.scene_id, self.state_id, self.ready_id = (
            f"{id}-scene",
            f"{id}-state",
            f"{id}-ready",
        )
        self._point_shape, self._weight_shape = points.shape, weights.shape
        model = {
            "type": "NurbsSurface" if self.surface else "NurbsCurve",
            "control_points": points.tolist(),
            "weights": weights.tolist(),
        }
        if self.surface:
            model.update(
                u_degree=int(geometry.p),
                v_degree=int(geometry.q),
                u_knots=np.asarray(geometry.U).tolist(),
                v_knots=np.asarray(geometry.V).tolist(),
            )
        else:
            model.update(degree=int(geometry.p), knots=np.asarray(geometry.U).tolist())
        self._initial = deepcopy(model)
        self._validated_arrays(model)
        if self.surface:
            from .analysis import isocurve_parameters

            for direction in ("u", "v"):
                isocurve_parameters(
                    geometry, direction, self.display["isocurves_" + direction]
                )
        if resolution is None:
            self._sampling, self._description = self._choose_sampling(
                self.geometry_from_state(), self.quality
            )
        else:
            self._description = None
            self._sampling = {
                level: self._make_sampling(model, count)
                for level, count in (
                    ("full", resolution),
                    ("preview", self.preview_resolution),
                )
            }
        # Evaluate once now so invalid input fails at construction, not in a callback.
        self._initial_scene = self.evaluate_scene(model)

    def _choose_sampling(self, geometry, quality):
        quality = validate_quality(quality)
        key = (
            quality,
            self.preview_quality,
            self.preview_resolution,
            hashlib.sha256(
                np.asarray(geometry.P).tobytes() + np.asarray(geometry.W).tobytes()
            ).digest(),
        )
        with self._sample_lock:
            cached = self._sample_cache.get(key)
        if cached is None:
            u, v, capped = automatic_axes(geometry, quality)
            description = descriptor(u, v, capped)
            if self.preview_quality == "full":
                description["preview"] = deepcopy(description["full"])
            elif self.preview_resolution is not None:
                from .sampling import preview_axes

                pu, pv = preview_axes(u, v, self.preview_resolution)
                description["preview"] = {
                    "u": pu.tolist(),
                    "v": pv.tolist() if pv is not None else None,
                }
            cached = sampling_from_descriptor(description, geometry), description
        with self._sample_lock:
            self._sample_cache[key] = cached
            self._sample_cache.move_to_end(key)
            if len(self._sample_cache) > 6:
                self._sample_cache.popitem(last=False)
        sampling, description = cached
        return sampling, deepcopy(description)

    def _request_sampling(self, geometry, request, refine=False):
        if self.resolution is not None:
            return self._sampling, None
        if refine:
            return self._choose_sampling(
                geometry, request.get("draw_quality", self.quality)
            )
        description = request.get("sampling")
        if description is None:
            return self._sampling, deepcopy(self._description)
        if not isinstance(description, dict):
            raise ValueError("Invalid display sampling descriptor")
        if self.preview_quality == "full":
            description = deepcopy(description)
            description["preview"] = deepcopy(description["full"])
        with self._sample_lock:
            cached = tuple(self._sample_cache.values())
        for sampling, previous in cached:
            if (
                description.get("full") == previous["full"]
                and description.get("preview") == previous["preview"]
            ):
                return sampling, deepcopy(description)
        return sampling_from_descriptor(description, geometry), deepcopy(description)

    def render_packet(self, request):
        """Compact IPC packet, including new topology after automatic refinement."""
        positions, model, quality, revision, description = self._render_result(request)
        result = dict(
            revision=revision,
            quality=quality,
            vertices=base64.b64encode(positions.tobytes()).decode("ascii"),
        )
        if description is not None and quality == "full":
            result.update(
                sampling=description,
                topologies=topologies(description),
                draw_quality=request.get("draw_quality", self.quality),
            )
        return result

    @property
    def fields(self):
        from .analysis import CURVE_FIELDS, SURFACE_FIELDS

        return {
            name: label
            for name, (label, _) in (
                SURFACE_FIELDS if self.surface else CURVE_FIELDS
            ).items()
        }

    @property
    def colormaps(self):
        from .analysis import colormap_tables

        return deepcopy(colormap_tables())

    def inspect(self, request):
        """Evaluate requested scientific display data without accepting an edit."""
        from .analysis import (
            scalar_field,
            frenet_frames,
            export_preview,
            control_point_normal,
            surface_isocurves,
        )

        allowed = {
            "revision",
            "quality",
            "control_points",
            "weights",
            "field",
            "frames",
            "frame_count",
            "frame_scale",
            "export_mesh",
            "nu",
            "nv",
            "normal_index",
            "sampling",
            "draw_quality",
            "tessellate",
            "isocurves",
            "isocurves_u",
            "isocurves_v",
        }
        if not isinstance(request, dict) or set(request) - allowed:
            raise ValueError("Unknown inspection fields")
        revision = request.get("revision", 0)
        if isinstance(revision, bool) or not isinstance(revision, int) or revision < 0:
            raise ValueError("Inspection revision must be a nonnegative integer")
        quality = request.get("quality", "full")
        if not isinstance(quality, str) or quality not in self._sampling:
            raise ValueError("quality must be full or preview")
        model = self.initial_state
        model["control_points"], model["weights"] = request.get(
            "control_points"
        ), request.get("weights")
        geometry = self.geometry_from_state(model)
        if request.get("tessellate"):
            result = self.render_packet(
                dict(
                    revision=max(revision, 1),
                    quality="full",
                    action="edit",
                    control_points=model["control_points"],
                    weights=model["weights"],
                    draw_quality=request.get("draw_quality", self.quality),
                )
            )
            result["revision"] = revision
            return result
        sampling, _ = self._request_sampling(geometry, request)
        u, v, _ = sampling[quality]
        result = {"revision": revision, "quality": quality}
        enabled = request.get("isocurves", False)
        if not isinstance(enabled, bool):
            raise ValueError("Inspection isocurves must be a boolean")
        if enabled:
            if not self.surface:
                raise ValueError("Isocurves require a surface")
            result["isocurves"] = surface_isocurves(
                geometry,
                u,
                v,
                fixed_u=request.get("isocurves_u", self.display["isocurves_u"]),
                fixed_v=request.get("isocurves_v", self.display["isocurves_v"]),
            )
        if "normal_index" in request:
            result["normal"] = control_point_normal(geometry, request["normal_index"])
        if request.get("field", "solid") != "solid":
            result["field"] = scalar_field(geometry, u, v, request["field"])
        if request.get("frames"):
            if self.surface:
                raise ValueError("Frenet frames require a curve")
            result["frames"] = frenet_frames(
                geometry, request.get("frame_count", 8), request.get("frame_scale", 0.1)
            )
        if request.get("export_mesh"):
            if not self.surface:
                raise ValueError("Export mesh requires a surface")
            result["mesh"] = export_preview(
                geometry, request.get("nu", 40), request.get("nv", 40)
            )
        return result

    @property
    def initial_state(self):
        """An independent JSON-compatible model, using coordinate-first arrays."""
        return deepcopy(self._initial)

    def _validated_arrays(self, model):
        if not isinstance(model, dict) or set(model) != set(self._initial):
            raise ValueError("Model fields do not match the explorer's geometry")
        for name in self._initial:
            if (
                name not in ("control_points", "weights")
                and model[name] != self._initial[name]
            ):
                raise ValueError(
                    "Only control-point coordinates and weights may be edited"
                )
        try:
            points, weights = np.asarray(model["control_points"]), np.asarray(
                model["weights"]
            )
        except (TypeError, ValueError) as error:
            raise ValueError(
                "Coordinates and weights must form rectangular numeric arrays"
            ) from error
        if points.shape != self._point_shape or weights.shape != points.shape[1:]:
            raise ValueError(
                "Control-point and weight shapes must match the original geometry"
            )
        if any(array.dtype.kind not in "iuf" for array in (points, weights)):
            raise ValueError("Coordinates and weights must be real numbers")
        points, weights = points.astype(float), weights.astype(float)
        if (
            not np.all(np.isfinite(points))
            or not np.all(np.isfinite(weights))
            or np.any(weights <= 0)
        ):
            raise ValueError(
                "Coordinates must be finite and weights finite and positive"
            )
        return points, weights

    def geometry_from_state(self, model=None):
        """Reconstruct an independent native NumPy geometry from browser state."""
        from ..numpy import NurbsCurve, NurbsSurface

        model = self._initial if model is None else model
        points, weights = self._validated_arrays(model)
        if self.surface:
            return NurbsSurface(
                control_points=points.copy(),
                weights=weights.copy(),
                u_degree=model["u_degree"],
                v_degree=model["v_degree"],
                u_knots=np.asarray(model["u_knots"], dtype=float),
                v_knots=np.asarray(model["v_knots"], dtype=float),
            )
        return NurbsCurve(
            control_points=points.copy(),
            weights=weights.copy(),
            degree=model["degree"],
            knots=np.asarray(model["knots"], dtype=float),
        )

    def evaluate_scene(self, model=None, *, revision=0, error=None):
        """Evaluate a model in Python and return rendering-only browser data."""
        model = self._initial if model is None else model
        geometry = self.geometry_from_state(model)
        sampling, description = self._request_sampling(geometry, {}, refine=True)
        u, v, triangles = sampling["full"]
        samples = geometry.get_value(u, v) if self.surface else geometry.get_value(u)
        if self.surface:
            control = np.arange(geometry.W.size).reshape(geometry.W.shape)
            control_lines = [row.tolist() for row in control] + [
                row.tolist() for row in control.T
            ]
        else:
            control_lines = [list(range(geometry.W.size))]
        samples = np.asarray(samples)
        if np.iscomplexobj(samples) or not np.all(np.isfinite(samples)):
            raise ValueError(
                "The edited geometry did not evaluate to finite real coordinates"
            )

        def display(array):
            array = np.asarray(array).reshape(self.dimension, -1)
            return np.pad(array, ((0, 3 - self.dimension), (0, 0))).T.tolist()

        return {
            "revision": int(revision),
            "model": deepcopy(model),
            "control_points": display(geometry.P),
            "control_shape": list(geometry.W.shape),
            "points": display(samples),
            "triangles": deepcopy(triangles),
            "quality": "full",
            "topologies": (
                topologies(description)
                if description is not None
                else {
                    quality: {
                        "triangles": deepcopy(grid[2]),
                        "count": int(grid[0].size),
                    }
                    for quality, grid in sampling.items()
                }
            ),
            "control_lines": control_lines,
            "dimension": self.dimension,
            "error": error,
            "edit_store": self.edit_id,
            "title": self.title,
            "colormaps": self.colormaps,
            "fields": self.fields,
            "display": deepcopy(self.display),
            "sampling": description,
            "automatic": self.resolution is None,
            "draw_quality": self.quality,
        }

    def _make_sampling(self, model, resolution):
        """Cache only parameter grids and topology; evaluation remains in nurbspy."""
        p = model["u_degree"] if self.surface else model["degree"]
        knots = model["u_knots"] if self.surface else model["knots"]
        u = np.linspace(
            knots[p], knots[-p - 1], resolution if self.surface else 5 * resolution
        )
        v, triangles = None, []
        if self.surface:
            q, knots_v = model["v_degree"], model["v_knots"]
            v = np.linspace(knots_v[q], knots_v[-q - 1], resolution)
            u, v = np.meshgrid(u, v, indexing="ij")
            grid = np.arange(u.size).reshape(u.shape)
            a, b = grid[:-1, :-1].ravel(), grid[1:, :-1].ravel()
            c, d = grid[1:, 1:].ravel(), grid[:-1, 1:].ravel()
            triangles = (
                np.stack(
                    (np.stack((a, b, c), axis=1), np.stack((a, c, d), axis=1)), axis=1
                )
                .reshape(-1, 3)
                .tolist()
            )
            u, v = u.ravel(), v.ravel()
            v.setflags(write=False)
        u.setflags(write=False)
        return u, v, triangles

    def _render_result(self, request):
        """Validate a stateless compact request and evaluate display vertices in Python.

        All scientific evaluation uses the native NumPy geometry. Float32 is
        solely the browser display encoding; the accepted model stays float64.
        No session geometry or scientific results are stored on this adapter.
        """
        if not isinstance(request, dict):
            raise ValueError("An edit request must be an object")
        if set(request) - {
            "revision",
            "action",
            "quality",
            "control_points",
            "weights",
            "sampling",
            "draw_quality",
        }:
            raise ValueError("Unknown edit fields; topology and knots cannot be edited")
        revision = request.get("revision")
        if isinstance(revision, bool) or not isinstance(revision, int) or revision < 1:
            raise ValueError("Edit revision must be a positive integer")
        quality = request.get("quality", "full")
        if quality not in self._sampling:
            raise ValueError("quality must be full or preview")
        model = self.initial_state
        action = request.get("action", "edit")
        if action == "edit":
            if "control_points" not in request or "weights" not in request:
                raise ValueError("Coordinates and weights are required")
            model["control_points"], model["weights"] = (
                request["control_points"],
                request["weights"],
            )
        elif action != "reset":
            raise ValueError("Unknown editor action")
        geometry = self.geometry_from_state(model)
        sampling, description = self._request_sampling(
            geometry, request, refine=quality == "full"
        )
        u, v, _ = sampling[quality]
        samples = np.asarray(
            geometry.get_value(u, v) if self.surface else geometry.get_value(u)
        )
        if np.iscomplexobj(samples) or not np.all(np.isfinite(samples)):
            raise ValueError(
                "The edited geometry did not evaluate to finite real coordinates"
            )
        with np.errstate(over="ignore", invalid="ignore"):
            positions = np.pad(samples, ((0, 3 - self.dimension), (0, 0))).T.astype(
                "<f4"
            )
        if not np.all(np.isfinite(positions)):
            raise ValueError("Coordinates exceed the browser display range")
        return positions, model, quality, revision, description

    def _render_update(self, request):
        return self._render_result(request)[:4]

    def process_edit(self, request, current_state=None):
        """Validate and evaluate an atomic browser request without shared state.

        Returns ``(scene, model)``. An invalid request acknowledges its revision
        and retains the last accepted model, allowing the browser queue to resume.
        """
        current_state = (
            self.initial_state if current_state is None else deepcopy(current_state)
        )
        # A browser Store is untrusted too. Fall back to the original model if it
        # was modified independently of this explorer's validation callback.
        try:
            self._validated_arrays(current_state)
        except (ValueError, TypeError):
            current_state = self.initial_state
        revision = request.get("revision", 0) if isinstance(request, dict) else 0
        valid_revision = (
            not isinstance(revision, bool)
            and isinstance(revision, int)
            and revision >= 1
        )
        try:
            if not isinstance(request, dict) or not valid_revision:
                raise ValueError("Edit revision must be a positive integer")
            action = request.get("action", "edit")
            if action == "reset":
                model = self.initial_state
            elif action == "edit":
                model = deepcopy(request.get("model"))
                if not isinstance(model, dict):
                    raise ValueError("An edit must include a complete geometry model")
            else:
                raise ValueError("Unknown editor action")
            scene = self.evaluate_scene(model, revision=revision)
            return scene, model
        except (ValueError, TypeError, KeyError, OverflowError) as error:
            scene = self.evaluate_scene(
                current_state,
                revision=revision if valid_revision else 0,
                error=str(error),
            )
            return scene, current_state
