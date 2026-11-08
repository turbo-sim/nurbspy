"""Reproducible figures using the shared native geometry model and Three.js viewport."""

from copy import deepcopy
from io import StringIO
from pathlib import Path
import re

import numpy as np
import yaml

from .analysis import display_options
from .model import _GeometryModel
from ..serialization import _geometry_record


def _geometry(record):
    from ..numpy import NurbsCurve, NurbsSurface

    if not isinstance(record, dict) or record.get("type") not in ("NurbsCurve", "NurbsSurface"):
        raise ValueError("Expected a serialized curve or surface")
    cls = NurbsSurface if record["type"] == "NurbsSurface" else NurbsCurve
    return cls.from_yaml(StringIO(yaml.safe_dump(record)))


def _size(value, name):
    if type(value) is not int or not 64 <= value <= 16384:
        raise ValueError(f"{name} must be an integer between 64 and 16384 pixels")
    return value


def _camera(value):
    if value is None:
        return None
    defaults = dict(projection="3d", navigation="arcball", position=[3., -4., 3.],
                    target=[0., 0., 0.], up=[0., 0., 1.], zoom=1., fov=38.,
                    near=.001, far=10000., ortho_half=3.)
    if not isinstance(value, dict) or set(value) - set(defaults):
        raise ValueError("Unknown camera settings")
    defaults.update(deepcopy(value))
    if defaults["projection"] not in ("3d", "xy", "yz", "zx"):
        raise ValueError("Invalid camera projection")
    if defaults["navigation"] not in ("arcball", "orbit", "trackball"):
        raise ValueError("Invalid camera navigation")
    for key in ("position", "target", "up"):
        vector = np.asarray(defaults[key])
        if vector.shape != (3,) or vector.dtype.kind not in "iuf" or not np.isfinite(vector).all():
            raise ValueError(f"Camera {key} requires three finite numbers")
        defaults[key] = vector.astype(float).tolist()
    direction = np.subtract(defaults["target"], defaults["position"])
    if np.linalg.norm(direction) == 0 or np.linalg.norm(np.cross(direction, defaults["up"])) < 1e-12:
        raise ValueError("Camera needs distinct position/target and a nonparallel up vector")
    for key in ("zoom", "fov", "near", "far", "ortho_half"):
        number = defaults[key]
        if isinstance(number, bool) or not isinstance(number, (int, float)) or not np.isfinite(number) or number <= 0:
            raise ValueError(f"Camera {key} must be finite and positive")
        defaults[key] = float(number)
    if defaults["far"] <= defaults["near"] or defaults["fov"] >= 179:
        raise ValueError("Invalid camera clipping planes or field of view")
    return defaults


class Figure:
    """Named geometry snapshots, camera, display settings, and PNG export.

    Numerical geometry stays in the existing NumPy/JAX classes. This object
    coordinates the existing per-geometry model; no numerical evaluator or second
    renderer is introduced. Geometry inputs and returned snapshots are independent.
    """

    def __init__(self, geometry=None, *, name="geometry", title="NURBS figure",
                 resolution=40, preview_resolution=None, display=None,
                 width=1000, height=750, background="#ffffff"):
        self.title = str(title)
        self.resolution = resolution
        self.preview_resolution = preview_resolution
        self.rendering = dict(width=_size(width, "width"), height=_size(height, "height"),
                              background=background)
        self.camera = None
        self._objects = {}
        self._markers = []
        self.active = None
        if geometry is not None:
            self.add(geometry, name=name, display=display)
        self._validate_rendering()

    def _validate_rendering(self):
        if not isinstance(self.rendering, dict) or set(self.rendering) != {"width", "height", "background"}:
            raise ValueError("Invalid rendering settings")
        for key in ("width", "height"):
            _size(self.rendering[key], key)
        if not isinstance(self.rendering["background"], str) or not re.fullmatch(r"#[0-9a-fA-F]{6}", self.rendering["background"]):
            raise ValueError("Background must be a six-digit hex colour")
        if isinstance(self.resolution, bool) or not isinstance(self.resolution, int) or not 8 <= self.resolution <= 100:
            raise ValueError("resolution must be an integer between 8 and 100")
        if self.preview_resolution is not None and (type(self.preview_resolution) is not int or not 8 <= self.preview_resolution <= self.resolution):
            raise ValueError("preview_resolution must be between 8 and resolution")

    @property
    def names(self):
        return tuple(self._objects)

    def add(self, geometry, *, name, display=None, visible=True):
        """Add a named, independently styled copy of a curve or surface."""
        if not isinstance(name, str) or not name.strip() or name in self._objects:
            raise ValueError("Object names must be nonempty and unique")
        if type(visible) is not bool:
            raise ValueError("visible must be a boolean")
        model = _GeometryModel(geometry, id="validation", resolution=self.resolution,
                               preview_resolution=self.preview_resolution, display=display)
        self._objects[name] = dict(geometry=_geometry_record(geometry),
                                   display=model.display, visible=visible)
        self.active = self.active or name
        return self

    def geometry(self, name=None):
        """Return an independent NumPy geometry; use update() to accept changes."""
        return _geometry(self._objects[self.active if name is None else name]["geometry"])

    def scatter(self, points, *, color="#263238", size=3.0, depth_test=False):
        """Add read-only sample markers, retained in YAML and PNG exports.

        Inputs have shape (dimension, count) or are a sequence of Point objects.
        Planar coordinates are promoted to z=0. Size is in logical pixels;
        depth_test=False makes markers an overlay visible above the surfaces.
        Markers are annotations, not editable NURBS geometry objects.
        """
        from .._control_points import curve_control_points
        values = curve_control_points(points, np)
        if values.shape[0] not in (2, 3) or values.dtype.kind not in "iuf" or not np.isfinite(values).all():
            raise ValueError("Scatter points must contain finite real 2D or 3D coordinates")
        if not isinstance(color, str) or not re.fullmatch(r"#[0-9a-fA-F]{6}", color):
            raise ValueError("Scatter color must be a six-digit hex colour")
        if isinstance(size, bool) or not isinstance(size, (int, float)) or not np.isfinite(size) or not 0 < size <= 64:
            raise ValueError("Scatter size must be positive and at most 64 pixels")
        if type(depth_test) is not bool:
            raise ValueError("Scatter depth_test must be a boolean")
        values = np.pad(values, ((0, 3-values.shape[0]), (0, 0)))
        self._markers.append(dict(points=values.astype(float).tolist(), color=color,
                                  size=float(size), depth_test=depth_test))
        return self

    def settings(self, name=None):
        """Return an independent object's appearance/diagnostic settings."""
        return deepcopy(self._objects[self.active if name is None else name]["display"])

    def update(self, name=None, *, geometry=None, display=None, visible=None):
        """Replace geometry or merge settings, suitable for scripted frame loops."""
        name = self.active if name is None else name
        old = self._objects[name]
        merged = {**old["display"], **(display or {})}
        geometry = self.geometry(name) if geometry is None else geometry
        model = _GeometryModel(geometry, id="validation", resolution=self.resolution,
                               preview_resolution=self.preview_resolution, display=merged)
        visible = old["visible"] if visible is None else visible
        if type(visible) is not bool:
            raise ValueError("visible must be a boolean")
        self._objects[name] = dict(geometry=_geometry_record(geometry), display=model.display, visible=visible)
        return self

    def remove(self, name):
        del self._objects[name]
        if self.active == name:
            self.active = next(iter(self._objects), None)
        return self

    def set_camera(self, **settings):
        """Set camera fields; orientation is determined by position/target/up."""
        merged = {**_camera(self.camera or {}), **settings}
        if merged["projection"] != "3d":
            planes = {"xy": ([0, 0, 1], [0, 1, 0]), "yz": ([1, 0, 0], [0, 0, 1]),
                      "zx": ([0, 1, 0], [1, 0, 0])}
            if merged["projection"] in planes:
                direction, up = planes[merged["projection"]]
                if "position" not in settings:
                    distance = max(float(np.linalg.norm(np.subtract(merged["position"], merged["target"]))), 1.)
                    merged["position"] = (np.asarray(merged["target"]) + distance * np.asarray(direction)).tolist()
                if "up" not in settings:
                    merged["up"] = up
        self.camera = _camera(merged)
        return self

    def fit(self):
        """Request a fresh fit to all visible objects on the next render."""
        self.camera = None
        return self

    def to_state(self):
        """Return a versioned, JSON-compatible, self-contained figure snapshot."""
        self._validate_rendering()
        if not self._objects or self.active not in self._objects:
            raise ValueError("A figure requires at least one object and a valid active name")
        state = deepcopy(dict(format="nurbspy-figure", version=1, title=self.title,
                             resolution=self.resolution, preview_resolution=self.preview_resolution,
                             rendering=self.rendering, camera=_camera(self.camera), active=self.active,
                             objects=self._objects))
        if self._markers:
            state["markers"] = deepcopy(self._markers)
        return state

    @classmethod
    def from_state(cls, state):
        if not isinstance(state, dict) or set(state)-{"markers"} != {"format", "version", "title", "resolution", "preview_resolution", "rendering", "camera", "active", "objects"}:
            raise ValueError("Expected a versioned nurbspy figure document")
        if state["format"] != "nurbspy-figure" or type(state["version"]) is not int or state["version"] != 1:
            raise ValueError("Unsupported figure format or version")
        rendering = state["rendering"]
        if not isinstance(rendering, dict) or set(rendering) != {"width", "height", "background"}:
            raise ValueError("Invalid rendering settings")
        figure = cls(title=state["title"], resolution=state["resolution"],
                     preview_resolution=state["preview_resolution"], **rendering)
        if not isinstance(state["objects"], dict):
            raise ValueError("Figure objects must be a mapping")
        for name, item in state["objects"].items():
            if not isinstance(item, dict) or set(item) != {"geometry", "display", "visible"}:
                raise ValueError("Invalid figure object")
            figure.add(_geometry(item["geometry"]), name=name, display=item["display"], visible=item["visible"])
        markers = state.get("markers", [])
        if not isinstance(markers, list):
            raise ValueError("Figure markers must be a list")
        for marker in markers:
            if not isinstance(marker, dict) or set(marker) != {"points", "color", "size", "depth_test"}:
                raise ValueError("Invalid scatter annotation")
            figure.scatter(**marker)
        figure.active, figure.camera = state["active"], _camera(state["camera"])
        figure.to_state()
        return figure

    def to_yaml(self, destination=None):
        text = yaml.safe_dump(self.to_state(), sort_keys=False, allow_unicode=True)
        if destination is None:
            return text
        if hasattr(destination, "write"):
            destination.write(text)
            return None
        path = Path(destination)
        path.write_text(text, encoding="utf-8")
        return path

    @classmethod
    def from_yaml(cls, source):
        text = source.read() if hasattr(source, "read") else Path(source).read_text(encoding="utf-8")
        try:
            return cls.from_state(yaml.safe_load(text))
        except yaml.YAMLError as error:
            raise ValueError("Invalid figure YAML") from error

    def copy(self):
        return self.from_state(self.to_state())

    def _models(self, names=None):
        self.to_state()
        return {name: _GeometryModel(self.geometry(name), id=f"object{i}", title=name,
                                     resolution=self.resolution, preview_resolution=self.preview_resolution,
                                     display=item["display"])
                for i, (name, item) in enumerate(self._objects.items()) if names is None or name in names}

    def _payload(self, names=None):
        models = self._models(names)
        scenes = {}
        for name, model in models.items():
            scene = deepcopy(model._initial_scene)
            settings = model.display
            request = dict(revision=0, quality="full", **model.initial_state)
            request = {key: request[key] for key in ("revision", "quality", "control_points", "weights")}
            request.update(field=settings["quantity"], frames=settings["frames"],
                           frame_count=settings["frame_count"], frame_scale=settings["frame_scale"],
                           export_mesh=settings["mesh"] == "export", nu=settings["mesh_u"], nv=settings["mesh_v"])
            scene["diagnostics"] = model.inspect(request)
            scenes[name] = scene
        return dict(state=self.to_state(), scenes=scenes)

    def _accept(self, state):
        accepted = self.from_state(state)
        self.__dict__.update(accepted.__dict__)

    def show(self, *, width=1100, height=780):
        """Open a blocking native viewer; retain camera and display changes on close."""
        from .figure_hosts import open_figure
        open_figure(self, read_only=True, width=width, height=height)

    def edit(self, *, width=1100, height=780):
        """Apply commits the figure and returns geometry (a name mapping for several objects).

        Cancel and the window close button leave this figure unchanged.
        """
        from .figure_hosts import open_figure
        return open_figure(self, read_only=False, width=width, height=height)

    def savefig(self, destination, *, width=None, height=None, timeout=60):
        """Render a PNG without a visible window, with the saved framing and overlays.

        A different output aspect ratio adds background margins; it never refits
        the geometry. Requires a Playwright Chromium engine or installed Edge/Chrome.
        """
        from .figure_hosts import render_png
        if not hasattr(destination, "write") and Path(destination).suffix.lower() != ".png":
            raise ValueError("Figure.savefig supports PNG files only")
        width, height = self._image_size(width, height)
        data, state = render_png(self, width, height, timeout=timeout)
        if hasattr(destination, "write"):
            destination.write(data)
            result = None
        else:
            result = Path(destination)
            result.write_bytes(data)
        if self.camera is None:
            self.camera = _camera(state["camera"])
        return result

    def _image_size(self, width, height):
        if width is None and height is None:
            width, height = self.rendering["width"], self.rendering["height"]
        elif width is None:
            width = round(height * self.rendering["width"] / self.rendering["height"])
        elif height is None:
            height = round(width * self.rendering["height"] / self.rendering["width"])
        return _size(width, "width"), _size(height, "height")

    def saveframes(self, frames, directory, *, width=None, height=None,
                   prefix="frame", progress=None, timeout=60):
        """Export an iterable of Figures as PNGs while reusing one browser.

        A generator can update and yield this same Figure on every iteration.
        Frames are rendered before the generator advances. ``progress(index,
        path)`` is called after each saved frame, with a one-based index.
        """
        from .figure_hosts import render_frames
        if not isinstance(prefix, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", prefix):
            raise ValueError("Frame prefix must contain only letters, digits, '_' or '-'")
        if progress is not None and not callable(progress):
            raise TypeError("progress must be a callback or None")
        width, height = self._image_size(width, height)
        return render_frames(frames, Path(directory), width, height, prefix=prefix,
                             progress=progress, timeout=timeout)
