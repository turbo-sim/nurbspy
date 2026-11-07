"""Versioned, backend-independent YAML geometry exchange (host-side only)."""
from pathlib import Path

import numpy as np
import yaml


_FORMAT = "nurbspy"
_VERSION = 1


def _numeric_array(value, name):
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a rectangular numeric array") from error
    if array.dtype.kind not in "iufc" or array.dtype.itemsize > (16 if array.dtype.kind == "c" else 8):
        raise ValueError(f"{name} must use integer, float16/32/64, or complex64/128 values")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite values")
    return array


def _encode_array(value, name):
    array = _numeric_array(value, name)
    record = {"dtype": str(array.dtype)}
    if np.iscomplexobj(array):
        record.update(real=array.real.tolist(), imag=array.imag.tolist())
    else:
        record["values"] = array.tolist()
    return record


def _decode_array(record, name):
    if not isinstance(record, dict) or not isinstance(record.get("dtype"), str):
        raise ValueError(f"{name} requires a dtype and numeric array data")
    try:
        dtype = np.dtype(record["dtype"])
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid dtype for {name}") from error
    if dtype.kind not in "iufc" or dtype.itemsize > (16 if dtype.kind == "c" else 8):
        raise ValueError(f"Unsupported numeric dtype for {name}")
    expected = {"dtype", "real", "imag"} if dtype.kind == "c" else {"dtype", "values"}
    if set(record) != expected:
        raise ValueError(f"Invalid array fields for {name}")
    try:
        if dtype.kind == "c":
            real, imag = (_numeric_array(record[key], name) for key in ("real", "imag"))
            if real.shape != imag.shape or np.iscomplexobj(real) or np.iscomplexobj(imag):
                raise ValueError(f"Real and imaginary arrays must have matching shapes for {name}")
            values = real + 1j * imag
        else:
            values = _numeric_array(record["values"], name)
            if np.iscomplexobj(values):
                raise ValueError(f"Complex data require a complex dtype for {name}")
            if dtype.kind in "iu":
                bounds = np.iinfo(dtype)
                if values.dtype.kind not in "iu" or np.any(values < bounds.min) or np.any(values > bounds.max):
                    raise ValueError(f"Values do not fit the integer dtype for {name}")
        with np.errstate(over="raise", invalid="raise"):
            result = values.astype(dtype)
        return _numeric_array(result, name).copy()
    except (TypeError, OverflowError, FloatingPointError) as error:
        raise ValueError(f"Invalid numeric data for {name}") from error


def _validate_parameters(kind, parameters):
    if kind not in ("NurbsCurve", "NurbsSurface") or not isinstance(parameters, dict):
        raise ValueError("Geometry type must be NurbsCurve or NurbsSurface")
    surface = kind == "NurbsSurface"
    fields = {"control_points", "weights", "u_degree", "v_degree", "u_knots", "v_knots"} if surface else {"control_points", "weights", "degree", "knots"}
    if set(parameters) != fields:
        raise ValueError("Geometry parameters do not match the declared type")
    P, W = parameters["control_points"], parameters["weights"]
    if P.ndim != (3 if surface else 2) or min(P.shape, default=0) < 1 or W.shape != P.shape[1:]:
        raise ValueError("Control points and weights have incompatible or empty shapes")
    directions = (("u_degree", "u_knots", P.shape[1]), ("v_degree", "v_knots", P.shape[2])) if surface else (("degree", "knots", P.shape[1]),)
    for degree_name, knots_name, count in directions:
        degree, knots = parameters[degree_name], parameters[knots_name]
        if isinstance(degree, bool) or not isinstance(degree, (int, np.integer)) or not 0 <= degree < count:
            raise ValueError(f"{degree_name} must be an integer smaller than the control-point count")
        if np.iscomplexobj(knots) or knots.ndim != 1 or knots.size != count + degree + 1:
            raise ValueError(f"{knots_name} has incompatible length or type")
        if np.any(np.diff(knots.astype(float)) < 0) or knots[degree] >= knots[-degree - 1]:
            raise ValueError(f"{knots_name} must be nondecreasing with a positive parameter domain")
        if np.any(np.unique(knots, return_counts=True)[1] > degree + 1):
            raise ValueError(f"{knots_name} has excessive knot multiplicity")


def _geometry_record(geometry):
    try:
        P, W = geometry.P, geometry.W
        kind = "NurbsSurface" if np.ndim(P) == 3 else "NurbsCurve"
        parameters = {"control_points": P, "weights": W}
        if kind == "NurbsSurface":
            parameters.update(u_degree=geometry.p, v_degree=geometry.q, u_knots=geometry.U, v_knots=geometry.V)
        else:
            parameters.update(degree=geometry.p, knots=geometry.U)
        arrays = {key: _numeric_array(value, key) if key not in ("degree", "u_degree", "v_degree") else value for key, value in parameters.items()}
        _validate_parameters(kind, arrays)
        encoded = {key: int(value) if key in ("degree", "u_degree", "v_degree") else _encode_array(value, key) for key, value in parameters.items()}
        return {"format": _FORMAT, "version": _VERSION, "type": kind, "parameters": encoded}
    except AttributeError as error:
        raise ValueError("Expected an initialized nurbspy curve or surface") from error


class GeometryYamlMixin:
    """Shared file I/O; these host-side methods are outside JIT/autodiff."""

    def to_yaml(self, destination=None):
        """Save geometry to a UTF-8 path/text stream, or return YAML text.

        Only mathematical geometry is stored, not Python objects, display
        settings, backend state, or the convenience Bezier/B-spline label.
        Complex arrays use explicit real/imaginary data with safe YAML types.
        """
        text = yaml.safe_dump(_geometry_record(self), sort_keys=False, allow_unicode=True)
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
        """Load a path or text stream into the class's NumPy or JAX backend.

        Use ``io.StringIO(text)`` to load YAML text returned by ``to_yaml()``.
        Arrays are reconstructed independently; arbitrary Python YAML tags are
        never loaded. Unsupported schema versions and malformed data fail.
        """
        if hasattr(source, "read"):
            text = source.read()
        else:
            text = Path(source).read_text(encoding="utf-8")
        try:
            record = yaml.safe_load(text)
        except yaml.YAMLError as error:
            raise ValueError("Invalid or unsupported YAML document") from error
        if not isinstance(record, dict) or set(record) != {"format", "version", "type", "parameters"}:
            raise ValueError("Expected a versioned nurbspy YAML geometry document")
        if record["format"] != _FORMAT or type(record["version"]) is not int or record["version"] != _VERSION:
            raise ValueError("Unsupported nurbspy YAML format or version")
        if record["type"] != cls.__name__:
            raise ValueError(f"Expected {cls.__name__}, found {record['type']!r}")
        encoded = record["parameters"]
        if not isinstance(encoded, dict):
            raise ValueError("Geometry parameters must be a mapping")
        parameters = {key: value if key in ("degree", "u_degree", "v_degree") else _decode_array(value, key) for key, value in encoded.items()}
        _validate_parameters(record["type"], parameters)
        geometry = cls(**parameters)
        if cls.__module__.startswith("nurbspy.jax."):
            # Constructors may promote weights or cast explicitly supplied knots
            # to the coordinate dtype. Restore the saved array leaves exactly.
            import equinox as eqx
            import jax.numpy as jnp
            names = ("control_points", "weights", "u_knots", "v_knots") if record["type"] == "NurbsSurface" else ("control_points", "weights", "knots")
            geometry = eqx.tree_at(
                lambda model: (model.P, model.W, model.U, model.V) if record["type"] == "NurbsSurface" else (model.P, model.W, model.U),
                geometry, replace=tuple(jnp.asarray(parameters[name]) for name in names),
            )
        return geometry
