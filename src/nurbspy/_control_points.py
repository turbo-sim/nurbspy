"""Normalize convenience input once, without guessing numerical array layout."""

from .point import Point


def _contains_point(value):
    if isinstance(value, Point):
        return True
    return isinstance(value, (list, tuple)) and any(_contains_point(v) for v in value)


def _stack_point(point, xp):
    return xp.stack([xp.asarray(coordinate) for coordinate in point])


def curve_control_points(values, xp):
    """Return (dimension, count), preserving legacy numerical array ordering."""
    if _contains_point(values):
        if (not isinstance(values, (list, tuple)) or not values
                or not all(isinstance(point, Point) for point in values)):
            raise ValueError("control_points must be a sequence of Point objects, without mixed entries")
        if len({len(point) for point in values}) != 1:
            raise ValueError("All control Points must have the same dimension")
        result = xp.stack([_stack_point(point, xp) for point in values], axis=1)
    else:
        result = xp.asarray(values)
    if result.ndim != 2 or min(result.shape) < 1:
        raise ValueError("control_points must have shape (dimension, count) or be a sequence of Points")
    if result.dtype.kind not in "biufc":
        raise TypeError("control_points must contain numeric coordinates")
    return result
