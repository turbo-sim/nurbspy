"""Small coordinate values for readable curve and surface construction."""

from dataclasses import dataclass
from numbers import Number

import numpy as np

__all__ = ["Point"]
_MISSING = object()


@dataclass(frozen=True, init=False)
class Point:
    """An immutable 2D or 3D coordinate value, independent of any geometry.

    Use ``Point(x, y)`` or ``Point(x, y, z)``. Coordinates can be real, complex,
    or scalar backend values. Weights belong to curves/surfaces, not points.
    Construction copies these values into the selected backend's control array;
    reusing a Point does not create a live link between geometries.
    """

    _coordinates: tuple

    def __init__(self, x, y, z=_MISSING):
        coordinates = (x, y) if z is _MISSING else (x, y, z)
        values = []
        for value in coordinates:
            numeric_scalar = isinstance(value, Number)
            if hasattr(value, "dtype"):
                numeric_scalar = (getattr(value, "ndim", -1) == 0
                                  and np.issubdtype(value.dtype, np.number))
            if not numeric_scalar:
                raise TypeError("Point coordinates must be numeric scalars")
            # Detach mutable zero-dimensional NumPy arrays, without converting
            # JAX scalar arrays/tracers to host values.
            values.append(value[()] if isinstance(value, np.ndarray) else value)
        object.__setattr__(self, "_coordinates", tuple(values))

    @property
    def x(self):
        return self._coordinates[0]

    @property
    def y(self):
        return self._coordinates[1]

    @property
    def z(self):
        if len(self) != 3:
            raise AttributeError("A 2D Point has no z coordinate")
        return self._coordinates[2]

    def __len__(self):
        return len(self._coordinates)

    def __iter__(self):
        return iter(self._coordinates)

    def __getitem__(self, index):
        return self._coordinates[index]

    def __repr__(self):
        return "Point(" + ", ".join(repr(value) for value in self) + ")"
