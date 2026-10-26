`legacy_geometry.pickle` contains a quadratic planar curve and a bilinear
unit-square surface serialized with pickle protocol 4 before moving the
NumPy implementation into `nurbspy.numpy` (parent commit `cd2600c`). It is
fixed regression data: do not regenerate it with the reorganized package.

The curve control points are `(0, 0)`, `(1, 1)`, `(2, 0)`. The surface has
unit weights and maps `(u, v)` to `(u, v, 0)`. Tests compare their values and
derivatives against analytical references after deserialization.
