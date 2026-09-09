# Basis polynomials

## Inputs

This example evaluates five cubic B-spline basis functions, together with
their first and second derivatives. `n=4` is the **highest index**, so there
are `n+1=5` functions; `p=3` is their degree. The knot vector has
`n+p+2=9` entries:

```text
[0, 0, 0, 0, 0.5, 1, 1, 1, 1]
```

Four repeated knots at each end clamp the basis. The single interior knot
at $u=0.5$ gives two spans with $C^2$ continuity. The equations are developed
in [B-spline theory](../theory/bspline.md).

## Complete script

Run `python demos/documentation/basis_polynomials.py`, or
{download}`download the script <../../../demos/documentation/basis_polynomials.py>`.

```{literalinclude} ../../../demos/documentation/basis_polynomials.py
:language: python
```

## Output

`N`, `dN`, and `ddN` all have shape `(5, 501)`. Each row holds one
basis function over the parameter samples. At the middle sample:

```text
Basis array shape: (5, 501)
Basis at u=0.5: [0.   0.25 0.5  0.25 0.  ]
```

The basis values sum to one. Their first and second derivatives sum to
zero, up to floating-point roundoff, because they differentiate that
constant sum.

```{figure} images/basis_polynomials.png
:alt: Five cubic B-spline basis functions and their first two derivatives.

Values, slopes, and second derivatives across the two knot spans.
```

The first and last bases interpolate the ends. At most four cubic bases
are nonzero inside a span. To obtain Bernstein polynomials, set `p=n`
and use `n+1` zeros followed by `n+1` ones.
