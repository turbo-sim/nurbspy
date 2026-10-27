import operator as _operator

import jax
import jax.numpy as jnp
import equinox as eqx
import optimistix as optx
import quadax as qdx

import matplotlib.pyplot as plt

from .nurbs_basis_functions import (
    compute_basis_polynomials,
    compute_all_basis_polynomials_derivatives,
    _compute_basis_derivatives,
)
from ..graphics import style_3d_axes, close_3d_box
from . import arclength as _arc


def binomial_coeff(n, k):
    """JAX-compatible binomial coefficient C(n, k) using the gamma function."""
    n = jnp.asarray(n, dtype=jnp.float64)
    k = jnp.asarray(k, dtype=jnp.float64)
    return jnp.exp(
        jax.scipy.special.gammaln(n + 1)
        - jax.scipy.special.gammaln(k + 1)
        - jax.scipy.special.gammaln(n - k + 1)
    )


# ----------------------------------------------------------- #
# Standalone functions to compute values
# ----------------------------------------------------------- #
def compute_nurbs_coordinates(P, W, p, U, u):
    """
    Evaluate the coordinates of a NURBS (Non-Uniform Rational B-Spline) curve for a given parameter `u`.

    This function computes the coordinates of the NURBS curve in *homogeneous space* using
    the standard B-spline basis and the control point weights (Equation 4.5 in The NURBS Book),
    and then maps them back to ordinary space via the rational perspective division (Equation 1.16).
    The implementation corresponds to Algorithm A4.1 from The NURBS Book.

    Parameters
    ----------
    P : ndarray (ndim, n+1)
        Array of control point coordinates.
        The first dimension spans spatial coordinates `(x, y, z, ...)`,
        and the second spans the control points along the curve `(0, 1, ..., n)`.

    W : ndarray (n+1,)
        Weights associated with each control point.

    p : int
        Degree of the B-spline basis functions.

    U : ndarray (r+1 = n + p + 2,)
        Knot vector in the u-direction.
        The first and last entries typically have multiplicity `p + 1` for a clamped spline.

    u : scalar or ndarray (N,)
        Parametric coordinate(s) at which to evaluate the curve.

    Returns
    -------
    C : ndarray (ndim, N)
        Coordinates of the evaluated curve points.
        The first dimension spans the spatial coordinates,
        and the second spans the parametric evaluation points `u`.

    Notes
    -----
    - The computation is fully vectorized and uses matrix multiplication for efficiency.
    - The function supports arbitrary spatial dimensions.
    """
    # Highest index of the control points (counting from zero)
    n = P.shape[1] - 1

    # Evaluate the B-spline basis functions N_i,p(u)
    N_basis = compute_basis_polynomials(n, p, U, u)

    # Map control points to homogeneous space: P_w = (x*w, y*w, z*w, w)
    P_w = jnp.concatenate((P * W[None, :], W[None, :]), axis=0)

    # Evaluate the curve in homogeneous space: C_w = Σ_i N_i,p(u) * P_w[i]
    # (implemented efficiently via matrix multiplication)
    C_w = P_w @ N_basis

    # Project back to Euclidean space: (x, y, z) = (x*w, y*w, z*w) / w
    return C_w[:-1, :] / C_w[-1:, :]


def compute_bspline_coordinates(P, p, U, u):
    """
    Evaluate the coordinates of a B-spline curve for a given parameter `u`.

    This function computes the coordinates of a polynomial B-spline curve using
    the standard basis expansion (Equation 3.1 in The NURBS Book).
    The implementation corresponds to Algorithm A3.1.

    Parameters
    ----------
    P : ndarray (ndim, n+1)
        Array of control point coordinates.
        The first dimension spans spatial coordinates `(x, y, z, ...)`,
        and the second spans the control points along the curve `(0, 1, ..., n)`.

    p : int
        Degree of the B-spline basis functions.

    U : ndarray (r+1 = n + p + 2,)
        Knot vector in the u-direction.
        The first and last entries typically have multiplicity `p + 1` for a clamped spline.

    u : scalar or ndarray (N,)
        Parametric coordinate(s) at which to evaluate the curve.

    Returns
    -------
    C : ndarray (ndim, N)
        Coordinates of the evaluated B-spline curve points.
        The first dimension spans the spatial coordinates,
        and the second spans the parametric evaluation points `u`.

    Notes
    -----
    - The computation is vectorized and uses matrix multiplication for efficiency.
    - For rational curves (NURBS), use `compute_nurbs_coordinates` instead.
    """
    # Highest index of the control points (counting from zero)
    n = P.shape[1] - 1

    # Evaluate B-spline basis functions N_i,p(u)
    N_basis = compute_basis_polynomials(n, p, U, u)

    # Evaluate curve coordinates using matrix multiplication
    # C = Σ_i N_i,p(u) * P[i]
    return P @ N_basis


# ----------------------------------------------------------- #
# Standalone functions to compute derivatives
# ----------------------------------------------------------- #
def compute_all_bspline_derivatives(P, p, U, u, up_to_order):
    """
    Compute all analytic derivatives of a polynomial B-spline curve up to a specified order.

    This function evaluates the curve and its parametric derivatives
    using the analytic derivatives of the B-spline basis functions.
    For each derivative order `k`, the derivative of the curve is given by

        C^(k)(u) = Σ_i P_i * d^k N_{i,p}(u) / du^k

    Parameters
    ----------
    P : ndarray (ndim, n+1)
        Control point coordinates.
    p : int
        Degree of the B-spline.
    U : ndarray (n+p+2,)
        Knot vector.
    u : ndarray (Nu,)
        Parametric evaluation points.
    up_to_order : int
        Maximum nonnegative derivative order. Orders above the degree are zero.

    Returns
    -------
    bspline_derivatives : ndarray (up_to_order+1, ndim, Nu)
        Derivatives of the B-spline curve, where
        `bspline_derivatives[k, :, :] = d^k C(u) / du^k`.
    """
    # Ensure inputs are JAX arrays
    P = jnp.asarray(P)
    U = jnp.asarray(U)
    u = jnp.atleast_1d(u)

    n = P.shape[1] - 1
    max_order = min(p, up_to_order)

    # Share the degree recurrence and form only the requested derivative rows.
    all_Nders = _compute_basis_derivatives(n, p, U, u, max_order=max_order)
    bspline_derivatives = jnp.einsum('di,kiu->kdu', P, all_Nders).astype(P.dtype)
    # Preserve the control-point output dtype and zero homogeneous derivatives
    # above p; the rational quotient recurrence below still uses every order.
    return jnp.pad(bspline_derivatives, ((0, up_to_order - max_order), (0, 0), (0, 0)))


def compute_all_nurbs_derivatives(P, W, p, U, u, up_to_order):
    """
    Compute all analytic derivatives of a NURBS curve up to a specified order.

    This function extends the polynomial B-spline derivative computation
    to rational NURBS curves by applying the quotient rule recursively,
    following Algorithm A4.2 from The NURBS Book (Piegl & Tiller, 2nd ed.):

    Parameters
    ----------
    P : ndarray (ndim, n+1)
        Control point coordinates.

    W : ndarray (n+1,)
        Control point weights.

    p : int
        Degree of the NURBS.

    U : ndarray (n+p+2,)
        Knot vector.

    u : ndarray (Nu,)
        Parametric evaluation points.

    up_to_order : int
        Maximum nonnegative derivative order, which may exceed the degree.

    Returns
    -------
    nurbs_derivatives : ndarray (up_to_order+1, ndim, Nu)
        Derivatives of the NURBS curve

    """
    # Ensure inputs are JAX arrays
    P = jnp.asarray(P)
    W = jnp.asarray(W)
    U = jnp.asarray(U)
    u = jnp.asarray(u)

    # Define sizes
    ndim, Nu = P.shape[0], u.size

    # Map control points to homogeneous coordinates: P_w = (x*w, y*w, z*w, w)
    P_w = jnp.concatenate((P * W[None, :], W[None, :]), axis=0)

    # Compute all B-spline derivatives in homogeneous space → (up_to_order+1, ndim+1, Nu)
    bspline_derivatives = compute_all_bspline_derivatives(P_w, p, U, u, up_to_order)

    # Split spatial and weight components
    A_ders = bspline_derivatives[:, :-1, :]  # shape (up_to_order+1, ndim, Nu)
    w_ders = bspline_derivatives[:, -1:, :]  # shape (up_to_order+1, 1, Nu)

    # Allocate NURBS derivatives (up_to_order+1, ndim, Nu)
    nurbs_derivatives = jnp.zeros((up_to_order + 1, ndim, Nu), dtype=P.dtype)

    # Zeroth derivative: C(u) = A0 / w0
    nurbs_derivatives = nurbs_derivatives.at[0].set(A_ders[0] / w_ders[0])

    # Recursive computation for higher-order derivatives (Algorithm A4.2)
    # Using simple Python loops - unrolled at JIT compilation with no runtime overhead
    # Rational derivatives can remain nonzero above the polynomial degree.
    # Only the homogeneous B-spline derivatives vanish beyond that degree.
    for order in range(1, up_to_order + 1):
        # Start with the corresponding derivative of A (homogeneous numerator)
        temp_num = A_ders[order]
        
        # Subtract recursive terms: temp_num -= Σ(i=1 to order) C(order,i) * w_ders[i] * C^(order-i)
        for i in range(1, order + 1):
            coeff = binomial_coeff(order, i)
            temp_num = temp_num - coeff * w_ders[i] * nurbs_derivatives[order - i]
        
        # Divide by zeroth weight derivative to get ordinary-space derivative
        nurbs_derivatives = nurbs_derivatives.at[order].set(temp_num / w_ders[0])
    
    return nurbs_derivatives

    # # Recursive computation for higher-order derivatives (Algorithm A4.2)



# ----------------------------------------------------------- #
# Main NURBS curve class
# ----------------------------------------------------------- #
class NurbsCurve(eqx.Module):
    """Create a NURBS (Non-Uniform Rational Basis Spline) curve object

    Parameters
    ----------
    control_points : ndarray with shape (ndim, n+1)
        Array containing the coordinates of the control points
        The first dimension of `P` spans the coordinates of the control points (any number of dimensions)
        The second dimension of `P` spans the u-direction control points (0,1,...,n)

    weights : ndarray with shape (n+1,)
        Array containing the weight of the control points

    degree : int
        Degree of the basis polynomials

    knots : ndarray with shape (r+1=n+p+2,)
        The knot vector in the u-direction
        Set the multiplicity of the first and last entries equal to `p+1` to obtain a clamped NURBS

    Notes
    -----
    This class includes methods to compute:

        - Curve coordinates for any number of dimensions
        - Analytic curve derivatives of any order and number of dimensions
        - The unitary tangent, normal and binormal vectors (Frenet-Serret reference frame) in 2D and 3D
        - The analytic curvature and torsion in 2D and 3D
        - The arc length of the curve in any number of dimensions.
            The arc length is compute by numerical quadrature using analytic derivative information

    The class can be used to represent polynomial and rational Bézier, B-Spline and NURBS curves
    The type of curve depends on the initialization arguments

        - Polymnomial Bézier: Provide the array of control points
        - Rational Bézier:    Provide the arrays of control points and weights
        - B-Spline:           Provide the array of control points, degree and knot vector
        - NURBS:              Provide the arrays of control points and weights, degree and knot vector

    In addition, this class supports operations with real and complex numbers
    The data type used for the computations is detected from the data type of the arguments
    Using complex numbers can be useful to compute the derivative of the shape using the complex step method


    References
    ----------
    The NURBS Book. See references to equations and algorithms throughout the code
    L. Piegl and W. Tiller
    Springer, second edition

    Curves and Surfaces for CADGD. See references to equations the source code
    G. Farin
    Morgan Kaufmann Publishers, fifth edition

    All references correspond to The NURBS book unless it is explicitly stated that they come from Farin's book

    """

    P: jnp.ndarray  # (ndim, n+1)
    W: jnp.ndarray  # (n+1,)
    p: int
    U: jnp.ndarray  # (r+1,)
    curve_type: str
    ndim: int

    def __init__(self, control_points=None, weights=None, degree=None, knots=None):
        # Void initialization
        if (
            control_points is None
            and weights is None
            and degree is None
            and knots is None
        ):
            self.curve_type = "Void"
            self.ndim = 0
            self.P = jnp.zeros((0, 0))
            self.W = jnp.zeros((0,))
            self.p = 0
            self.U = jnp.zeros((0,))
            # self.U_values = jnp.zeros((0,))
            # self.U_mults = jnp.zeros((0,))
            return

        # Convert inputs to JAX arrays
        P = jnp.asarray(control_points)
        W = None if weights is None else jnp.asarray(weights)
        p = degree
        U = None if knots is None else jnp.asarray(knots)
        n = P.shape[1] - 1

        # Automatic curve type deduction from arguments
        if weights is None and degree is None and knots is None:
            self.curve_type = "Bezier"
            self.ndim = P.shape[0]
            W = jnp.ones((n + 1,), dtype=P.dtype)
            p = n
            U = jnp.concatenate(
                (jnp.zeros(p), jnp.linspace(0.0, 1.0, n - p + 2), jnp.ones(p))
            )

        elif degree is None and knots is None:
            self.curve_type = "R-Bezier"
            self.ndim = P.shape[0]
            p = n
            U = jnp.concatenate(
                (jnp.zeros(p), jnp.linspace(0.0, 1.0, n - p + 2), jnp.ones(p))
            )

        elif weights is None and knots is not None:
            self.curve_type = "B-Spline"
            self.ndim = P.shape[0]
            W = jnp.ones((n + 1,), dtype=P.dtype)

        elif weights is None and knots is None:
            self.curve_type = "B-Spline"
            self.ndim = P.shape[0]
            U = jnp.concatenate(
                (jnp.zeros(p), jnp.linspace(0.0, 1.0, n - p + 2), jnp.ones(p))
            )
            W = jnp.ones((n + 1,), dtype=P.dtype)

        elif weights is not None and knots is None:
            # Clamped NURBS (weights given, knots auto-generated)
            self.curve_type = "NURBS"
            self.ndim = P.shape[0]
            U = jnp.concatenate(
                (jnp.zeros(p), jnp.linspace(0.0, 1.0, n - p + 2), jnp.ones(p))
            )
            W = jnp.asarray(weights, dtype=P.dtype)

        else:
            # General NURBS (weights and knots provided)
            self.curve_type = "NURBS"
            self.ndim = P.shape[0]
            U = jnp.asarray(knots, dtype=P.dtype)
            W = jnp.asarray(weights, dtype=P.dtype)

        # Assign class attributes
        self.P = P
        self.W = W
        self.p = p
        self.U = U

        # Validation (only once at initialization)
        if self.P.ndim != 2:
            raise ValueError("control_points must have shape (ndim, n+1)")
        if self.W.ndim != 1:
            raise ValueError("weights must be 1D (n+1,)")
        if not jnp.isscalar(self.p):
            raise ValueError("degree must be a scalar integer")
        if self.U.ndim != 1:
            raise ValueError("knots must be 1D (r+1,)")
        if self.P.shape[1] != self.W.shape[0]:
            raise ValueError("Mismatch between number of control points and weights")
        if len(self.U) != self.P.shape[1] + self.p + 1:
            raise ValueError(
                f"Knot vector length {len(self.U)} does not match n+p+1={self.P.shape[1]+self.p}"
            )

    # ---------------------------------------------------------------------------------------------------------------- #
    # Define functions to compute NURBS properties
    # ---------------------------------------------------------------------------------------------------------------- #
    @eqx.filter_jit
    def get_value(self, u):
        """Evaluate the coordinates of the curve for the input `u` parametrization

        Parameters
        ----------
        u : scalar or ndarray with shape (N,)
            Parameter used to evaluate the curve

        Returns
        -------
        C : ndarray with shape (ndim, N)
            Array containing the coordinates of the curve
            The first dimension of `C` spans the `(x,y,z)` coordinates
            The second dimension of `C` spans the `u` parametrization sample points

        """
        return compute_nurbs_coordinates(self.P, self.W, self.p, self.U, u)

    @eqx.filter_jit
    def get_derivative(self, u, order):
        """Evaluate the derivative of the curve for the input u-parametrization

        Parameters
        ----------
        u : scalar or ndarray with shape (N,)
            Scalar or array containing the u-parameter used to evaluate the curve

        order : integer
            Order of the derivative

        Returns
        -------
        dC : ndarray with shape (ndim, N)
            Array containing the derivative of the desired order
            The first dimension of `dC` spans the `(x,y,z)` coordinates
            The second dimension of `dC` spans the `u` parametrization sample points

        """

        # Compute the array of curve derivatives up to order `derivative_order` and slice the desired values
        return compute_all_nurbs_derivatives(self.P, self.W, self.p, self.U, u, order)[
            order, ...
        ]

    @eqx.filter_jit
    def get_tangent(self, u):
        """
        Evaluate the unit tangent vector along the curve for the given u-parameterization.

        The tangent is defined as:
            t(u) = C'(u) / ||C'(u)||

        Parameters
        ----------
        u : scalar or ndarray (N,)
            Parametric coordinates.

        Returns
        -------
        tangent : ndarray (ndim, N)
            Unit tangent vectors.
        """
        dC = self.get_derivative(u, 1)
        norm = jnp.linalg.norm(dC, axis=0, keepdims=True)
        return dC / norm

    @eqx.filter_jit
    def get_normal(self, u):
        """
        Evaluate the unit normal vector along the curve for the given u-parameterization.

        For 2D or 3D curves, the normal is defined as:
            - 2D:  n(u) = (-t_y, t_x)
            - 3D:  n(u) = (C'(u) x (C''(u) x C'(u))) / ||C'(u) x (C''(u) x C'(u))||

        Parameters
        ----------
        u : scalar or ndarray (N,)
            Parametric coordinates.

        Returns
        -------
        normal : ndarray (ndim, N)
            Unit normal vectors.
        """

        # Compute curve derivatives up to second order
        ders = compute_all_nurbs_derivatives(
            self.P, self.W, self.p, self.U, u, up_to_order=2
        )
        dC, ddC = ders[1], ders[2]

        # Compute and normalize tangent vector
        t = dC / jnp.linalg.norm(dC, axis=0, keepdims=True)

        # Define inner functions for 2D and 3D normal computation
        # Return padded (3, N) arrays so jax.lax.cond shapes match
        def normal_2d(_):
            # Normal is obtained by a +90° rotation of the tangent vector.
            n = jnp.vstack([-t[1], t[0]])
            n /= jnp.linalg.norm(n, axis=0, keepdims=True)
            # Pad to (3, N) to match 3D branch output shape
            n3 = jnp.pad(n, ((0, 3 - n.shape[0]), (0, 0)))
            return n3

        def normal_3d(_):
            # Embed in 3D for cross product operations
            dC3 = jnp.pad(dC, ((0, 3 - self.ndim), (0, 0)))
            ddC3 = jnp.pad(ddC, ((0, 3 - self.ndim), (0, 0)))

            # Compute n = dC × (ddC × dC)
            n_num = jnp.cross(
                dC3,
                jnp.cross(ddC3, dC3, axisa=0, axisb=0, axisc=0),
                axisa=0, axisb=0, axisc=0,
            )

            # Normalize safely (avoid division by zero)
            n_norm = jnp.linalg.norm(n_num, axis=0, keepdims=True)
            n_norm = jnp.where(n_norm == 0.0, 1.0, n_norm)
            n = n_num / n_norm
            return n  # Already (3, N)
        
        # Select appropriate formulation using jax.lax.cond (JIT-safe conditional)
        n3 = jax.lax.cond(self.ndim == 2, normal_2d, normal_3d, operand=None)

        # Slice back to original curve dimensionality (2D or 3D)
        n = n3[: self.ndim, :]

        return n

    @eqx.filter_jit
    def get_binormal(self, u):
        """
        Evaluate the unit binormal vector along the curve for the given u-parameterization.

        The binormal is defined as:
            b(u) = (C'(u) x C''(u)) / ||C'(u) x C''(u)||

        Parameters
        ----------
        u : scalar or ndarray (N,)
            Parametric coordinates.

        Returns
        -------
        binormal : ndarray (ndim, N)
            Unit binormal vectors.
        """
        # Derivatives
        ders = compute_all_nurbs_derivatives(
            self.P, self.W, self.p, self.U, u, up_to_order=2
        )
        dC, ddC = ders[1], ders[2]

        # Embed to 3D
        dC3 = jnp.pad(dC, ((0, 3 - self.ndim), (0, 0)))
        ddC3 = jnp.pad(ddC, ((0, 3 - self.ndim), (0, 0)))

        # Compute binormal
        b_num = jnp.cross(dC3, ddC3, axisa=0, axisb=0, axisc=0)

        # Normalize safely
        b_norm = jnp.linalg.norm(b_num, axis=0, keepdims=True)
        b_norm = jnp.where(b_norm == 0.0, 1.0, b_norm)
        b3 = b_num / b_norm

        # Slice back to original dimension
        b = b3[: self.ndim, :]
        return b

    @eqx.filter_jit
    def get_curvature(self, u):
        """Evaluate the curvature of the curve for the input u-parametrization

        The definition of the curvature is given by equation 10.7 (Farin's textbook)

        Parameters
        ----------
        u : scalar or ndarray with shape (N,)
            Scalar or array containing the u-parameter used to evaluate the curvature

        Returns
        -------
        curvature : scalar or ndarray with shape (N, )
            Scalar or array containing the curvature of the curve

        """
        # Derivatives
        ders = compute_all_nurbs_derivatives(
            self.P, self.W, self.p, self.U, u, up_to_order=2
        )
        dC, ddC = ders[1], ders[2]

        # Embed to 3D for consistent cross products
        dC3 = jnp.pad(dC, ((0, 3 - self.ndim), (0, 0)))
        ddC3 = jnp.pad(ddC, ((0, 3 - self.ndim), (0, 0)))

        # Cross product and norms
        cross_dd_d = jnp.cross(dC3, ddC3, axisa=0, axisb=0, axisc=0)
        num = jnp.linalg.norm(cross_dd_d, axis=0)
        denom = jnp.linalg.norm(dC3, axis=0) ** 3
        denom = jnp.where(denom == 0.0, 1.0, denom)

        curvature = num / denom
        return curvature

    @eqx.filter_jit
    def get_torsion(self, u):
        """Evaluate the torsion of the curve for the input u-parametrization

        The definition of the torsion is given by equation 10.8 (Farin's textbook)

        Parameters
        ----------
        u : scalar or ndarray with shape (N,)
            Scalar or array containing the u-parameter used to evaluate the torsion

        Returns
        -------
        torsion : scalar or ndarray with shape (N, )
            Scalar or array containing the torsion of the curve

        """
        # Derivatives
        ders = compute_all_nurbs_derivatives(
            self.P, self.W, self.p, self.U, u, up_to_order=3
        )
        dC, ddC, dddC = ders[1], ders[2], ders[3]

        # Embed to 3D for consistent cross products
        dC3 = jnp.pad(dC, ((0, 3 - self.ndim), (0, 0)))
        ddC3 = jnp.pad(ddC, ((0, 3 - self.ndim), (0, 0)))
        dddC3 = jnp.pad(dddC, ((0, 3 - self.ndim), (0, 0)))

        # Cross product and normalization
        cross_d_dd = jnp.cross(dC3, ddC3, axisa=0, axisb=0, axisc=0)
        num = jnp.sum(cross_d_dd * dddC3, axis=0)
        denom = jnp.sum(cross_d_dd**2, axis=0)
        denom = jnp.where(denom == 0.0, 1.0, denom)

        torsion = num / denom
        return torsion


    @eqx.filter_jit
    def get_arclength(self, u1=0.0, u2=1.0, tol=None, *, method="quad_adaptive", **options):
        """Dispatch to an explicit arc-length method (adaptive by default).

        ``method`` is static under ``eqx.filter_jit``: only the selected method
        is traced/compiled, and AD acts on that calculation. The choices are
        ``"quad_adaptive"``, ``"quad_fixed"`` and ``"ode"``.

        ``tol=None`` means 1e-6 for adaptive quadrature and ODE. The optional
        third positional tolerance is retained. Fixed quadrature accepts
        ``n_points`` instead and rejects an explicit tolerance. Other keyword
        options are forwarded to the selected method; unsupported options raise
        TypeError rather than being silently ignored. See each named method for
        its numerical assumptions, diagnostics and differentiation semantics.
        """
        if method == "quad_fixed":
            if tol is not None:
                raise TypeError("quad_fixed uses n_points, not tol")
            return self.get_arclength_quad_fixed(u1, u2, **options)
        if method == "quad_adaptive":
            return self.get_arclength_quad_adaptive(
                u1, u2, 1e-6 if tol is None else tol, **options)
        if method == "ode":
            return self.get_arclength_ode(u1, u2, 1e-6 if tol is None else tol, **options)
        raise ValueError("method must be 'quad_adaptive', 'quad_fixed' or 'ode'")

    @eqx.filter_jit
    def get_arclength_quad_adaptive(self, u1=0.0, u2=1.0, tol=1e-6, *,
                                    atol=None, rtol=None, adjoint="direct",
                                    max_intervals=512, return_info=False):
        """Integrate speed using adaptive knot-aware Gauss-Kronrod quadrature.

        Reversed limits give a signed integral and equal limits give zero.
        Requires real finite geometry, positive weights, in-domain limits and
        no interior curve discontinuities. Zero speed is allowed for length.

        Parameters
        ----------
        u1, u2 : float, optional
            In-domain integration limits, defaulting to 0 and 1.
        tol : float, optional
            Default for each unspecified tolerance (1e-6).
        atol, rtol : float, optional
            Absolute/relative controls: target max(atol, rtol*abs(length)).
        adjoint : {"direct", "leibniz"}, optional
            Direct differentiates the accepted mesh; Leibniz integrates
            sensitivities with their own adaptive solve. A primal error
            estimate does not certify gradient accuracy.
        max_intervals : int, optional
            Static workspace limit, default 512.
        return_info : bool, optional
            Return ArcLengthInfo with value/error/work/status instead of a
            scalar. Nonzero status reports invalid input or solver failure.
            Scalar calls raise on failure, including under JIT.

        Notes
        -----
        Normalization and endpoint grading improve rational-layer resolution,
        but extreme features can evade sampling or exhaust floating-point
        resolution. Error estimates are not certificates. At zero velocity AD
        selects a zero norm subgradient; collapsed length is not differentiable.
        """
        info = _arc._adaptive_length(
            self, u1, u2, atol=tol if atol is None else atol,
            rtol=tol if rtol is None else rtol, adjoint=adjoint,
            max_intervals=max_intervals)
        return info if return_info else _arc._checked_value(info)

    @eqx.filter_jit
    def get_arclength_ode(self, u1=0.0, u2=1.0, tol=1e-6, *, atol=None, rtol=None,
                          max_steps=4096, return_info=False):
        """Integrate speed with knot-aware Dopri5 and local error controls.

        See get_arclength_quad_adaptive for geometry, signed interval and
        diagnostic conventions. Each unspecified atol/rtol defaults to tol.
        Local ODE controls do not guarantee the global length error.
        max_steps is a static workspace limit. ODE error_estimate is NaN and
        evaluations is -1: Diffrax exposes attempted step counts, not a global
        error estimate or exact RHS count. Scalar calls raise on failure.
        """
        info = _arc._ode_length(
            self, u1, u2, atol=tol if atol is None else atol,
            rtol=tol if rtol is None else rtol, max_steps=max_steps)
        return info if return_info else _arc._checked_value(info)

    @eqx.filter_jit
    def get_arclength_quad_fixed(self, u1=0.0, u2=1.0, n_points=41):
        """
        Compute the arc length of the NURBS curve over the parameter interval [u₁, u₂]
        using fixed-point numerical quadrature.

        The arc length of a parametric curve C(u) = [x(u), y(u), z(u)]ᵀ is defined as

            L(u₁, u₂) = ∫‖C'(u)‖ du,

        where C'(u) = dC/du and ‖C'(u)‖ = √(x'(u)² + y'(u)² + z'(u)²) is the local curve speed.

        This function evaluates the integral in Eq. (1) using a Clenshaw-Curtis quadrature rule
        with a prescribed number of nodes `n_points`. The integrand is evaluated at fixed
        quadrature points, making this method computationally cheaper than adaptive ODE
        integration but generally less accurate for rational B-splines.

        Parameters
        ----------
        u1 : float, optional
            Lower bound of the parameter domain (default: 0.0).

        u2 : float, optional
            Upper bound of the parameter domain (default: 1.0).

        n_points : int, optional
            Requested minimum number of quadrature points (default: 41).
            Rounded up to a rule with ``4*k+1`` nodes, with at least 5 nodes,
            as required by Quadax's embedded Clenshaw-Curtis rule.

        Returns
        -------
        L : float
            Arc length of the curve segment C(u) for u ∈ [u₁, u₂].

        """

        # Define the integrand
        def integrand(u, *args):
            dC = self.get_derivative(u, 1)  # (ndim, len(u))
            return jnp.linalg.norm(dC, axis=0)  # ||C'(u)||

        # Perform fixed quadrature over [u1, u2]
        # Only the integral is used. Quadax's error estimate can have undefined
        # gradients for constant speed, so keep that diagnostic out of autodiff.
        error_norm = lambda error: jax.lax.stop_gradient(jnp.max(jnp.abs(error)))
        n_points = _operator.index(n_points)
        if n_points < 1:
            raise ValueError("n_points must be a positive integer")
        # Quadax >=0.3 takes a polynomial order divisible by four, not a
        # node count. Order 40 preserves the existing default 41-node rule.
        # Round other requests upwards so they use at least n_points nodes.
        order = max(4, 4 * ((n_points + 2) // 4))
        rule = qdx.ClenshawCurtisRule(order, norm=error_norm)
        arclength, err, *_ = rule.integrate(integrand, u1, u2, args=())
        return jnp.asarray(arclength).squeeze()

    def reparametrize_by_arclength(self, tol=1e-6, *, max_steps=512,
                                   steps_per_span=16, max_intervals=512):
        """Return a dense ODE map ``u_of_s`` and the total arc length.

        Adaptive quadrature first computes each knot-span length. Independent
        reciprocal ODEs then integrate normalized parameter against normalized
        distance on each span. Dense interpolants give fast queries without
        quadrature or root finding. Exact span anchors prevent cumulative drift
        across knots; ``steps_per_span`` sets a minimum dense-output resolution.

        This method requires a regular curve with positive speed, including
        one-sided endpoint limits. Sampled stationary points, flat spans,
        failed solves and nonmonotone sampled output are rejected explicitly.
        No epsilon is added to hide a reciprocal singularity. Regularity and
        monotonicity sampling are checks, not everywhere-valid proofs.

        ``tol`` is the quadrature and local ODE control, not a global inverse-
        map or unit-speed error guarantee. Tighter controls may be needed for
        sensitivities. ``max_steps`` bounds dense storage per knot span.

        The returned compiled callable accepts scalar/array distances in
        ``[0, total_length]``, preserves shape and supports JAX differentiation
        on regular curves. It is a geometry snapshot: rebuild after changes to
        control points, weights or knots. Keep construction inside a transform
        when differentiating with respect to the geometry.
        """
        mapping = _arc._prepare_map(self, tol=tol, max_steps=max_steps,
                                    steps_per_span=steps_per_span,
                                    max_intervals=max_intervals)
        # Preserve the compiled-function interface, including jax.jit(inverse)
        # and inverse.lower(...). The reusable builder above caches span solves.
        return jax.jit(lambda s: mapping(s)), mapping.total_length


    def reparametrize_by_coordinate(self, dim=0, tol=1e-9):
        """
        Construct a callable function that reparametrizes the NURBS curve by one of its
        coordinate components, returning the corresponding parameter values u(x).

        Let the curve be defined parametrically as

            C(u) = [C₀(u), C₁(u), ..., Cₙ₋₁(u)]ᵀ,   u ∈ [0, 1].

        This function reparametrizes the curve using the coordinate C_dim(u) as the
        independent variable. For a given coordinate value x*, the corresponding
        parameter u* is obtained by solving

            R(u) = C_dim(u) - x* = 0.

        This equation is solved using a bounded Newton iteration implemented through
        `optimistix.root_find`, ensuring u* ∈ [0, 1]. The process is vectorized and
        JIT-compiled under JAX for efficient batch evaluation.

        The resulting callable maps coordinate values x → u(x), providing the inverse
        parameterization where one coordinate (e.g. x) serves as the independent variable.
        The actual curve points can subsequently be obtained as C(u(x)) if desired.

        The mapping is well-defined only if C_dim(u) is monotonic over [0, 1].

        Parameters
        ----------
        dim : int, optional
            Index of the coordinate used as independent variable. Default is 0
            (the first coordinate component, typically x).
        tol : float, optional
            Relative and absolute tolerance for the Newton solver (default: 1e-9).

        Returns
        -------
        callable
            A JAX-compiled, vectorized function handle u(x) that returns the parameter
            values corresponding to the specified coordinate values x, i.e.

                u(x) = C_i⁻¹(x),

            where C_i⁻¹ denotes the inverse mapping of the coordinate function C_i(u) over the domain [0, 1].
        """
        func_handle = lambda xq: self._get_u_for_coordinate(xq, dim=dim, tol=tol)
        return jax.jit(jax.vmap(func_handle))

    def _get_u_for_coordinate(self, x_query, dim=0, max_iters=64, tol=1e-9):
        # Residual function: difference between coordinate component and target
        def residual(u, args):
            C = self.get_value(u).squeeze()
            return C[dim] - x_query

        # Heuristic initial guess from normalized coordinate
        # (works well if the curve is monotonic in this dimension)
        u0 = jnp.array([0.5])

        # Configure Newton solver
        solver = optx.Newton(rtol=tol, atol=tol)

        # Run bounded Newton solve for u ∈ [0, 1]
        result = optx.root_find(
            residual,
            solver=solver,
            y0=u0,
            options={"lower": 0.0, "upper": 1.0},
            throw=False,
            max_steps=max_iters,
        )

        # Return scalar result
        return result.value.squeeze()


    # ---------------------------------------------------------------------------------------------------------------- #
    # Define functions to solve point projection problem
    # ---------------------------------------------------------------------------------------------------------------- #
    @eqx.filter_jit
    def project_points(self, Q_all, tol=1e-9, *, maxiter=100, samples_per_span=64,
                       gtol=None, xtol=1e-12, ftol=1e-14):
        """Project query columns using sampled initialization and one local solve.

        The knot-aware grid is evaluated once for the batch. Its nearest point
        starts a bounded Newton/line-search minimization on the actual knot
        domain. ``samples_per_span`` is a static interior sampling budget;
        strongly unequal positive weights add logarithmic endpoint coverage
        within that budget. Finite sampling cannot guarantee a global minimum.

        ``tol`` now controls tangent-scaled spatial stationarity, with threshold
        ``tol * (1 + distance)``. ``gtol`` overrides it when supplied. The default
        1e-9 preserves the old effective internal tolerance; the old public 1e-6
        argument was ignored. ``xtol`` and ``ftol`` detect stagnation, which alone
        is never accepted as convergence. Nonconverged solves raise an error.
        Static options must be finite/nonnegative; the sampling budget is positive.

        Inputs have shape (ndim,N); returned parameters have shape (N,).
        Geometry must be finite, with positive weights and continuous knots
        (interior multiplicity at most degree). Empty query batches are allowed.

        Forward and reverse AD use the exact unregularized stationarity Hessian
        at a locally unique regular minimum with stable active bounds. Invalid
        sensitivities at collapsed geometry, weak bounds and nonsmooth knots
        are NaN. Local checks cannot certify global uniqueness. For the minimized
        squared-distance loss only, freezing the solved parameters also gives
        the envelope gradient; this does not apply to arbitrary downstream losses.
        """
        from ._curve_projection import project_curve_points
        from ._point_projection import validate_options

        if self.P.ndim != 2 or self.P.size == 0:
            raise ValueError("A nonempty NurbsCurve is required")
        points = jnp.asarray(Q_all)
        if points.ndim != 2 or points.shape[0] != self.ndim:
            raise ValueError("Q_all must have shape (ndim,N)")
        validate_options(maxiter, samples_per_span, ftol, tol, xtol)
        gtol = tol if gtol is None else gtol
        validate_options(maxiter, samples_per_span, ftol, gtol, xtol)
        return project_curve_points(self.P, self.W, self.U, points, self.p,
                                    maxiter, ftol, gtol, xtol, samples_per_span)

    def _project_point_scalar(self, Q, max_iters=100, tol=1e-9):
        """Compatibility helper, using the same sampled single-start solver."""
        point = jnp.asarray(Q).reshape(-1, 1)
        return self.project_points(point, tol=tol, maxiter=max_iters)[0]


    # ---------------------------------------------------------------------------------------------------------------- #
    # Define functions for plotting
    # ---------------------------------------------------------------------------------------------------------------- #
    def plot(
        self,
        fig=None,
        ax=None,
        curve=True,
        control_points=True,
        frenet_serret=False,
        axis_off=False,
        ticks_off=False,
        rescale=True,
    ):
        """Create a plot and return the figure and axes handles"""

        if fig is None:

            # One dimension (law of evolution)
            if self.ndim == 1:
                fig = plt.figure(figsize=(6, 5))
                ax = fig.add_subplot(111)
                ax.set_xlabel("$u$ parameter", fontsize=12, color="k", labelpad=12)
                ax.set_ylabel("NURBS curve value", fontsize=12, color="k", labelpad=12)
                # ax_xy.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.1f'))
                # ax_xy.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.1f'))
                # for t in ax.xaxis.get_major_ticks(): t.label.set_fontsize(12)
                # for t in ax.yaxis.get_major_ticks(): t.label.set_fontsize(12)
                if ticks_off:
                    ax.set_xticks([])
                    ax.set_yticks([])
                if axis_off:
                    ax.axis("off")

            # Two dimensions (plane curve)
            elif self.ndim == 2:
                fig = plt.figure(figsize=(6, 5))
                ax = fig.add_subplot(111)
                ax.set_xlabel("$x$ axis", fontsize=12, color="k", labelpad=12)
                ax.set_ylabel("$y$ axis", fontsize=12, color="k", labelpad=12)
                # ax_xy.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.1f'))
                # ax_xy.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.1f'))
                # for t in ax.xaxis.get_major_ticks(): t.label.set_fontsize(12)
                # for t in ax.yaxis.get_major_ticks(): t.label.set_fontsize(12)
                if ticks_off:
                    ax.set_xticks([])
                    ax.set_yticks([])
                if axis_off:
                    ax.axis("off")

            # Three dimensions (space curve)
            elif self.ndim == 3:
                fig = plt.figure(figsize=(6, 5))
                ax = fig.add_subplot(111, projection="3d")
                style_3d_axes(ax, elev=30, azim=-120, pane_alpha=1.0)
                ax.set_xlabel("$x$ axis", fontsize=12, color="k", labelpad=12)
                ax.set_ylabel("$y$ axis", fontsize=12, color="k", labelpad=12)
                ax.set_zlabel("$z$ axis", fontsize=12, color="k", labelpad=12)
                # ax_xy.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.1f'))
                # ax_xy.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.1f'))
                # ax_xy.zaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.1f'))
                # for t in ax.xaxis.get_major_ticks(): t.label.set_fontsize(8)
                # for t in ax.yaxis.get_major_ticks(): t.label.set_fontsize(8)
                # for t in ax.zaxis.get_major_ticks(): t.label.set_fontsize(8)
                if ticks_off:
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.set_zticks([])
                if axis_off:
                    ax.axis("off")

            else:
                raise Exception("The number of dimensions must be 1, 2 or 3")

        # Add objects to the plot
        if curve:
            self.plot_curve(fig, ax)
        if control_points:
            self.plot_control_points(fig, ax)
        if frenet_serret:
            self.plot_frenet_serret(fig, ax)

        # Set the scaling of the axes
        if rescale:
            self.rescale_plot(fig, ax)

        return fig, ax

    def plot_curve(
        self, fig, ax, linewidth=1.5, linestyle="-", color="black", u1=0.00, u2=1.00
    ):
        """Plot the coordinates of the NURBS curve"""

        # One dimension (law of evolution)
        if self.ndim == 1:
            u = jnp.linspace(u1, u2, 501)
            X = self.get_value(u)
            (line,) = ax.plot(u, X[0, :])
            line.set_linewidth(linewidth)
            line.set_linestyle(linestyle)
            line.set_color(color)
            line.set_marker(" ")

        # Two dimensions (plane curve)
        elif self.ndim == 2:
            u = jnp.linspace(u1, u2, 501)
            X, Y = self.get_value(u)
            (line,) = ax.plot(X, Y)
            line.set_linewidth(linewidth)
            line.set_linestyle(linestyle)
            line.set_color(color)
            line.set_marker(" ")

        # Three dimensions (space curve)
        elif self.ndim == 3:
            u = jnp.linspace(u1, u2, 501)
            X, Y, Z = self.get_value(u)
            (line,) = ax.plot(X, Y, Z)
            line.set_linewidth(linewidth)
            line.set_linestyle(linestyle)
            line.set_color(color)
            line.set_marker(" ")

        else:
            raise Exception("The number of dimensions must be 1, 2 or 3")

        return fig, ax

    def plot_control_points(
        self,
        fig,
        ax,
        linewidth=1.25,
        linestyle="-.",
        color="red",
        markersize=5,
        markerstyle="o",
    ):
        """Plot the control points of the NURBS curve"""

        # One dimension (law of evolution)
        if self.ndim == 1:
            Px = self.P
            u = jnp.linspace(0.0, 1.0, Px.size)
            (line,) = ax.plot(u, Px[0, :])
            line.set_linewidth(linewidth)
            line.set_linestyle(linestyle)
            line.set_color(color)
            line.set_marker(markerstyle)
            line.set_markersize(markersize)
            line.set_markeredgewidth(linewidth)
            line.set_markeredgecolor(color)
            line.set_markerfacecolor("w")
            line.set_zorder(4)
            # line.set_label(' ')

        # Two dimensions (plane curve)
        elif self.ndim == 2:
            Px, Py = self.P
            (line,) = ax.plot(Px, Py)
            line.set_linewidth(linewidth)
            line.set_linestyle(linestyle)
            line.set_color(color)
            line.set_marker(markerstyle)
            line.set_markersize(markersize)
            line.set_markeredgewidth(linewidth)
            line.set_markeredgecolor(color)
            line.set_markerfacecolor("w")
            line.set_zorder(4)
            # line.set_label(' ')

        # Three dimensions (space curve)
        elif self.ndim == 3:
            Px, Py, Pz = self.P
            (line,) = ax.plot(Px, Py, Pz)
            line.set_linewidth(linewidth)
            line.set_linestyle(linestyle)
            line.set_color(color)
            line.set_marker(markerstyle)
            line.set_markersize(markersize)
            line.set_markeredgewidth(linewidth)
            line.set_markeredgecolor(color)
            line.set_markerfacecolor("w")
            line.set_zorder(4)
            # line.set_label(' ')

        else:
            raise Exception("The number of dimensions must be 2 or 3")

        return fig, ax

    def plot_frenet_serret(self, fig, ax, frame_number=5, frame_scale=0.10):
        """Plot some Frenet-Serret reference frames along the NURBS curve"""

        # Compute the tangent, normal and binormal unitary vectors
        u = jnp.linspace(0.0, 1.0, frame_number)
        position = self.get_value(u)
        tangent = self.get_tangent(u)
        normal = self.get_normal(u)
        binormal = self.get_binormal(u)

        # Compute a length scale (fraction of the curve arc length)
        scale = float(frame_scale * self.get_arclength(0.0, 1.0))

        # Two dimensions (plane curve)
        if self.ndim == 2:

            # Plot the frames of reference
            for k in range(frame_number):

                # Plot the tangent vector
                x, y = position[:, k]
                u, v = tangent[:, k]
                ax.quiver(x, y, u, v, color="red", scale=7.5)

                # Plot the normal vector
                x, y = position[:, k]
                u, v = normal[:, k]
                ax.quiver(x, y, u, v, color="blue", scale=7.5)

            # Plot the origin of the vectors
            x, y = position
            (points,) = ax.plot(x, y)
            points.set_linestyle(" ")
            points.set_marker("o")
            points.set_markersize(5)
            points.set_markeredgewidth(1.25)
            points.set_markeredgecolor("k")
            points.set_markerfacecolor("w")
            points.set_zorder(4)
            # points.set_label(' ')

        # Three dimensions (space curve)
        elif self.ndim == 3:

            # Plot the frames of reference
            for k in range(frame_number):

                # Plot the tangent vector
                x, y, z = position[:, k]
                u, v, w = tangent[:, k]
                ax.quiver(x, y, z, u, v, w, color="red", length=scale, normalize=True)

                # Plot the norma vector
                x, y, z = position[:, k]
                u, v, w = normal[:, k]
                ax.quiver(x, y, z, u, v, w, color="blue", length=scale, normalize=True)

                # Plot the binormal vector
                x, y, z = position[:, k]
                u, v, w = binormal[:, k]
                ax.quiver(x, y, z, u, v, w, color="green", length=scale, normalize=True)

            # Plot the origin of the vectors
            x, y, z = position
            (points,) = ax.plot(x, y, z)
            points.set_linestyle(" ")
            points.set_marker("o")
            points.set_markersize(5)
            points.set_markeredgewidth(1.25)
            points.set_markeredgecolor("k")
            points.set_markerfacecolor("w")
            points.set_zorder(4)
            # points.set_label(' ')

        else:
            raise Exception("The number of dimensions must be 2 or 3")

        return fig, ax

    def rescale_plot(self, fig, ax):
        """Adjust the aspect ratio of the figure"""

        # Two dimensions (plane curve)
        if self.ndim == 2:

            # Set the aspect ratio of the data
            ax.set_aspect(1.0)

            # Adjust pad
            plt.tight_layout(pad=1.0)

        # Three dimensions (space curve)
        if self.ndim == 3:

            # Set axes aspect ratio
            ax.autoscale(enable=True)
            x_min, x_max = ax.get_xlim()
            y_min, y_max = ax.get_ylim()
            z_min, z_max = ax.get_zlim()
            x_mid = (x_min + x_max) / 2
            y_mid = (y_min + y_max) / 2
            z_mid = (z_min + z_max) / 2
            # L = jnp.max((x_max - x_min, y_max - y_min, z_max - z_min)) / 2
            L = jnp.max(jnp.array([x_max - x_min, y_max - y_min, z_max - z_min])) / 2

            ax.set_xlim3d(x_mid - 1.0 * L, x_mid + 1.0 * L)
            ax.set_ylim3d(y_mid - 1.0 * L, y_mid + 1.0 * L)
            ax.set_zlim3d(z_mid - 1.0 * L, z_mid + 1.0 * L)

            # Draw the full box outline now that the limits are final
            close_3d_box(ax)

            # Adjust pad
            plt.tight_layout(pad=1.0)

    def plot_curvature(self, fig=None, ax=None, color="black", linestyle="-"):

        # Create the figure
        if fig is None:
            fig = plt.figure(figsize=(6, 5))
            ax = fig.add_subplot(111)
        ax.set_xlabel("$u$ parameter", fontsize=12, color="k", labelpad=12)
        ax.set_ylabel("Curvature", fontsize=12, color="k", labelpad=12)
        # ax_xy.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.1f'))
        # ax_xy.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.1f'))
        for t in ax.xaxis.get_major_ticks():
            t.label.set_fontsize(12)
        for t in ax.yaxis.get_major_ticks():
            t.label.set_fontsize(12)
        # ax_xy.set_xticks([])
        # ax_xy.set_yticks([])
        # ax_xy.axis('off')

        # Plot the curvature distribution
        u = jnp.linspace(0, 1, 501)
        curvature = self.get_curvature(u)
        (line,) = ax.plot(u, curvature)
        line.set_linewidth(1.25)
        line.set_linestyle(linestyle)
        line.set_color(color)
        line.set_marker(" ")
        line.set_markersize(3.5)
        line.set_markeredgewidth(1)
        line.set_markeredgecolor("k")
        line.set_markerfacecolor("w")

        # Adjust pad
        plt.tight_layout(pad=5.0, w_pad=None, h_pad=None)

        return fig, ax

    def plot_torsion(self, fig=None, ax=None, color="black", linestyle="-"):

        # Create the figure
        if fig is None:
            fig = plt.figure(figsize=(6, 5))
            ax = fig.add_subplot(111)
        ax.set_xlabel("$u$ parameter", fontsize=12, color="k", labelpad=12)
        ax.set_ylabel("Torsion", fontsize=12, color="k", labelpad=12)
        # ax_xy.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.1f'))
        # ax_xy.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.1f'))
        for t in ax.xaxis.get_major_ticks():
            t.label.set_fontsize(12)
        for t in ax.yaxis.get_major_ticks():
            t.label.set_fontsize(12)
        # ax_xy.set_xticks([])
        # ax_xy.set_yticks([])
        # ax_xy.axis('off')

        # Plot the curvature distribution
        u = jnp.linspace(0, 1, 501)
        torsion = self.get_torsion(u)
        (line,) = ax.plot(u, torsion)
        line.set_linewidth(1.25)
        line.set_linestyle(linestyle)
        line.set_color(color)
        line.set_marker(" ")
        line.set_markersize(3.5)
        line.set_markeredgewidth(1)
        line.set_markeredgecolor("k")
        line.set_markerfacecolor("w")

        # Set the aspect ratio of the figure
        ratio = 1.00
        x1, x2 = ax.get_xlim()
        y1, y2 = ax.get_ylim()
        ax.set_aspect(jnp.abs((x2 - x1) / (y2 - y1)) * ratio)

        # Adjust pad
        plt.tight_layout(pad=5.0, w_pad=None, h_pad=None)

        return fig, ax


# ---------------------------------------------------------------------------------------------------------------- #
# Miscellaneous functions
# ---------------------------------------------------------------------------------------------------------------- #
def merge_nurbs_curves(nurbs_list):
    """
    Merge multiple NURBS curves into a single continuous curve with C⁰ continuity.

    Parameters
    ----------
    nurbs_list : list of NurbsCurve
        List of NURBS curve instances to merge. All must have the same polynomial degree.

    Returns
    -------
    merged : NurbsCurve
        A new NURBS curve representing the concatenation of all input curves.

    Notes
    -----
    * Each curve is mapped to a subinterval of [0, 1] with equal length.
    * Adjacent curves are connected with multiplicity p + 1 (C⁰ continuity).
    * A small epsilon offset is applied at internal joins to avoid Gmsh issues
      related to repeated knots with multiplicity exactly equal to degree.
    """
    if len(nurbs_list) < 2:
        raise ValueError("Need at least two NURBS curves to merge.")

    # Check that all curves share the same degree
    p = nurbs_list[0].p
    for c in nurbs_list[1:]:
        if c.p != p:
            raise ValueError("All NURBS curves must have the same degree.")

    # Merge control points and weights
    P = jnp.concatenate([c.P for c in nurbs_list], axis=1)
    W = jnp.concatenate([c.W for c in nurbs_list], axis=0)

    # Knot vector construction
    eps = 1e-8
    n_segments = len(nurbs_list)
    delta = 1.0 / n_segments

    U_start = jnp.zeros((p + 1,))
    U_end = jnp.ones((p + 1,))
    U_list = [U_start]

    # Build knot pieces for each curve and junction
    for i, c in enumerate(nurbs_list):
        n = c.P.shape[1] - 1
        U_local = c.U[p + 1 : n + 1]  # internal knots
        U_scaled = i * delta + U_local * delta
        U_list.append(U_scaled)

        # Add midpoint knot between curves (except last)
        if i < n_segments - 1:
            u_mid = jnp.full((p + 1,), (i + 1) * delta)
            u_mid = u_mid.at[0].set((i + 1) * delta - eps)
            u_mid = u_mid.at[-1].set((i + 1) * delta + eps)
            U_list.append(u_mid)

    U_list.append(U_end)

    # Concatenate all parts
    U = jnp.concatenate(U_list)

    # Return merged curve
    return nurbs_list[0].__class__(
        control_points=P,
        weights=W,
        degree=p,
        knots=U,
    )
