import jax
import jax.numpy as jnp
import equinox as eqx
import optimistix as optx
import quadax

import matplotlib.pyplot as plt

from .nurbs_basis_functions import (
    compute_basis_polynomials,
    compute_all_basis_polynomials_derivatives,
)


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
        Maximum derivative order to compute (0 ≤ order ≤ p).

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
    ndim, Nu = P.shape[0], u.size
    max_order = min(p, up_to_order)

    # Get all basis derivatives up to order p → (p+1, n+1, Nu)
    all_Nders = compute_all_basis_polynomials_derivatives(n, p, U, u)

    # Allocate curve derivatives (up_to_order+1, ndim, Nu)
    bspline_derivatives = jnp.zeros((up_to_order + 1, ndim, Nu), dtype=P.dtype)

    # Compute derivatives by matrix multiplication for all relevant orders
    for k in range(p + 1):
        bspline_derivatives = bspline_derivatives.at[k].set(P @ all_Nders[k])

    return bspline_derivatives


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
        Maximum derivative order to compute (0 ≤ order ≤ p).

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
    max_order = min(p, up_to_order)

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
    def outer_body(order, nurbs_derivatives):
        # Start with the corresponding derivative of A (homogeneous numerator)
        temp_num = A_ders[order]

        # Subtract recursive terms involving lower derivatives
        def inner_body(i, temp_num):
            coeff = binomial_coeff(order, i)
            return temp_num - coeff * w_ders[i] * nurbs_derivatives[order - i]

        temp_num = jax.lax.fori_loop(1, order + 1, inner_body, temp_num)

        # Divide by zeroth weight derivative to get ordinary-space derivative
        nurbs_derivatives = nurbs_derivatives.at[order].set(temp_num / w_ders[0])
        return nurbs_derivatives

    nurbs_derivatives = jax.lax.fori_loop(
        1, max_order + 1, outer_body, nurbs_derivatives
    )
    return nurbs_derivatives


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
    U_values: jnp.ndarray
    U_mults: jnp.ndarray

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
            self.U_values = jnp.zeros((0,))
            self.U_mults = jnp.zeros((0,))
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
        self.U_values, self.U_mults = jnp.unique(U, return_counts=True)

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
            n(u) = (C'(u) x (C''(u) x C'(u))) / ||C'(u) x (C''(u) x C'(u))||

        Parameters
        ----------
        u : scalar or ndarray (N,)
            Parametric coordinates.

        Returns
        -------
        normal : ndarray (ndim, N)
            Unit normal vectors.
        """

        # Derivatives
        ders = compute_all_nurbs_derivatives(
            self.P, self.W, self.p, self.U, u, up_to_order=2
        )
        dC, ddC = ders[1], ders[2]

        # Embed to 3D for consistent cross product operations
        dC3 = jnp.pad(dC, ((0, 3 - self.ndim), (0, 0)))
        ddC3 = jnp.pad(ddC, ((0, 3 - self.ndim), (0, 0)))

        # Compute normaø, n = dC × (ddC × dC)
        n_num = jnp.cross(
            dC3,
            jnp.cross(ddC3, dC3, axisa=0, axisb=0, axisc=0),
            axisa=0,
            axisb=0,
            axisc=0,
        )

        # Normalize safely (avoid division by zero)
        n_norm = jnp.linalg.norm(n_num, axis=0, keepdims=True)
        n_norm = jnp.where(n_norm == 0.0, 1.0, n_norm)
        n3 = n_num / n_norm

        # Slice back to original dimension
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
    def get_arclength(self, u1=0.0, u2=1.0, n_points=41):
        """Compute the arc length of a parametric curve in the interval [u1,u2] using numerical quadrature

        The definition of the arc length is given by equation 10.3 (Farin's textbook)

        Parameters
        ----------
        u1 : scalar
            Lower limit of integration for the arc length computation

        u2 : scalar
            Upper limit of integration for the arc length computation

        Returns
        -------
        L : scalar
            Arc length of NURBS curve in the interval [u1, u2]

        """

        # Define the integrand
        def integrand(u, *args):
            dC = self.get_derivative(u, 1)  # (ndim, len(u))
            return jnp.linalg.norm(dC, axis=0)  # ||C'(u)||

        # Perform fixed quadrature over [u1, u2]
        rule = quadax.ClenshawCurtisRule(n_points)
        arclength, err, *_ = rule.integrate(integrand, u1, u2, args=())
        return jnp.asarray(arclength).squeeze()

    # ---------------------------------------------------------------------------------------------------------------- #
    # Define functions to solve point projection problem
    # ---------------------------------------------------------------------------------------------------------------- #
    @eqx.filter_jit
    def project_points_to_curve(self, Q_all):
        """
        Vectorized projection of multiple points onto the NURBS curve.

        Parameters
        ----------
        Q_all : array_like, shape (ndim, n_points)
            Points to project onto the curve.

        Returns
        -------
        u_all : ndarray, shape (n_points,)
            Parameter values of the orthogonal projections.
        """
        return jax.vmap(self.project_point_to_curve, in_axes=1)(Q_all)

    @eqx.filter_jit
    def project_point_to_curve(self, Q, max_iters=32):
        """
        Project a point onto the NURBS curve by solving the orthogonality condition.

        The projection point `C(u*)` minimizes the squared Euclidean distance to `Q`:

            f(u) = ||C(u) - Q||²

        The stationary condition is obtained from:

            f'(u) = 2 (C(u) - Q) · C'(u) = 0

        which ensures that the vector from the curve to the point is orthogonal to the tangent
        of the curve at the projection point.

        The nonlinear equation is solved with a bounded Newton method (`optimistix.Newton`),
        ensuring that `u_star` remains within [0, 1].

        The initial guess is determined heuristically from the control polygon
        (see `_initial_guess_from_polygon`).

        Parameters
        ----------
        Q : array_like, shape (ndim,)
            Coordinates of the point to be projected onto the curve.
        max_iters : int, optional
            Maximum number of Newton iterations used by the solver. Default is 32.

        Returns
        -------
        u_star : float
            Parameter value in [0, 1] corresponding to the orthogonal projection of `Q` onto the curve.
        """
        # Ensure shape (ndim,)
        Q = jnp.asarray(Q).squeeze()

        # Function whose root defines orthogonality condition
        def residual(u, args):
            C = self.get_value(u).squeeze()
            dC = self.get_derivative(u, 1).squeeze()
            r = jnp.sum((C - Q) * dC)
            return r

        # Heuristic initial guess from control polygon
        u0 = jnp.array([self._projection_initial_guess(Q)])

        # Run bounded Newton solver
        solver = optx.Newton(rtol=1e-6, atol=1e-8)
        result = optx.root_find(
            residual,
            solver=solver,
            y0=u0,
            options={"lower": 0.0, "upper": 1.0},
            throw=False,
            max_steps=max_iters,
        )

        # Return scalar value of u
        u_star = result.value.squeeze()

        return u_star

    # def _projection_initial_guess(self, Q):
    #     """
    #     Estimate a good initial guess for u by locating the control point closest to Q
    #     and mapping its index proportionally to the active knot span [U[p], U[-p-1]].
    #     This provides a fast and robust starting value for the projection solver.
    #     """
    #     P, U, p = self.P, self.U, self.p
    #     diff = P - Q[:, None]
    #     dist2 = jnp.sum(diff * diff, axis=0)
    #     j = jnp.argmin(dist2)
    #     n = P.shape[1] - 1
    #     u_min, u_max = U[p], U[-p - 1]  # interior knot range
    #     u0 = u_min + (u_max - u_min) * (j / n)
    #     return jnp.clip(u0, 0.0, 1.0)

    def _projection_initial_guess(self, Q, n_samples=101):
        """
        Estimate a good initial guess for the projection parameter `u`
        using a greedy sampling strategy.

        The curve is sampled at `n_samples` uniformly spaced parameter values
        within [U[0], U[-1]], and the point `C(u)` closest to the query
        point `Q` is selected as the initial guess.

        This provides a robust starting point for the Newton-based projection,
        especially for highly curved or nonuniform NURBS.
        """
        U = self.U
        u_min, u_max = U[0], U[-1]
        u_candidates = jnp.linspace(u_min, u_max, n_samples)
        C_candidates = self.get_value(u_candidates)  # (ndim, n_samples)
        dist2 = jnp.sum((C_candidates - Q[:, None]) ** 2, axis=0)
        i_best = jnp.argmin(dist2)
        return u_candidates[i_best]

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
                ax.view_init(azim=-120, elev=30)
                ax.grid(False)
                ax.xaxis.pane.fill = False
                ax.yaxis.pane.fill = False
                ax.zaxis.pane.fill = False
                ax.xaxis.pane.set_edgecolor("k")
                ax.yaxis.pane.set_edgecolor("k")
                ax.zaxis.pane.set_edgecolor("k")
                ax.xaxis.pane._alpha = 1.0
                ax.yaxis.pane._alpha = 1.0
                ax.zaxis.pane._alpha = 1.0
                ax.set_xlabel("$x$ axis", fontsize=12, color="k", labelpad=12)
                ax.set_ylabel("$y$ axis", fontsize=12, color="k", labelpad=12)
                ax.set_zlabel("$z$ axis", fontsize=12, color="k", labelpad=12)
                # ax_xy.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.1f'))
                # ax_xy.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.1f'))
                # ax_xy.zaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.1f'))
                # for t in ax.xaxis.get_major_ticks(): t.label.set_fontsize(8)
                # for t in ax.yaxis.get_major_ticks(): t.label.set_fontsize(8)
                # for t in ax.zaxis.get_major_ticks(): t.label.set_fontsize(8)
                ax.xaxis.set_rotate_label(False)
                ax.yaxis.set_rotate_label(False)
                ax.zaxis.set_rotate_label(False)
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
        scale = float(frame_scale * self.get_arclength(0, 1))

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
