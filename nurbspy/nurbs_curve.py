# -------------------------------------------------------------------------------------------------------------------- #
# Import packages
# -------------------------------------------------------------------------------------------------------------------- #
import numpy as np
import scipy.special
import scipy.integrate
import scipy.optimize
import matplotlib as mpl
import matplotlib.pyplot as plt

from mpl_toolkits    import mplot3d
from .nurbs_basis_functions    import compute_basis_polynomials, compute_basis_polynomials_derivatives


# -------------------------------------------------------------------------------------------------------------------- #
# Define the NURBS curve class
# -------------------------------------------------------------------------------------------------------------------- #
class NurbsCurve:

    """ Create a NURBS (Non-Uniform Rational Basis Spline) curve object

        Parameters
        ----------
        control_points : ndarray with shape (ndim, n+1)
            Array containing the coordinates of the control points
            The first dimension of ´P´ spans the coordinates of the control points (any number of dimensions)
            The second dimension of ´P´ spans the u-direction control points (0,1,...,n)

        weights : ndarray with shape (n+1,)
            Array containing the weight of the control points

        degree : int
            Degree of the basis polynomials

        knots : ndarray with shape (r+1=n+p+2,)
            The knot vector in the u-direction
            Set the multiplicity of the first and last entries equal to ´p+1´ to obtain a clamped NURBS


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


    # ---------------------------------------------------------------------------------------------------------------- #
    # Select the type of curve based on the initialization arguments
    # ---------------------------------------------------------------------------------------------------------------- #
    def __init__(self, control_points=None, weights=None, degree=None, knots=None):


        # Void initialization
        if control_points is None and weights is None and degree is None and knots is None:
            pass


        # Polynomial Bezier curve initialization
        elif weights is None and degree is None and knots is None:

            # Set the curve type flag
            self.curve_type = 'Bezier'

            # Set the number of dimensions of the problem
            self.ndim = np.shape(control_points)[0]

            # Maximum index of the control points (counting from zero)
            n = np.shape(control_points)[1] - 1

            # Define the weight of the control points
            weights = np.ones((n + 1), dtype=control_points.dtype)

            # Define the order of the basis polynomials
            degree = n

            # Define the knot vector (clamped spline)
            knots = np.concatenate((np.zeros(degree), np.linspace(0, 1, n - degree + 2), np.ones(degree)))


        # Rational Bezier curve initialization
        elif degree is None and knots is None:

            # Set the curve type flag
            self.curve_type = 'R-Bezier'

            # Set the number of dimensions of the problem
            self.ndim = np.shape(control_points)[0]

            # Maximum index of the control points (counting from zero)
            n = np.shape(control_points)[1] - 1

            # Define the order of the basis polynomials
            degree = n

            # Define the knot vector (clamped spline)
            knots = np.concatenate((np.zeros(degree), np.linspace(0, 1, n - degree + 2), np.ones(degree)))


        # B-Spline curve initialization (both degree and knot vector are provided)
        elif weights is None and knots is not None:

            # Set the curve type flag
            self.curve_type = 'B-Spline'

            # Set the number of dimensions of the problem
            self.ndim = np.shape(control_points)[0]

            # Maximum index of the control points (counting from zero)
            n = np.shape(control_points)[1] - 1

            # Define the weight of the control points
            weights = np.ones((n + 1), dtype=control_points.dtype)


        # B-Spline curve initialization (degree is given but the knot vector is not provided)
        elif weights is None and knots is None:

            # Set the curve type flag
            self.curve_type = 'B-Spline'

            # Set the number of dimensions of the problem
            self.ndim = np.shape(control_points)[0]

            # Maximum index of the control points (counting from zero)
            n = np.shape(control_points)[1] - 1

            # Define the knot vector (clamped spline)
            knots = np.concatenate((np.zeros(degree), np.linspace(0, 1, n - degree + 2), np.ones(degree)))

            # Define the weight of the control points
            weights = np.ones((n + 1), dtype=control_points.dtype)


        # NURBS curve initialization
        else:

            # Set the curve type flag
            self.curve_type = 'NURBS'

            # Set the number of dimensions of the problem
            self.ndim = np.shape(control_points)[0]


        # Declare input variables as instance variables
        self.P = control_points
        self.W = weights
        self.p = degree
        self.U = knots

        # Knot vector in OpenCascade format
        self.U_values, self.U_mults = np.unique(self.U, return_counts=True)



    # ---------------------------------------------------------------------------------------------------------------- #
    # Compute the coordinates of the curve
    # ---------------------------------------------------------------------------------------------------------------- #
    def get_value(self, u):

        """ Evaluate the coordinates of the curve for the input u parametrization

        Parameters
        ----------
        u : scalar or ndarray with shape (N,)
            Parameter used to evaluate the curve

        Returns
        -------
        C : ndarray with shape (ndim, N)
            Array containing the coordinates of the curve
            The first dimension of ´C´ spans the ´(x,y,z)´ coordinates
            The second dimension of ´C´ spans the ´u´ parametrization sample points

        """

        # Evaluate the NURBS curve for the the input u-parametrization
        C = self.compute_nurbs_coordinates(self.P, self.W, self.p, self.U, u)

        return C


    @staticmethod
    def compute_nurbs_coordinates(P, W, p, U, u):

        """ Evaluate the coordinates of a NURBS curve for the input u parametrization

        This function computes the coordinates of the NURBS curve in homogeneous space using equation 4.5 and then
        maps the coordinates to ordinary space using the perspective map given by equation 1.16. See algorithm A4.1

        Parameters
        ----------
        P : ndarray with shape (ndim, n+1)
            Array containing the coordinates of the control points
            The first dimension of ´P´ spans the coordinates of the control points (any number of dimensions)
            The second dimension of ´P´ spans the u-direction control points ´(0,1,...,n)´

        W : ndarray with shape (n+1,)
            Array containing the weight of the control points

        p : int
            Degree of the B-Spline basis polynomials

        U : ndarray with shape (r+1=n+p+2,)
            Knot vector in the u-direction
            Set the multiplicity of the first and last entries equal to ´p+1´ to obtain a clamped spline

        u : scalar or ndarray with shape (N,)
            Parameter used to evaluate the curve

        Returns
        -------
        C : ndarray with shape (ndim, N)
            Array containing the coordinates of the curve
            The first dimension of ´C´ spans the ´(x,y,z)´ coordinates
            The second dimension of ´C´ spans the ´u´ parametrization sample points

        """

        # Check the shape of the input parameters
        if P.ndim > 2:           raise Exception('P must be an array of shape (ndim, n+1)')
        if W.ndim > 1:           raise Exception('W must be an array of shape (n+1,)')
        if not np.isscalar(p):   raise Exception('p must be an scalar')
        if U.ndim > 1:           raise Exception('U must be an array of shape (r+1=n+p+2,)')
        if np.isscalar(u):       u = np.asarray(u)
        elif u.ndim > 1:         raise Exception('u must be a scalar or an array of shape (N,)')

        # Shape of the array of control points
        n_dim, nn = np.shape(P)

        # Highest index of the control points (counting from zero)
        n = nn - 1

        # Compute the B-Spline basis polynomials
        N_basis = compute_basis_polynomials(n, p, U, u)

        # Map the control points to homogeneous space | P_w = (x*w,y*w,z*w,w)
        P_w = np.concatenate((P * W[np.newaxis, :], W[np.newaxis, :]), axis=0)

        # Compute the coordinates of the NURBS curve in homogeneous space
        # The summations over n is performed exploiting matrix multiplication (vectorized code)
        C_w = np.dot(P_w, N_basis)

        # Map the coordinates back to the ordinary space
        C = C_w[0:-1,:]/C_w[-1, :]

        return C


    @staticmethod
    def compute_bspline_coordinates(P, p, U, u):

        """ Evaluate the coordinates of a B-Spline curve for the input u parametrization

        This function computes the coordinates of a B-Spline curve as given by equation 3.1. See algorithm A3.1

        Parameters
        ----------
        P : ndarray with shape (ndim, n+1)
            Array containing the coordinates of the control points
            The first dimension of ´P´ spans the coordinates of the control points (any number of dimensions)
            The second dimension of ´P´ spans the u-direction control points ´(0,1,...,n)´

        p : int
            Degree of the B-Spline basis polynomials

        U : ndarray with shape (r+1=n+p+2,)
            Knot vector in the u-direction
            Set the multiplicity of the first and last entries equal to ´p+1´ to obtain a clamped spline

        u : scalar or ndarray with shape (N,)
            Parameter used to evaluate the curve

        Returns
        -------
        C : ndarray with shape (ndim, N)
            Array containing the coordinates of the curve
            The first dimension of ´C´ spans the ´(x,y,z)´ coordinates
            The second dimension of ´C´ spans the ´u´ parametrization sample points

        """

        # Check the shape of the input parameters
        if P.ndim > 2:           raise Exception('P must be an array of shape (ndim, n+1)')
        if not np.isscalar(p):   raise Exception('p must be an scalar')
        if U.ndim > 1:           raise Exception('U must be an array of shape (r+1=n+p+2,)')
        if np.isscalar(u):       u = np.asarray(u)
        elif u.ndim > 1:         raise Exception('u must be a scalar or an array of shape (N,)')

        # Highest index of the control points (counting from zero)
        n = np.shape(P)[1] - 1

        # Compute the B-Spline basis polynomials
        N_basis = compute_basis_polynomials(n, p, U, u)

        # Compute the coordinates of the B-Spline curve
        # The summation over n is performed exploiting matrix multiplication (vectorized code)
        C = np.dot(P, N_basis)

        return C


    # ---------------------------------------------------------------------------------------------------------------- #
    # Compute the derivatives of the curve
    # ---------------------------------------------------------------------------------------------------------------- #
    def get_derivative(self, u, order):

        """ Evaluate the derivative of the curve for the input u-parametrization

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
            The first dimension of ´dC´ spans the ´(x,y,z)´ coordinates
            The second dimension of ´dC´ spans the ´u´ parametrization sample points

        """

        # Compute the array of curve derivatives up to order `derivative_order` and slice the desired values
        dC = self.compute_nurbs_derivatives(self.P, self.W, self.p, self.U, u, order)[order, ...]

        return dC


    def compute_nurbs_derivatives(self, P, W, p, U, u, up_to_order):

        """ Compute the derivatives of a NURBS curve in ordinary space up to the desired order

        This function computes the analytic derivatives of the NURBS curve in ordinary space using equation 4.8 and
        the derivatives of the NURBS curve in homogeneous space obtained from compute_bspline_derivatives()

        The derivatives are computed recursively in a fashion similar to algorithm A4.2

        Parameters
        ----------
        P : ndarray with shape (ndim, n+1)
            Array containing the coordinates of the control points
            The first dimension of ´P´ spans the coordinates of the control points ´(x,y,z)´
            The second dimension of ´P´ spans the u-direction control points ´(0,1,...,n)´

        W : ndarray with shape (n+1,)
            Array containing the weight of the control points

        p : int
            Degree of the B-Spline basis polynomials

        U : ndarray with shape (r+1=n+p+2,)
            Knot vector in the u-direction
            Set the multiplicity of the first and last entries equal to ´p+1´ to obtain a clamped spline

        u : scalar or ndarray with shape (N,)
            Parameter used to evaluate the curve

        up_to_order : integer
            Order of the highest derivative

        Returns
        -------
        nurbs_derivatives: ndarray of shape (up_to_order+1, ndim, Nu)
            The first dimension spans the order of the derivatives (0, 1, 2, ...)
            The second dimension spans the coordinates (x,y,z,...)
            The third dimension spans u-parametrization sample points

        """

        # Set the data type used to initialize arrays (set `complex` if an argument is complex and `float` if not)
        u = np.asarray(u)
        if (P.dtype == np.complex128) or (W.dtype == np.complex128) or (u.dtype == np.complex128):
            data_type = np.complex128
        else:
            data_type = np.float64

        # Map the control points to homogeneous space | P_w = (x*w,y*w,z*w,w)
        P_w = np.concatenate((P * W[np.newaxis, :], W[np.newaxis, :]), axis=0)

        # Compute the derivatives of the NURBS curve in homogeneous space
        bspline_derivatives = self.compute_bspline_derivatives(P_w, p, U, u, up_to_order)
        A_ders = bspline_derivatives[:, 0:-1, :]
        w_ders = bspline_derivatives[:, [-1], :]

        # Initialize array of derivatives
        n_dim, Nu = np.shape(P)[0], np.asarray(u).size                               # Get sizes
        nurbs_derivatives = np.zeros((up_to_order+1, n_dim, Nu), dtype=data_type)    # Initialize array with zeros

        # Compute the derivatives of up to the desired order
        # See algorithm A4.2 from the NURBS book
        for order in range(up_to_order+1):

            # Update the numerator of equation 4.8 recursively
            temp_numerator = A_ders[[order], ...]
            for i in range(1, order+1):
                temp_numerator -= scipy.special.binom(order, i) * w_ders[[i], ...] * nurbs_derivatives[[order-i], ...]

            # Compute the k-th order NURBS curve derivative
            nurbs_derivatives[order, ...] = temp_numerator/w_ders[[0], ...]

        return nurbs_derivatives


    @staticmethod
    def compute_bspline_derivatives(P, p, U, u, up_to_order):

        """ Compute the derivatives of a B-Spline (or NURBS curve in homogeneous space) up to order `derivative_order`

        This function computes the analytic derivatives of a B-Spline curve using equation 3.3. See algorithm A3.2

        Parameters
        ----------
        P : ndarray with shape (ndim, n+1)
            Array containing the coordinates of the control points
            The first dimension of ´P´ spans the coordinates of the control points ´(x,y,z)´
            The second dimension of ´P´ spans the u-direction control points ´(0,1,...,n)´

        p : int
            Degree of the B-Spline basis polynomials

        U : ndarray with shape (r+1=n+p+2,)
            Knot vector in the u-direction
            Set the multiplicity of the first and last entries equal to ´p+1´ to obtain a clamped spline

        u : scalar or ndarray with shape (N,)
            Parameter used to evaluate the curve

        up_to_order : integer
            Order of the highest derivative

        Returns
        -------
        bspline_derivatives: ndarray of shape (up_to_order+1, ndim, Nu)
            The first dimension spans the order of the derivatives (0, 1, 2, ...)
            The second dimension spans the coordinates (x,y,z,...)
            The third dimension spans u-parametrization sample points

        """

        # Set the data type used to initialize arrays (set `complex` if an argument is complex and `float` if not)
        u = np.asarray(u)
        if (P.dtype == np.complex128) or (u.dtype == np.complex128):
            data_type = np.complex128
        else:
            data_type = np.float64

        # Set the B-Spline coordinates as the zero-th derivative
        n_dim, Nu = np.shape(P)[0], np.asarray(u).size
        bspline_derivatives = np.zeros((up_to_order+1, n_dim, Nu), dtype=data_type)

        # Compute the derivatives of up to the desired order (start at index 1 and end at index `p`)
        # See algorithm A3.2 from the NURBS book
        for order_u in range(min(p, up_to_order) + 1):

            # Highest index of the control points (counting from zero)
            n = np.shape(P)[1] - 1

            # Compute the B-Spline basis polynomials
            N_basis = compute_basis_polynomials_derivatives(n, p, U, u, order_u)

            # Compute the coordinates of the B-Spline
            # The summations over n is performed exploiting matrix multiplication (vectorized code)
            bspline_derivatives[order_u, :, :] = np.dot(P, N_basis)

        # Note that derivative with order higher than `p` are not computed and are be zero from initialization
        # These zero-derivatives are required to compute the higher order derivatives of rational curves

        return bspline_derivatives


    # ---------------------------------------------------------------------------------------------------------------- #
    # Miscellaneous methods
    # ---------------------------------------------------------------------------------------------------------------- #
    def attach_nurbs(self, new_nurbs):

        """ Attatch a new NURBS curve to the end of the instance NURBS curve and return the merged NURBS curve"""

        # Check that the NURBS curves have the same degree
        if self.p != new_nurbs.p:
            raise Exception("In order to merge, the two NURBS curves must have the same degree")

        # Combine the control points
        P = np.concatenate((self.P, new_nurbs.P), axis=1)

        # Combine the control point weights
        W = np.concatenate((self.W, new_nurbs.W), axis=0)

        # Highest index of the control points
        n1 = np.shape(self.P)[1] - 1
        n2 = np.shape(new_nurbs.P)[1] - 1

        # Combine the knot vectors (inner knot has p+1 multiplicity)
        eps = 0
        U_start = np.zeros((self.p + 1,))
        U_end = np.ones((self.p + 1,))
        U_mid = np.ones((self.p + 1,)) / 2
        U_mid[0] = U_mid[0] - eps       # Quick and dirty fix for GMSH (avoid multiplicity equal to degree)
        U_mid[-1] = U_mid[-1] + eps     # Quick and dirty fix for GMSH (avoid multiplicity equal to degree)
        U1 = 0.00 + self.U[self.p + 1:n1 + 1] / 2
        U2 = 0.50 + new_nurbs.U[self.p + 1:n2 + 1] / 2
        U = np.concatenate((U_start, U1, U_mid, U2, U_end))

        # Create the merged NURBS curve
        mergedNurbs = NurbsCurve(control_points=P, weights=W, degree=self.p, knots=U)

        return mergedNurbs


    # ---------------------------------------------------------------------------------------------------------------- #
    # Compute the Frenet-Serret unitary vectors
    # ---------------------------------------------------------------------------------------------------------------- #
    def get_tangent(self, u):

        """ Evaluate the unitary tangent vector to the curve for the input u-parametrization

        The definition of the unitary tangent vector is given by equation 10.5 (Farin's textbook)

        Parameters
        ----------
        u : scalar or ndarray with shape (N,)
            Scalar or array containing the u-parameter used to evaluate the function

        Returns
        -------
        tangent : ndarray with shape (ndim, N)
            Array containing the unitary tangent vector
            The first dimension of ´tangent´ spans the ´(x,y,z)´ coordinates
            The second dimension of ´tangent´ spans the ´u´ parametrization sample points

        """


        # Compute the curve derivatives
        dC, = self.compute_nurbs_derivatives(self.P, self.W, self.p, self.U, u, up_to_order=1)[[1], ...]

        # Compute the tangent vector
        numerator = dC
        denominator = (np.sum(numerator ** 2, axis=0)) ** (1 / 2)
        tangent = numerator / denominator

        return tangent


    def get_normal(self, u):

        """ Evaluate the unitary normal vector to the curve for the input u-parametrization

        The definition of the unitary normal vector is given by equation 10.5 (Farin's textbook)

        Parameters
        ----------
        u : scalar or ndarray with shape (N,)
            Scalar or array containing the u-parameter used to evaluate the function

        Returns
        -------
        normal : ndarray with shape (ndim, N)
            Array containing the unitary normal vector
            The first dimension of ´normal´ spans the ´(x,y,z)´ coordinates
            The second dimension of ´normal´ spans the ´u´ parametrization sample points

        """

        # Compute the curve derivatives
        dC, ddC = self.compute_nurbs_derivatives(self.P, self.W, self.p, self.U, u, up_to_order=2)[[1, 2], ...]


        # Compute the normal vector
        if self.ndim == 2:
            dC = np.concatenate((dC, np.zeros((1, np.asarray(u).size))), axis=0)
            ddC = np.concatenate((ddC, np.zeros((1, np.asarray(u).size))), axis=0)
            numerator = np.cross(dC, np.cross(ddC, dC, axisa=0, axisb=0, axisc=0), axisa=0, axisb=0, axisc=0)
            denominator = (np.sum(numerator ** 2, axis=0)) ** (1 / 2)
            normal = (numerator / denominator)[[0, 1], ...]

        elif self.ndim == 3:
            numerator = np.cross(dC, np.cross(ddC, dC, axisa=0, axisb=0, axisc=0), axisa=0, axisb=0, axisc=0)
            denominator = (np.sum(numerator ** 2, axis=0)) ** (1 / 2)
            normal = numerator / denominator

        else: raise Exception("The number of dimensions must be 2 or 3")

        return normal


    def get_binormal(self, u):

        """ Evaluate the unitary binormal vector to the curve for the input u-parametrization

        The definition of the unitary binormal vector is given by equation 10.5 (Farin's textbook)

        Parameters
        ----------
        u : scalar or ndarray with shape (N,)
            Scalar or array containing the u-parameter used to evaluate the function

        Returns
        -------
        binormal : ndarray with shape (ndim, N)
            Array containing the unitary binormal vector
            The first dimension of ´binormal´ spans the ´(x,y,z)´ coordinates
            The second dimension of ´binormal´ spans the ´u´ parametrization sample points

        """

        # Compute the curve derivatives
        dC, ddC = self.compute_nurbs_derivatives(self.P, self.W, self.p, self.U, u, up_to_order=2)[[1, 2], ...]

        # Compute the binormal vector
        if self.ndim == 2:
            dC = np.concatenate((dC, np.zeros((1, np.asarray(u).size))), axis=0)
            ddC = np.concatenate((ddC, np.zeros((1, np.asarray(u).size))), axis=0)
            numerator = np.cross(dC, ddC, axisa=0, axisb=0, axisc=0)
            denominator = (np.sum(numerator ** 2, axis=0)) ** (1 / 2)
            binormal = (numerator / denominator)[[0, 1], ...]

        elif self.ndim == 3:
            numerator = np.cross(dC, ddC, axisa=0, axisb=0, axisc=0)
            denominator = (np.sum(numerator ** 2, axis=0)) ** (1 / 2)
            binormal = numerator / denominator

        else: raise Exception("The number of dimensions must be 2 or 3")

        return binormal


    def get_normal_2D(self, u):

        """ Evaluate the the unitary normal vector using the special formula 2D formula

        Parameters
        ----------
        u : scalar or ndarray with shape (N,)
            Scalar or array containing the u-parameter used to evaluate the function

        Returns
        -------
        normal : ndarray with shape (2, N)
            Array containing the unitary normal vector
            The first dimension of ´normal´ spans the ´(x,y,z)´ coordinates
            The second dimension of ´normal´ spans the ´u´ parametrization sample points

        """

        # Compute the curve derivatives
        u = np.asarray(u)
        dC = self.compute_nurbs_derivatives(self.P, self.W, self.p, self.U, u, up_to_order=1)[1, ...]

        # Compute the normal vector
        if self.ndim == 2:
            numerator = np.concatenate((-dC[[1], :], dC[[0], :]), axis=0)
            denominator = (np.sum(numerator ** 2, axis=0)) ** (1 / 2)
            normal = numerator / denominator

        else: raise Exception("The number of dimensions must be 2")

        return normal


    # ---------------------------------------------------------------------------------------------------------------- #
    # Compute the curvature, torsion, and arc-length
    # ---------------------------------------------------------------------------------------------------------------- #
    def get_curvature(self, u):

        """ Evaluate the curvature of the curve for the input u-parametrization

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

        # Compute the curve derivatives
        u = np.asarray(u)
        dC, ddC = self.compute_nurbs_derivatives(self.P, self.W, self.p, self.U, u, up_to_order=2)[[1, 2], ...]

        # Compute the curvature
        if self.ndim == 2:
            dC = np.concatenate((dC, np.zeros((1, np.asarray(u).size))), axis=0)
            ddC = np.concatenate((ddC, np.zeros((1, np.asarray(u).size))), axis=0)
            numerator = np.sum(np.cross(ddC, dC, axisa=0, axisb=0, axisc=0) ** 2, axis=0) ** (1 / 2)
            denominator = (np.sum(dC ** 2, axis=0)) ** (3 / 2)
            curvature = (numerator / denominator)

        elif self.ndim == 3:
            numerator = np.sum(np.cross(ddC, dC, axisa=0, axisb=0, axisc=0) ** 2, axis=0) ** (1 / 2)
            denominator = (np.sum(dC ** 2, axis=0)) ** (3 / 2)
            curvature = numerator / denominator

        else: raise Exception("The number of dimensions must be 2 or 3")

        return curvature


    def get_torsion(self, u):

        """ Evaluate the torsion of the curve for the input u-parametrization

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

        # Compute the curve derivatives
        u = np.asarray(u)
        dC, ddC, dddC = self.compute_nurbs_derivatives(self.P, self.W, self.p, self.U, u, up_to_order=3)[[1, 2, 3], ...]

        # Compute the torsion
        if self.ndim == 2:
            dC = np.concatenate((dC, np.zeros((1, np.asarray(u).size))), axis=0)
            ddC = np.concatenate((ddC, np.zeros((1, np.asarray(u).size))), axis=0)
            dddC = np.concatenate((dddC, np.zeros((1, np.asarray(u).size))), axis=0)
            numerator = np.sum(np.cross(dC, ddC, axisa=0, axisb=0, axisc=0) * dddC, axis=0)
            denominator = np.sum(np.cross(dC, ddC, axisa=0, axisb=0, axisc=0)**2, axis=0)
            torsion = (numerator / denominator)

        elif self.ndim == 3:
            numerator = np.sum(np.cross(dC, ddC, axisa=0, axisb=0, axisc=0) * dddC, axis=0)
            denominator = np.sum(np.cross(dC, ddC, axisa=0, axisb=0, axisc=0)**2, axis=0)
            torsion = numerator / denominator

        else: raise Exception("The number of dimensions must be 2 or 3")

        return torsion


    def get_arclength(self, u1=0.00, u2=1.00):

        """ Compute the arc length of a parametric curve in the interval [u1,u2] using numerical quadrature

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

        # Compute the arc length differential analytically
        def get_arclegth_differential(u):
            dCdu = self.get_derivative(u, order=1)
            dLdu = np.sqrt(np.sum(dCdu ** 2, axis=0))  # dL/du = [(dx_0/du)^2 + ... + (dx_n/du)^2]^(1/2)
            return dLdu.item() if dLdu.size == 1 else dLdu

        # Compute the arc length of C(t) in the interval [u1, u2] by numerical integration
        arclength = scipy.integrate.fixed_quad(get_arclegth_differential, u1, u2, n=10)[0]

        return arclength



    # ---------------------------------------------------------------------------------------------------------------- #
    # Plot the NURBS curve
    # ---------------------------------------------------------------------------------------------------------------- #
    def plot(self, fig=None, ax=None,
             curve=True, control_points=True, frenet_serret=False, axis_off=False, ticks_off=False):

        """ Create a plot and return the figure and axes handles """

        if fig is None:

            # One dimension (law of evolution)
            if self.ndim == 1:
                fig = plt.figure(figsize=(6, 5))
                ax = fig.add_subplot(111)
                ax.set_xlabel('$u$ parameter', fontsize=12, color='k', labelpad=12)
                ax.set_ylabel('NURBS curve value', fontsize=12, color='k', labelpad=12)
                # ax_xy.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.1f'))
                # ax_xy.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.1f'))
                # for t in ax.xaxis.get_major_ticks(): t.label.set_fontsize(12)
                # for t in ax.yaxis.get_major_ticks(): t.label.set_fontsize(12)
                if ticks_off:
                    ax.set_xticks([])
                    ax.set_yticks([])
                if axis_off:
                    ax.axis('off')

            # Two dimensions (plane curve)
            elif self.ndim == 2:
                fig = plt.figure(figsize=(6, 5))
                ax = fig.add_subplot(111)
                ax.set_xlabel('$x$ axis', fontsize=12, color='k', labelpad=12)
                ax.set_ylabel('$y$ axis', fontsize=12, color='k', labelpad=12)
                # ax_xy.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.1f'))
                # ax_xy.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.1f'))
                # for t in ax.xaxis.get_major_ticks(): t.label.set_fontsize(12)
                # for t in ax.yaxis.get_major_ticks(): t.label.set_fontsize(12)
                if ticks_off:
                    ax.set_xticks([])
                    ax.set_yticks([])
                if axis_off:
                    ax.axis('off')

            # Three dimensions (space curve)
            elif self.ndim == 3:
                fig = mpl.pyplot.figure(figsize=(6, 5))
                ax = fig.add_subplot(111, projection='3d')
                ax.view_init(azim=-120, elev=30)
                ax.grid(False)
                ax.xaxis.pane.fill = False
                ax.yaxis.pane.fill = False
                ax.zaxis.pane.fill = False
                ax.xaxis.pane.set_edgecolor('k')
                ax.yaxis.pane.set_edgecolor('k')
                ax.zaxis.pane.set_edgecolor('k')
                ax.xaxis.pane._alpha = 0.9
                ax.yaxis.pane._alpha = 0.9
                ax.zaxis.pane._alpha = 0.9
                ax.set_xlabel('$x$ axis', fontsize=12, color='k', labelpad=12)
                ax.set_ylabel('$y$ axis', fontsize=12, color='k', labelpad=12)
                ax.set_zlabel('$z$ axis', fontsize=12, color='k', labelpad=12)
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
                    ax.axis('off')

            else: raise Exception('The number of dimensions must be 1, 2 or 3')


        # Add objects to the plot
        if curve:          self.plot_curve(fig, ax)
        if control_points: self.plot_control_points(fig, ax)
        if frenet_serret:  self.plot_frenet_serret(fig, ax)

        # Set the scaling of the axes
        self.rescale_plot(fig, ax)

        return fig, ax


    def plot_curve(self, fig, ax, linewidth=1.5, linestyle='-', color='black', u1=0.00, u2=1.00):

        """ Plot the coordinates of the NURBS curve """

        # One dimension (law of evolution)
        if self.ndim == 1:
            u = np.linspace(u1, u2, 501)
            X = np.real(self.get_value(u))
            line, = ax.plot(u, X[0,:])
            line.set_linewidth(linewidth)
            line.set_linestyle(linestyle)
            line.set_color(color)
            line.set_marker(' ')
            # line.set_label(' ')

        # Two dimensions (plane curve)
        elif self.ndim == 2:
            u = np.linspace(u1, u2, 501)
            X, Y = np.real(self.get_value(u))
            line, = ax.plot(X, Y)
            line.set_linewidth(linewidth)
            line.set_linestyle(linestyle)
            line.set_color(color)
            line.set_marker(' ')
            # line.set_label(' ')

        # Three dimensions (space curve)
        elif self.ndim == 3:
            u = np.linspace(u1, u2, 501)
            X, Y, Z = np.real(self.get_value(u))
            line, = ax.plot(X, Y, Z)
            line.set_linewidth(linewidth)
            line.set_linestyle(linestyle)
            line.set_color(color)
            line.set_marker(' ')
            # line.set_label(' ')

        else: raise Exception('The number of dimensions must be 1, 2 or 3')

        return fig, ax


    def plot_control_points(self, fig, ax, linewidth=1.25, linestyle='-.', color='red', markersize=5, markerstyle='o'):

        """ Plot the control points of the NURBS curve """

        # One dimension (law of evolution)
        if self.ndim == 1:
            Px = np.real(self.P)
            u = np.linspace(0, 1, Px.size)
            line, = ax.plot(u, Px[0,:])
            line.set_linewidth(linewidth)
            line.set_linestyle(linestyle)
            line.set_color(color)
            line.set_marker(markerstyle)
            line.set_markersize(markersize)
            line.set_markeredgewidth(linewidth)
            line.set_markeredgecolor(color)
            line.set_markerfacecolor('w')
            line.set_zorder(4)
            # line.set_label(' ')


        # Two dimensions (plane curve)
        elif self.ndim == 2:
            Px, Py = np.real(self.P)
            line, = ax.plot(Px, Py)
            line.set_linewidth(linewidth)
            line.set_linestyle(linestyle)
            line.set_color(color)
            line.set_marker(markerstyle)
            line.set_markersize(markersize)
            line.set_markeredgewidth(linewidth)
            line.set_markeredgecolor(color)
            line.set_markerfacecolor('w')
            line.set_zorder(4)
            # line.set_label(' ')

        # Three dimensions (space curve)
        elif self.ndim == 3:
            Px, Py, Pz = np.real(self.P)
            line, = ax.plot(Px, Py, Pz)
            line.set_linewidth(linewidth)
            line.set_linestyle(linestyle)
            line.set_color(color)
            line.set_marker(markerstyle)
            line.set_markersize(markersize)
            line.set_markeredgewidth(linewidth)
            line.set_markeredgecolor(color)
            line.set_markerfacecolor('w')
            line.set_zorder(4)
            # line.set_label(' ')

        else: raise Exception('The number of dimensions must be 2 or 3')

        return fig, ax


    def plot_frenet_serret(self, fig, ax, frame_number=5, frame_scale=0.10):

        """ Plot some Frenet-Serret reference frames along the NURBS curve """

        # Compute the tangent, normal and binormal unitary vectors
        h = 1e-12
        u = np.linspace(0+h, 1-h, frame_number)
        position = np.real(self.get_value(u))
        tangent = np.real(self.get_tangent(u))
        normal = np.real(self.get_normal(u))
        binormal = np.real(self.get_binormal(u))

        # Two dimensions (plane curve)
        if self.ndim == 2:

            # Plot the frames of reference
            for k in range(frame_number):

                # Plot the tangent vector
                x, y = position[:, k]
                u, v = tangent[:, k]
                ax.quiver(x, y, u, v, color='red', scale=7.5)

                # Plot the normal vector
                x, y = position[:, k]
                u, v = normal[:, k]
                ax.quiver(x, y, u, v, color='blue', scale=7.5)

            # Plot the origin of the vectors
            x, y = position
            points, = ax.plot(x, y)
            points.set_linestyle(' ')
            points.set_marker('o')
            points.set_markersize(5)
            points.set_markeredgewidth(1.25)
            points.set_markeredgecolor('k')
            points.set_markerfacecolor('w')
            points.set_zorder(4)
            # points.set_label(' ')

        # Three dimensions (space curve)
        elif self.ndim == 3:

            # Compute a length scale (fraction of the curve arc length)
            scale = frame_scale * self.get_arclength(0, 1)

            # Plot the frames of reference
            for k in range(frame_number):

                # Plot the tangent vector
                x, y, z = position[:, k]
                u, v, w = tangent[:, k]
                ax.quiver(x, y, z, u, v, w, color='red', length=scale, normalize=True)

                # Plot the norma vector
                x, y, z = position[:, k]
                u, v, w = normal[:, k]
                ax.quiver(x, y, z, u, v, w, color='blue', length=scale, normalize=True)

                # Plot the binormal vector
                x, y, z = position[:, k]
                u, v, w = binormal[:, k]
                ax.quiver(x, y, z, u, v, w, color='green', length=scale, normalize=True)

            # Plot the origin of the vectors
            x, y, z = position
            points, = ax.plot(x, y, z)
            points.set_linestyle(' ')
            points.set_marker('o')
            points.set_markersize(5)
            points.set_markeredgewidth(1.25)
            points.set_markeredgecolor('k')
            points.set_markerfacecolor('w')
            points.set_zorder(4)
            # points.set_label(' ')


        else: raise Exception('The number of dimensions must be 2 or 3')

        return fig, ax


    def rescale_plot(self, fig, ax):

        """ Adjust the aspect ratio of the figure """

        # Two dimensions (plane curve)
        if self.ndim == 2:

            # Set the aspect ratio of the data
            ax.set_aspect(1.0)

            # Adjust pad
            plt.tight_layout(pad=5.0, w_pad=None, h_pad=None)

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
            L = np.max((x_max - x_min, y_max - y_min, z_max - z_min)) / 2
            ax.set_xlim3d(x_mid - 1.0 * L, x_mid + 1.0 * L)
            ax.set_ylim3d(y_mid - 1.0 * L, y_mid + 1.0 * L)
            ax.set_zlim3d(z_mid - 1.0 * L, z_mid + 1.0 * L)

            # Adjust pad
            plt.tight_layout(pad=5.0, w_pad=None, h_pad=None)


    def plot_curvature(self, fig=None, ax=None, color='black', linestyle='-'):

        # Create the figure
        if fig is None:
            fig = plt.figure(figsize=(6, 5))
            ax = fig.add_subplot(111)
        ax.set_xlabel('$u$ parameter', fontsize=12, color='k', labelpad=12)
        ax.set_ylabel('Curvature', fontsize=12, color='k', labelpad=12)
        # ax_xy.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.1f'))
        # ax_xy.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.1f'))
        for t in ax.xaxis.get_major_ticks(): t.label.set_fontsize(12)
        for t in ax.yaxis.get_major_ticks(): t.label.set_fontsize(12)
        # ax_xy.set_xticks([])
        # ax_xy.set_yticks([])
        # ax_xy.axis('off')

        # Plot the curvature distribution
        u = np.linspace(0, 1, 1000)
        curvature = np.real(self.get_curvature(u))
        line, = ax.plot(u, curvature)
        line.set_linewidth(1.25)
        line.set_linestyle(linestyle)
        line.set_color(color)
        line.set_marker(" ")
        line.set_markersize(3.5)
        line.set_markeredgewidth(1)
        line.set_markeredgecolor("k")
        line.set_markerfacecolor("w")
        # line.set_label(' ')

        # Set the aspect ratio of the data
        # ax_xy.set_aspect(1.0)

        # # Set the aspect ratio of the figure
        # ratio = 1.00
        # x1, x2 = ax_xy.get_xlim()
        # y1, y2 = ax_xy.get_ylim()
        # ax_xy.set_aspect(np.abs((x2-x1)/(y2-y1))*ratio)

        # Adjust pad
        plt.tight_layout(pad=5.0, w_pad=None, h_pad=None)

        return fig, ax

    def plot_torsion(self, fig=None, ax=None, color='black', linestyle='-'):

        # Create the figure
        if fig is None:
            fig = plt.figure(figsize=(6, 5))
            ax = fig.add_subplot(111)
        ax.set_xlabel('$u$ parameter', fontsize=12, color='k', labelpad=12)
        ax.set_ylabel('Torsion', fontsize=12, color='k', labelpad=12)
        # ax_xy.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.1f'))
        # ax_xy.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.1f'))
        for t in ax.xaxis.get_major_ticks(): t.label.set_fontsize(12)
        for t in ax.yaxis.get_major_ticks(): t.label.set_fontsize(12)
        # ax_xy.set_xticks([])
        # ax_xy.set_yticks([])
        # ax_xy.axis('off')

        # Plot the curvature distribution
        u = np.linspace(0, 1, 1000)
        torsion = np.real(self.get_torsion(u))
        line, = ax.plot(u, torsion)
        line.set_linewidth(1.25)
        line.set_linestyle(linestyle)
        line.set_color(color)
        line.set_marker(" ")
        line.set_markersize(3.5)
        line.set_markeredgewidth(1)
        line.set_markeredgecolor("k")
        line.set_markerfacecolor("w")
        # line.set_label(' ')

        # Set the aspect ratio of the data
        # ax_xy.set_aspect(1.0)

        # Set the aspect ratio of the figure
        ratio = 1.00
        x1, x2 = ax.get_xlim()
        y1, y2 = ax.get_ylim()
        ax.set_aspect(np.abs((x2 - x1) / (y2 - y1)) * ratio)

        # Adjust pad
        plt.tight_layout(pad=5.0, w_pad=None, h_pad=None)

        return fig, ax


    # ---------------------------------------------------------------------------------------------------------------- #
    # Define the point projection problem class (Pygmo's user-defined problem)
    # ---------------------------------------------------------------------------------------------------------------- #
    def project_point_to_curve(self, P):

        """ Solve the point projection problem for the prescribed point `P` """

        # Create the problem instance
        problem = self.PointToCurveProjectionProblem(self.get_value, self.get_derivative, P)

        # Bounds for u
        bounds = list(zip(*problem.get_bounds()))  # -> [(0.0, 1.0)]

        # Objective and gradient
        objective = lambda x: problem.fitness(x)[0]
        gradient = lambda x: problem.gradient(x)

        # Generate midpoints of knot spans
        U0 = self.U[0:-1] + 0.5 * (self.U[1:] - self.U[0:-1])

        # Select best starting point
        best_x0 = None
        best_f = np.inf
        for u0 in U0:
            x0 = np.array([u0])
            f_val = objective(x0)
            if f_val < best_f:
                best_f = f_val
                best_x0 = x0

        # Run optimization
        result = scipy.optimize.minimize(
            fun=objective,
            x0=best_x0,
            jac=gradient,
            bounds=bounds,
            method="L-BFGS-B",
            options={
                "disp": False,
                "maxiter": 100,
                "ftol": 1e-6,
                "gtol": 1e-6,
            },
        )

        if not result.success:
            raise RuntimeError(f"Optimization failed: {result.message}")

        return result.x[0]

        # # Import pygmo
        # import pygmo as pg

        # # Create the optimization algorithm
        # myAlgorithm = pg.algorithm(pg.nlopt(algorithm_name))
        # myAlgorithm.extract(pg.nlopt).xtol_rel = 1e-6
        # myAlgorithm.extract(pg.nlopt).ftol_rel = 1e-6
        # myAlgorithm.extract(pg.nlopt).xtol_abs = 1e-6
        # myAlgorithm.extract(pg.nlopt).ftol_abs = 1e-6
        # myAlgorithm.extract(pg.nlopt).maxeval = 100
        # myAlgorithm.set_verbosity(0)

        # # Create the optimization problem
        # myProblem = pg.problem(self.PointToCurveProjectionProblem(self.get_value, self.get_derivative, P))

        # # Create the population
        # myPopulation = pg.population(prob=myProblem, size=0)

        # # Create a list with the different starting points
        # U0 = self.U[0:-1] + 1/2 * (self.U[1:] - self.U[0:-1])
        # for u0 in U0:
        #     myPopulation.push_back([u0])

        # # Solve the optimization problem (evolve the population in Pygmo's jargon)
        # myPopulation = myAlgorithm.evolve(myPopulation)

        # # Get the optimum
        # u = myPopulation.champion_x[0]

        # return u

    class PointToCurveProjectionProblem:

        def __init__(self, C, dC, P):
            """ Solve point inversion problem: min(u) ||C(u) - P|| """
            self.C_func = C
            self.dC_func = dC
            self.P = np.reshape(P, (P.shape[0], 1))

        @staticmethod
        def get_bounds():
            """ Set the bounds for the optimization problem """
            return [0.00], [1.00]

        def fitness(self, x):
            """ Evaluate the deviation between the prescribed point and the parametrized point """
            u = np.asarray([x[0]])
            C = self.C_func(u)
            P = self.P
            return np.asarray([np.sum(np.sum((C - P) ** 2, axis=0) ** (1 / 2))])

        def gradient(self, x):
            """ Compute the gradient of the fitness function analytically """
            u = np.asarray([x[0]])
            C = self.C_func(u)
            dC = self.dC_func(u, order=1)
            P = self.P
            numerator = np.sum((C - P) * dC, axis=0)
            denominator = np.sum(np.sum((C - P) ** 2, axis=0) ** (1 / 2))
            if np.abs(denominator) > 0:
                gradient = numerator/denominator
            else:
                gradient = np.asarray(0)[np.newaxis]
            return gradient
