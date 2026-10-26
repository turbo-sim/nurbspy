# -------------------------------------------------------------------------------------------------------------------- #
# Import packages
# -------------------------------------------------------------------------------------------------------------------- #
import numpy as np

from .nurbs_surface import NurbsSurface


# -------------------------------------------------------------------------------------------------------------------- #
# Define the bilinear NURBS surface class
# -------------------------------------------------------------------------------------------------------------------- #
class NurbsSurfaceExtruded:

    """ Create the NURBS surface obtained by sweeping the NURBS curve `C(u)` a distance `d` along a direction `D`

        Create the NURBS surface given by
            S(u,v) = C(u) + v*d*D

    Parameters
    ----------
    C : NURBS curve object
        See NurbsSurface class documentation

    D : ndaray with shape (3,)
        Sweeping/extrusion direction. D may be a unitary or a non-unitary vector

    d : scalar
        Sweeping/extrusion length. d may be positive, negative or zero

    References
    ----------
    The NURBS book. Chapter 8.3
    L. Piegl and W. Tiller
    Springer, second edition

    """

    def __init__(self, C, D, d):

        # Declare input variables as instance variables
        self.C = C
        self.D = D/np.sum(D**2)**(1/2)
        self.d = d

        # Check the number of dimensions of the problem
        if np.shape(C.P)[0] != 3: raise Exception("The input NURBS must be three-dimensional")
        if np.shape(D)[0] != 3: raise Exception("The sweeping direction must be a three-dimensional vector")

        # Make the extrusion surface NURBS representation
        self.NurbsSurface = None
        self.make_Nurbs_surface()


    def make_Nurbs_surface(self):

        """ Make a NURBS surface representation of the extruded surface """

        # Define the array of control points
        P = np.concatenate((self.C.P[:, :, np.newaxis], self.C.P[:, :, np.newaxis] + self.d*self.D[:, np.newaxis, np.newaxis]), axis=2)

        # Define array of control point weights
        W = np.concatenate((self.C.W[:, np.newaxis], self.C.W[:, np.newaxis]), axis=1)

        # Maximum index of the control points (counting from zero)
        m = np.shape(P)[2] - 1

        # Define the order of the basis polynomials
        # Linear (p = 1), Quadratic (p = 2), Cubic (p = 3), etc.
        # Set p = n (number of control points minus one) to obtain a Bezier
        p = self.C.p
        q = 1

        # Define the knot vectors (clamped spline)
        # p+1 zeros, n minus p equispaced points between 0 and 1, and p+1 ones.  In total r+1 points where r=n+p+1
        # q+1 zeros, m minus q equispaced points between 0 and 1, and q+1 ones. In total s+1 points where s=m+q+1
        U = self.C.U
        V = np.concatenate((np.zeros(q), np.linspace(0, 1, m - q + 2), np.ones(q)))

        # Create the NURBS surface
        self.NurbsSurface = NurbsSurface(control_points=P, weights=W, u_degree=p, v_degree=q, u_knots=U, v_knots=V)


