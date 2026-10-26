# -------------------------------------------------------------------------------------------------------------------- #
# Import packages
# -------------------------------------------------------------------------------------------------------------------- #
import numpy as np

from .nurbs_surface import NurbsSurface


# -------------------------------------------------------------------------------------------------------------------- #
# Define the bilinear NURBS surface class
# -------------------------------------------------------------------------------------------------------------------- #
class NurbsSurfaceRuled:

    """ Create the NURBS ruled surface between NURBS curves `C1(u)` and `C2(u)`

        Create the NURBS ruled surface given by
            S(u,v) = (1-v)*C1(u) + v*C2(u)

    Parameters
    ----------
    C1, C2 : NURBS curve objects
        See NurbsCurve class documentation

    References
    ----------
    The NURBS book. Chapter 8.4
    L. Piegl and W. Tiller
    Springer, second edition

    """

    def __init__(self, C1, C2):

        # Declare input variables as instance variables
        self.C1 = C1
        self.C2 = C2

        # Check the that the NURBS curves are conforming
        # It is possible to develop a more robust implementation that can handle NURBSs with different knot vectors but
        # that would require degree elevation / knot insertion (possible line of development)
        if any([size1 != size2 for size1, size2 in zip(np.shape(C1.P), np.shape(C2.P))]):
            raise Exception("The curves must have conforming arrays of control points")

        if C1.p != C2.p:
            raise Exception("The curves must have the same degree")

        if np.shape(C1.U) != np.shape(C2.U):
            raise Exception("The curves must have the same number of knots")

        if any([np.abs(u1-u2)>1e-12 for u1, u2 in zip(C1.U, C2.U)]):
            raise Exception("The curves must have the same knot values")

        # Make the ruled surface NURBS representation
        self.NurbsSurface = None
        self.make_nurbs_surface()


    def make_nurbs_surface(self):

        """ Make a NURBS surface representation of the ruled surface """

        # Define the array of control points
        P = np.concatenate((self.C1.P[:, :, np.newaxis], self.C2.P[:, :, np.newaxis]), axis=2)

        # Maximum index of the control points (counting from zero)
        m = np.shape(P)[2] - 1

        # Define array of control point weights
        W = np.concatenate((self.C1.W[:, np.newaxis], self.C2.W[:, np.newaxis]), axis=1)

        # Define the order of the basis polynomials
        # Linear (p = 1), Quadratic (p = 2), Cubic (p = 3), etc.
        # Set p = n (number of control points minus one) to obtain a Bezier
        p = self.C1.p
        q = 1

        # Define the knot vectors (clamped spline)
        # p+1 zeros, n minus p equispaced points between 0 and 1, and p+1 ones.  In total r+1 points where r=n+p+1
        # q+1 zeros, m minus q equispaced points between 0 and 1, and q+1 ones. In total s+1 points where s=m+q+1
        U = self.C1.U
        V = np.concatenate((np.zeros(q), np.linspace(0, 1, m - q + 2), np.ones(q)))

        # Create the NURBS surface
        self.NurbsSurface = NurbsSurface(control_points=P, weights=W, u_degree=p, v_degree=q, u_knots=U, v_knots=V)


