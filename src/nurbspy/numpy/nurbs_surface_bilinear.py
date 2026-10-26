# -------------------------------------------------------------------------------------------------------------------- #
# Import packages
# -------------------------------------------------------------------------------------------------------------------- #
import numpy as np

from .nurbs_surface import NurbsSurface


# -------------------------------------------------------------------------------------------------------------------- #
# Define the bilinear NURBS surface class
# -------------------------------------------------------------------------------------------------------------------- #
class NurbsSurfaceBilinear:

    """ Create a NURBS representation of the bilinear patch defined by corners P00, P01, P10, and P11

        Create a NURBS representation of the bilinear patch
            S(u,v) = (1-v)*[(1-u)*P00 + u*P01] + v*[(1-u)*P10 + u*P11]

        Note that a bilinear patch is a ruled surface with segments (P00, P01) and (P10, P11) as generating curves
            S(u,v) = (1-v)*C1(u) + v*C2(u)
            C1(u) = (1-u)*P00 + u*P01
            C2(u) = (1-u)*P10 + u*P11

    Parameters
    ----------
    P00, P01, P10, P11 : ndarrays with shape (ndim,)
        Coordinates of the corner points defining the bilinear surface (ndim=3)

    References
    ----------
    The NURBS book. Chapter 8.2
    L. Piegl and W. Tiller
    Springer, second edition

    """

    def __init__(self, P00, P01, P10, P11):

        # Declare input variables as instance variables
        self.P00 = P00
        self.P01 = P01
        self.P10 = P10
        self.P11 = P11
        self.ndim = 3

        # Check the number of dimensions of the problem
        ndims = [np.shape(P00)[0], np.shape(P01)[0], np.shape(P10)[0], np.shape(P11)[0]]
        if any([ndim != 3 for ndim in ndims]):
            raise Exception("The input points must be three-dimensional")

        # Make the bilinear patch NURBS representation
        self.NurbsSurface = None
        self.make_nurbs_surface()


    def make_nurbs_surface(self):

        """ Make a NURBS surface representation of the bilinear surface """

        # Define the array of control points
        n_dim, n, m = self.ndim, 2, 2
        P = np.zeros((n_dim, n, m))
        P[:, 0, 0] = self.P00
        P[:, 1, 0] = self.P01
        P[:, 0, 1] = self.P10
        P[:, 1, 1] = self.P11

        # Create the NURBS surface
        self.NurbsSurface = NurbsSurface(control_points=P)



