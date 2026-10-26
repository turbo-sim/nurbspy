# -------------------------------------------------------------------------------------------------------------------- #
# Import packages
# -------------------------------------------------------------------------------------------------------------------- #
import numpy as np

from .nurbs_surface import NurbsSurface


# -------------------------------------------------------------------------------------------------------------------- #
# Define the Coons NURBS surface class
# -------------------------------------------------------------------------------------------------------------------- #
class NurbsSurfaceCoons:

    """ Create the NURBS Coons surface comprised within boundaries `C_south(u)`, `C_north(u)`, `C_west(v)`, and `C_east(v)`

    Parameters
    ----------
    C_south, C_north, C_west, C_east: NURBS curve objects
        See NurbsSurface class documentation

    References
    ----------
    The NURBS book. Chapter X.Y
    L. Piegl and W. Tiller
    Springer, second edition

    """

    def __init__(self, C_south, C_north, C_west, C_east):

        # Declare input variables as instance variables
        self.C_north = C_north
        self.C_south = C_south
        self.C_west  = C_west
        self.C_east  = C_east
        self.P_north = C_north.P
        self.P_south = C_south.P
        self.P_west  = C_west.P
        self.P_east  = C_east.P
        self.W_north = C_north.W
        self.W_south = C_south.W
        self.W_west  = C_west.W
        self.W_east  = C_east.W

        # Define a tolerance to check compatibility
        tol = 1e-12

        # Check the that the NURBS curves are conforming
        # It is possible to develop a more robust implementation that can handle NURBSs with different knot vectors

        # Check the number of control points
        if any([size1 != size2 for size1, size2 in zip(np.shape(C_north.P), np.shape(C_south.P))]):
            raise Exception("The curves(C_north,C_south) must have conforming arrays of control points")

        if any([size1 != size2 for size1, size2 in zip(np.shape(C_west.P), np.shape(C_east.P))]):
            raise Exception("The curves (C_west,C_east) must have conforming arrays of control points")


        # Check the number of weights
        if any([size1 != size2 for size1, size2 in zip(np.shape(C_north.W), np.shape(C_south.W))]):
            raise Exception("The curves(C_north,C_south) must have conforming arrays of weights")

        if any([size1 != size2 for size1, size2 in zip(np.shape(C_west.W), np.shape(C_east.W))]):
            raise Exception("The curves (C_west,C_east) must have conforming arrays of weights")


        # Check the curve degrees
        if C_north.p != C_south.p:
            raise Exception("The curves(C_north,C_south) must have the same degree")

        if C_west.p != C_east.p:
            raise Exception("The curves(C_west,C_east) must have the same degree")


        # Check the knot vectors
        if np.shape(C_north.U) != np.shape(C_south.U):
            raise Exception("The curves (C_north,C_south) must have the same number of knots")

        if np.shape(C_west.U) != np.shape(C_east.U):
            raise Exception("The curves (C_west,C_east) must have the same number of knots")

        if any([np.abs(u1-u2)>tol for u1, u2 in zip(C_north.U, C_south.U)]):
            raise Exception("The curves (C_north,C_south) must have the same number of knot values")

        if any([np.abs(u1-u2)>tol for u1, u2 in zip(C_west.U, C_east.U)]):
            raise Exception("The curves (C_west,C_east) have the same number of knot values")


        # Check corner control point compatibility
        if np.all(np.abs(self.C_south.P[:, 0] - self.C_west.P[:, 0]) < tol):
            self.P_sw = C_south.P[:, 0]
        else:
            raise Exception("The sourth-west corner is not compatible")

        if np.all(np.abs(self.C_south.P[:, -1] - self.C_east.P[:, 0]) < tol):
            self.P_se = C_south.P[:, -1]
        else:
            raise Exception("The sourth-east corner is not compatible")

        if np.all(np.abs(self.C_north.P[:, -1] - self.C_east.P[:, -1]) < tol):
            self.P_ne = C_north.P[:, -1]
        else:
            raise Exception("The north-east corner is not compatible")

        if np.all(np.abs(self.C_north.P[:, 0] - self.C_west.P[:, -1]) < tol):
            self.P_nw = C_north.P[:, 0]
        else:
            raise Exception("The north-west corner is not compatible")


        # Check corner weight compatibility
        if np.abs(self.C_south.W[0] - self.C_west.W[0]) < tol:
            self.W_sw = C_south.W[0]
        else:
            raise Exception("The sourth-west weight is not compatible")

        if np.abs(self.C_south.W[-1] - self.C_east.W[0]) < tol:
            self.W_se = C_south.W[-1]
        else:
            raise Exception("The sourth-east weight is not compatible")

        if np.abs(self.C_north.W[-1] - self.C_east.W[-1]) < tol:
            self.W_ne = C_north.W[-1]
        else:
            raise Exception("The north-east weight is not compatible")

        if np.abs(self.C_north.W[0] - self.C_west.W[-1]) < tol:
            self.W_nw = C_north.W[0]
        else:
            raise Exception("The north-west weight is not compatible")


        # Make the Coons surface NURBS representation
        self.NurbsSurface = None
        self.make_nurbs_surface()



    def make_nurbs_surface(self):

        """ Make a NURBS surface representation of the Coons surface """

        # Size of the array of control points (use broadcasting to get the right shapes)
        ndim, Nu, Nv = 3, np.shape(self.P_south)[1], np.shape(self.P_west)[1]

        # Map the boundary control points to homogeneous space and set the right number of dimensions for broadcasting
        Pw_north = np.concatenate((self.P_north*self.W_north[np.newaxis, :], self.W_north[np.newaxis, :]), axis=0)[:, :, np.newaxis]
        Pw_south = np.concatenate((self.P_south*self.W_south[np.newaxis, :], self.W_south[np.newaxis, :]), axis=0)[:, :, np.newaxis]
        Pw_west  = np.concatenate((self.P_west *self.W_west [np.newaxis, :], self.W_west [np.newaxis, :]), axis=0)[:, np.newaxis, :]
        Pw_east  = np.concatenate((self.P_east *self.W_east [np.newaxis, :], self.W_east [np.newaxis, :]), axis=0)[:, np.newaxis, :]

        # Map the corner control points to homogeneous space and set the right number of dimensions for broadcasting
        Pw_sw = np.concatenate((self.P_sw*self.W_sw[np.newaxis], self.W_sw[np.newaxis]), axis=0)[:, np.newaxis, np.newaxis]
        Pw_se = np.concatenate((self.P_se*self.W_se[np.newaxis], self.W_se[np.newaxis]), axis=0)[:, np.newaxis, np.newaxis]
        Pw_ne = np.concatenate((self.P_ne*self.W_ne[np.newaxis], self.W_ne[np.newaxis]), axis=0)[:, np.newaxis, np.newaxis]
        Pw_nw = np.concatenate((self.P_nw*self.W_nw[np.newaxis], self.W_nw[np.newaxis]), axis=0)[:, np.newaxis, np.newaxis]

        # Compute the array of control points by transfinite interpolation
        u = np.linspace(0, 1, Nu)[np.newaxis, :, np.newaxis]
        v = np.linspace(0, 1, Nv)[np.newaxis, np.newaxis, :]
        term_1a = (1 - v) * Pw_south + v * Pw_north
        term_1b = (1 - u) * Pw_west + u * Pw_east
        term_2 = (1 - u) * (1 - v) * Pw_sw + u * v * Pw_ne + (1 - u) * v * Pw_nw + u * (1 - v) * Pw_se
        Pw = term_1a + term_1b - term_2

        # Compute the array of control points in ordinary space and the array of control point weights (inverse map)
        P = Pw[0:-1, :, :]/Pw[[-1], :, :]
        W = Pw[-1, :, :]

        # Define the order of the basis polynomials
        p = self.C_south.p
        q = self.C_west.p

        # Definite the knot vectors
        U = self.C_south.U
        V = self.C_west.U

        # Create the NURBS surface
        self.NurbsSurface = NurbsSurface(control_points=P, weights=W, u_degree=p, v_degree=q, u_knots=U, v_knots=V)

