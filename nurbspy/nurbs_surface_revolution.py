# -------------------------------------------------------------------------------------------------------------------- #
# Import packages
# -------------------------------------------------------------------------------------------------------------------- #
import numpy as np

from .nurbs_surface import NurbsSurface


# -------------------------------------------------------------------------------------------------------------------- #
# Define the bilinear NURBS surface class
# -------------------------------------------------------------------------------------------------------------------- #
class NurbsSurfaceRevolution:

    """ Create the NURBS surface obtained by revolving a generatrix NURBS curve `C(u)` about an axis

    Parameters
    ----------
    generatrix : NURBS curve object
        See NurbsSurface class documentation

    axis_point : ndaray of with shape (3,)
        Point that together with a direction defines the axis of rotation

    axis_direction : ndarrays with shape (3,)
        Direction that together with a point defines the axis of rotation

    theta_start : scalar
        Start angle, measured with respect to the generatrix

    theta_end : scalar
        End angle, measured with respect to the generatrix

    References
    ----------
    The NURBS book. Chapter 8.5
    L. Piegl and W. Tiller
    Springer, second edition

    """

    def __init__(self, generatrix, axis_point, axis_direction, angle_start, angle_end):

        # Set the data type used to initialize arrays (set `complex` if any argument is complex and `float` if not)
        temp = generatrix.P
        for item in locals().values():
            data_type = np.asarray(item).dtype
            if np.issubdtype(data_type, np.complex128):
                self.data_type = np.complex128
                break
            else:
                self.data_type = np.float64

        # Declare input variables as instance variables (adopt the notation used in the NURBS book)
        self.C = generatrix
        self.S = axis_point
        self.T = axis_direction/np.sum(axis_direction**2)**(1/2)
        self.theta_start = angle_start
        self.theta_end = angle_end

        # Check the number of dimensions of the problem
        if np.shape(generatrix.P)[0] != 3:   raise Exception("The input NURBS must be three-dimensional")
        if np.shape(axis_direction)[0] != 3: raise Exception("The axis direction must be a three-dimensional vector")
        if np.shape(axis_point)[0] != 3:     raise Exception("The axis point must be a three-dimensional vector")

        # Make the extrusion surface NURBS representation
        self.NurbsSurface = None
        self.make_nurbs_surface()


    def make_nurbs_surface(self):

        """ Make a NURBS surface representation of the revolution surface """

        # Rename variables for brevity
        S = self.S
        T = self.T
        theta_start = self.theta_start
        theta_end = self.theta_end

        # Correct theta_end if necessary
        if theta_end < theta_start: theta_end = theta_end + 2*np.pi

        # Angle spanned by the circular arc
        theta = theta_end - theta_start

        # Get the number of NURBS segments used to represent the circular arc
        if    theta <= 1/2*np.pi: n_arcs = 1
        elif  theta <= 2/2*np.pi: n_arcs = 2
        elif  theta <= 3/2*np.pi: n_arcs = 3
        elif  theta <= 4/2*np.pi: n_arcs = 4
        else: raise Exception('Opps, something went wrong...')

        # Angle spanned by each segment
        delta_theta = theta/n_arcs

        # Initialize arrays of control points and weights
        n = 2*n_arcs                                                  # Highest index of the u-direction control points
        m = np.shape(self.C.P)[1] - 1                                 # Highest index of the v_direction control points
        P_array = np.zeros((3, n + 1, m + 1), dtype=self.data_type)   # Array of control points
        W_array = np.zeros((n + 1, m + 1), dtype=self.data_type)      # Weight of the control points

        # Loop over the generatrix control points
        for j in range(m+1):

            # Current control point of the generatrix
            P = self.C.P[:, j]

            # Current weight of the generatrix
            W = self.C.W[j]

            # Compute the axis point that is closest to the control point
            O = self.project_point_to_line(S, T, P)

            # Compute the projected distance between the current control point and the axis
            R = np.sum((P-O)**2)**(1/2)

            # Compute the current section principal directions (use dummy directions in case P and O coincide)
            if np.abs(np.sum(P-O)) < 1e-12:
                X = np.asarray([1, 0, 0])
                Y = np.asarray([0, 1, 0])
            else:
                X = (P-O)/np.sum((P-O)**2)**(1/2)
                Y = np.cross(T, X)

            # Get the coordinates and weight of the first control point
            P0 = O + R * np.cos(theta_start) * X + R * np.sin(theta_start) * Y
            T0 = -np.sin(theta_start) * X + np.cos(theta_start) * Y

            # Store control points and weights
            P_array[:, 0, j] = P0
            W_array[0, j] = W

            # Get the coordinates and weights of the other control points
            index, angle = 0, theta_start
            for i in range(n_arcs):

                # Angle spanned by the current segment
                angle = angle + delta_theta

                # Get the end-point and end-tangent of the current segment
                P2 = O + R * np.cos(angle) * X + R * np.sin(angle) * Y
                T2 = -np.sin(angle) * X + np.cos(angle) * Y

                # Solve the intersection between tangent lines T0 and T2 to compute the intermediate point
                P1 = self.intersect_lines(P0, T0, P2, T2)

                # Compute the weight of the intermediate point
                W1 = np.cos(delta_theta / 2)

                # Store control points and weights
                P_array[:, index + 1, j] = P1
                P_array[:, index + 2, j] = P2
                W_array[index + 1, j] = W*W1
                W_array[index + 2, j] = W

                # Get ready for the next segment!
                index = index + 2
                P0 = P2
                T0 = T2

        # Define the order of the basis polynomials
        p = 2
        q = self.C.p

        # Define the knot vectors
        # Knot multiplicity p+1 at the endpoints
        U = 5 + np.zeros((n + p + 2))
        U[[0, 1, 2]] = 0
        U[[-1, -2, -3]] = 1

        # Set the multiplicity 2 at the interior knots to connect the segments
        if n_arcs == 1:
            pass
        elif n_arcs == 2:
            U[[3, 4]] = 1 / 2
        elif n_arcs == 3:
            U[[3, 4]], U[[5, 6]] = 1 / 3, 2 / 3
        elif n_arcs == 4:
            U[[3, 4]], U[[5, 6]], U[[7, 8]] = 1 / 4, 2 / 4, 3 / 4
        else:
            raise Exception('Opps, something went wrong...')

        # The knot vector in the v-direction is given by the knot vector of the generatrix NURBS
        V = self.C.U

        # Create the NURBS surface
        self.NurbsSurface = NurbsSurface(control_points=P_array, weights=W_array, u_degree=p, v_degree=q, u_knots=U, v_knots=V)


    def intersect_lines(self, P0, T0, P2, T2):

        """ Compute the point of intersection between two lines in 2D or 3D """

        # Compute the intersection by reducing the 3x2 system to a 2x2 system using dot products
        A = np.asarray([[np.sum(T0 * T0), -np.sum(T2 * T0)], [np.sum(T0 * T2), -np.sum(T2 * T2)]])
        b = np.asarray([np.sum(P2 * T0) - np.sum(P0 * T0), np.sum(P2 * T2) - np.sum(P0 * T2)])
        u, v = np.linalg.solve(A, b)
        P1 = P0 + u * T0

        if np.sum(np.abs(((P0 + u * T0) - (P2 + v * T2)))) > 1e-12:
            raise Exception("Something went wrong computing the line intersection")

        return P1


    def project_point_to_line(self, S, T, P):

        """ Compute the projection of a point `P` into the line given by `S + u*T` """

        # Analytic formula (not hard to derive by hand)
        P_projected = S + np.sum(T * (P - S)) / np.sum(T * T) * T

        return P_projected