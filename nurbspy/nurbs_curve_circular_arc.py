# -------------------------------------------------------------------------------------------------------------------- #
# Import packages
# -------------------------------------------------------------------------------------------------------------------- #
import numpy as np

from .nurbs_curve import NurbsCurve


# -------------------------------------------------------------------------------------------------------------------- #
# Define the NURBS curve class
# -------------------------------------------------------------------------------------------------------------------- #
class CircularArc:

    """ Create a NURBS representation of a circular arc

        Create a NURBS representation of the circular arc parametrized by
            C(u) = O + R*cos(u)*X + R*sin(u)*Y
            theta_start <= u <= theta_end

    Parameters
    ----------

    O : ndarray with shape (ndim,)
        Center of the circular arc. ndim=2 or ndim=3

    X : ndarray with shape (ndim,)
        Unit vector lying in the plane of the circle

    Y : ndarray with shape (ndim,)
        Unit vector lying in the plane of the circle and orthogonal to X

    R : scalar
        Radius of the circle

    theta_start : scalar
        Start angle, measured with respect to X

    theta_end : scalar
        End angle, measured with respect to X

    References
    ----------
    The NURBS book. Chapter 7.5
    L. Piegl and W. Tiller
    Springer, second edition

    """

    def __init__(self, O, X, Y, R, theta_start, theta_end):

        # Set the data type used to initialize arrays (set `complex` if any argument is complex and `float` if not)
        for item in locals().values():
            data_type = np.asarray(item).dtype
            if np.issubdtype(data_type, np.complex128):
                self.data_type = np.complex128
                break
            else:
                self.data_type = np.float64

        # Declare input variables as instance variables
        self.O = O
        self.X = X/np.sum(X**2)**(1/2)
        self.Y = Y/np.sum(Y**2)**(1/2)
        self.R = R
        self.theta_start = theta_start
        self.theta_end = theta_end

        # Check the number of dimensions of the problem
        self.ndim = np.shape(O)[0]
        if not (self.ndim == 2 or self.ndim == 3):
            raise Exception("The number of dimensions of the problem must be 2 or 3")

        # Make the circular arc NURBS representation
        self.NurbsCurve = None
        self.make_nurbs_circular_arc()


    def make_nurbs_circular_arc(self):

        # Rename variables for brevity
        O = self.O
        X = self.X
        Y = self.Y
        R = self.R
        theta_start = self.theta_start
        theta_end = self.theta_end
        ndim = self.ndim

        # Correct theta_end if necessary
        if theta_end < theta_start: theta_end = theta_end + 2*np.pi

        # Angle spanned by the circular arc
        theta = theta_end - theta_start

        # Get the number of NURBS segments used to represent the circular arc
        if    theta <= 1/2*np.pi: n_arcs = 1
        elif  theta <= 2/2*np.pi: n_arcs = 2
        elif  theta <= 3/2*np.pi: n_arcs = 3
        elif  theta <= 4/2*np.pi: n_arcs = 4
        else: raise Exception('Ooops, something went wrong...')

        # Angle spanned by each segment
        delta_theta = theta/n_arcs

        # Initialize arrays of control points and weights
        n = 2*n_arcs                                        # Highest index of the control points (counting from zero)
        P = np.zeros((ndim, n + 1), dtype=self.data_type)   # Array of control points
        W = np.zeros((n + 1), dtype=self.data_type)         # Weight of the control points

        # Get the coordinates and weight of the first control point
        P0 = O + R * np.cos(theta_start) * X + R * np.sin(theta_start) * Y
        T0 = -np.sin(theta_start) * X + np.cos(theta_start) * Y
        W0 = 1.00
        P[:, 0] = P0
        W[0] = W0

        # Get the coordinates and weights of the other control points
        index, angle = 0, theta_start
        for i in range(n_arcs):

            # Angle spanned by the current segment
            angle = angle + delta_theta

            # End point of the current segment
            P2 = O + R * np.cos(angle) * X + R * np.sin(angle) * Y
            T2 = -np.sin(angle) * X + np.cos(angle) * Y

            # Solve the intersection between tangent lines T0 and T2 to compute the intermediate point
            P1 = self.intersect_lines(P0, T0, P2, T2)
            W1 = np.cos(delta_theta/2)

            # Store control points and weights
            P[:, index+1] = P1
            P[:, index+2] = P2
            W[index+1] = W1
            W[index+2] = 1

            # Get ready for the next segment!
            index = index + 2
            P0 = P2
            T0 = T2

        # Define the order of the basis polynomials
        # Linear (p = 1), Quadratic (p = 2), Cubic (p = 3), etc.
        # Set p = n (number of control points minus one) to obtain a Bezier
        p = 2

        # Knot multiplicity p+1 at the endpoints
        U = 5 + np.zeros((n + p + 2))
        U[[0, 1, 2]] = 0
        U[[-1, -2, -3]] = 1

        # Set the multiplicity of the interior knots to 2 to connect the segments
        if   n_arcs == 1: pass
        elif n_arcs == 2: U[[3, 4]] = 1/2
        elif n_arcs == 3: U[[3, 4]], U[[5, 6]]  = 1/3, 2/3
        elif n_arcs == 4: U[[3, 4]], U[[5, 6]], U[[7, 8]] = 1/4, 2/4, 3/4
        else: raise Exception('Ooops, something went wrong...')

        # Create the NURBS curve
        self.P = P
        self.W = W
        self.p = p
        self.U = U
        self.NurbsCurve = NurbsCurve(control_points=P, weights=W, degree=p, knots=U)


    def intersect_lines(self, P0, T0, P2, T2):

        # Compute the intersection after reducing the 3x2 system to a 2x2 system using the dot product
        A = np.asarray([[np.sum(T0 * T0), -np.sum(T0 * T2)], [np.sum(T0 * T2), -np.sum(T2 * T2)]])
        b = np.asarray([np.sum(P2 * T0) - np.sum(P0 * T0), np.sum(P2 * T2) - np.sum(P0 * T2)])
        u, v = np.linalg.solve(A, b)
        P1 = P0 + u * T0

        if np.sum(np.abs(((P0 + u * T0) - (P2 + v * T2)))) > 1e-12:
            raise Exception("Something went wrong computing the line intersection")

        return P1


    def plot(self):

        # Plot the NURBS curve
        fig, ax = self.NurbsCurve.plot()

        # Plot additional details (full circle, circle center and start/end points)
        if self.ndim == 2:
            u = np.linspace(0, 2 * np.pi, 100)
            P_start = np.real(self.NurbsCurve.P[:, 0])
            P_end = np.real(self.NurbsCurve.P[:, -1])
            O = np.real(self.O[:, np.newaxis])
            X = np.real(self.X[:, np.newaxis])
            Y = np.real(self.Y[:, np.newaxis])
            R = np.real(self.R)
            x, y = O + R * np.cos(u) * X + R * np.sin(u) * Y
            ax.plot(x, y, 'b')
            ax.plot([0, self.O[0]], [0, self.O[1]], 'ko-', markersize=3.5, markerfacecolor='w')
            ax.plot([self.O[0], P_start[0]], [self.O[1], P_start[1]], 'ko-', markersize=3.5, markerfacecolor='w')
            ax.plot([self.O[0], P_end[0]], [self.O[1], P_end[1]], 'ko-', markersize=3.5, markerfacecolor='w')

        elif self.ndim == 3:

            # Plot the full circle
            u = np.linspace(0, 2*np.pi, 100)
            P_start = np.real(self.NurbsCurve.P[:, 0])
            P_end = np.real(self.NurbsCurve.P[:, -1])
            O = np.real(self.O[:, np.newaxis])
            X = np.real(self.X[:, np.newaxis])
            Y = np.real(self.Y[:, np.newaxis])
            R = np.real(self.R)
            x, y, z =  O + R*np.cos(u)*X + R*np.sin(u)*Y
            ax.plot(x, y, z, 'b')
            ax.plot([0, self.O[0]], [0, self.O[1]], [0, self.O[2]], 'ko-', markersize=3.5, markerfacecolor='w')
            ax.plot([self.O[0], P_start[0]], [self.O[1], P_start[1]], [self.O[2], P_start[2]], 'ko-', markersize=3.5, markerfacecolor='w')
            ax.plot([self.O[0], P_end[0]], [self.O[1], P_end[1]], [self.O[2], P_end[2]], 'ko-', markersize=3.5, markerfacecolor='w')

        else:
            raise Exception('The number of dimensions must be 2 or 3')


        # Rescale the plot
        self.NurbsCurve.rescale_plot(fig, ax)

        return fig, ax







