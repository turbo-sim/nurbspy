"""NURBS geometry with NumPy defaults and an optional explicit JAX backend.

Existing root-level imports retain the NumPy implementation. Use
``nurbspy.numpy`` or ``nurbspy.jax`` to select a backend explicitly.
"""

# Preserve the historical exports, including shared graphics helpers.
from .numpy import *
from .graphics import *

from ._backward_compatibility import register_numpy_module_aliases as _register_numpy_module_aliases

_register_numpy_module_aliases()
del _register_numpy_module_aliases

# Package info
__version__ = "1.6.0"
PACKAGE_NAME = "nurbspy"
URL_GITHUB = "https://github.com/turbo-sim/nurbspy"
URL_DOCS = "https://turbo-sim.github.io/nurbspy/"
# URL_DTU = "https://thermalpower.dtu.dk/"
BREAKLINE = 80 * "-"


def print_banner():
    """Prints a banner."""
    banner = r"""
        _   __           __                    
       / | / /_  _______/ /_  _________  __  __
      /  |/ / / / / ___/ __ \/ ___/ __ \/ / / /
     / /|  / /_/ / /  / /_/ (__  ) /_/ / /_/ / 
    /_/ |_/\__,_/_/  /_.___/____/ .___/\__, /  
                               /_/    /____/   
    """
    print(BREAKLINE)
    print(banner)
    print(BREAKLINE)
    # https://manytools.org/hacker-tools/ascii-banner/
    # Style: Slant


def print_package_info():
    """Prints package information with predefined values."""

    info = f""" Version:       {__version__}
 Repository:    {URL_GITHUB}
 Documentation: {URL_DOCS}"""
    print_banner()
    print(BREAKLINE)
    print(info)
    print(BREAKLINE)

# NURBS curve minimal working example
def minimal_example():

    # Import packages
    import numpy as np
    import matplotlib.pyplot as plt
    from .numpy.nurbs_curve import NurbsCurve
    print_package_info()

    # Define the array of control points
    P = np.zeros((2,5))
    P[:, 0] = [0.20, 0.50]
    P[:, 1] = [0.40, 0.70]
    P[:, 2] = [0.80, 0.60]
    P[:, 3] = [0.80, 0.40]
    P[:, 4] = [0.40, 0.20]

    # Create and plot the Bezier curve
    bezierCurve = NurbsCurve(control_points=P)
    bezierCurve.plot()
    plt.show()


# print_package_info()
# set_plot_options()
