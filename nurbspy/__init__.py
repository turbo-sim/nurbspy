# Import geometry modules
from .nurbs_basis_functions    import *
from .nurbs_curve              import *
from .nurbs_curve_circular_arc import *
from .nurbs_surface            import *
from .nurbs_surface_bilinear   import *
from .nurbs_surface_ruled      import *
from .nurbs_surface_extruded   import *
from .nurbs_surface_revolution import *
from .nurbs_surface_coons      import *

# Import minimum working example
from .minimal_example          import *



# Package info
__version__ = "1.1.5"
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


