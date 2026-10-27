"""Preserve public module paths from the original NumPy-only package layout."""

from importlib import import_module
import sys


def register_numpy_module_aliases():
    """Bind legacy imports to the canonical NumPy modules without copying them.

    Both ``sys.modules`` entries and parent-package attributes are needed for
    direct module imports, ``from`` imports, and older pickle references to
    resolve to the same objects. Keep this finite list explicit: new modules
    belong in their backend and do not automatically become legacy imports.

    New pickles use the canonical ``nurbspy.numpy`` paths; compatibility with
    older releases that lack that subpackage is not guaranteed.
    """
    package = sys.modules[__package__]
    module_names = (
        "nurbs_basis_functions",
        "nurbs_curve",
        "nurbs_curve_circular_arc",
        "nurbs_surface",
        "nurbs_surface_bilinear",
        "nurbs_surface_ruled",
        "nurbs_surface_extruded",
        "nurbs_surface_revolution",
        "nurbs_surface_coons",
    )
    for name in module_names:
        module = import_module(f".numpy.{name}", package=__package__)
        sys.modules[f"{__package__}.{name}"] = module
        setattr(package, name, module)
