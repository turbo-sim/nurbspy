"""Compatibility imports for the shared adapter to existing NurbsPy projection."""

from nurbspy.fitting._projection import SurfaceProjector


def project_points(model, points, *, tolerance=1e-9):
    """Return checked local distances using the installed projection adapter."""
    import jax

    with jax.enable_x64():
        projector = SurfaceProjector(model, points, tolerance=tolerance)
        return projector.project(model)
