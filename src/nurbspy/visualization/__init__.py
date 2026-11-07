"""Optional desktop visualization and embeddable geometry editing.

Importing this module does not import pywebview, Dash, or Flask. Frameworks are
loaded only when opening a desktop window or requesting a Dash layout.
"""

from .dash import GeometryView, GeometryEditor
from .desktop import edit


def plot(geometry, *args, **kwargs):
    """Return a Matplotlib figure and axes without opening a desktop editor."""
    from .plotting import plot as render

    return render(geometry, *args, **kwargs)


def show(geometry, *, backend="interactive", **kwargs):
    """Inspect geometry; use backend=matplotlib for a static plotting window."""
    if backend == "interactive":
        from .desktop import show as render

        return render(geometry, **kwargs)
    if backend == "matplotlib":
        import matplotlib.pyplot as plt

        plot(geometry, **kwargs)
        plt.show()
        return None
    raise ValueError("backend must be interactive or matplotlib")


__all__ = ["GeometryView", "GeometryEditor", "edit", "show", "plot"]
