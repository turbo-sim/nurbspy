"""View or edit a NURBS surface; change the global settings below."""

from pathlib import Path
import numpy as np
import nurbspy as nrb

EDITABLE = True  # True allows point/weight editing; False only views the surface.
OUTPUT = Path(__file__).resolve().parent / "output" / Path(__file__).stem

# A weighted bicubic saddle. Set z to zero to try a planar surface.
x, y = np.meshgrid(np.linspace(-1.5, 1.5, 4), np.linspace(-1.5, 1.5, 4), indexing="ij")
z = 0.45 * (x * x - y * y)
weights = np.ones((4, 4))
weights[1:3, 1:3] = 1.8
points = [[nrb.Point(x[i, j], y[i, j], z[i, j]) for j in range(4)] for i in range(4)]
surface = nrb.NurbsSurface(points, weights=weights)

# Enable a mesh in Display, or set mesh="display" or mesh="export" explicitly.
DISPLAY = dict(
    view="3d",  # Perspective view; "xy", "yz", "zx" look straight at those planes.
    quantity="solid",  # One colour; "mean_curvature" colours by surface curvature.
    colormap="coolwarm",  # Colour palette for quantities; unused with "solid".
    mesh="off",  # Hide mesh lines; "display" shows triangles, "export" previews STL.
    mesh_u=40,  # Requested STL samples along the first surface direction (u).
    mesh_v=40,  # Requested STL samples along the second surface direction (v).
)


def main():
    """Open the selected mode and return accepted edits, if any."""
    if not EDITABLE:
        surface.show(title="NURBS surface viewer", display=DISPLAY)
        return None
    edited = surface.edit(title="NURBS surface editor", display=DISPLAY)
    if edited is not None:
        print("Edited control points:\n", edited.P.T)
        print("Edited weights:\n", edited.W)
        # To save accepted edits beside this demo, uncomment:
        # OUTPUT.mkdir(parents=True, exist_ok=True)
        # edited.to_yaml(OUTPUT / "edited_surface.yaml")
    return edited


if __name__ == "__main__":
    main()
