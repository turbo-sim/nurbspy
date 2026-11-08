"""View or edit a NURBS curve; change the global settings below."""

from pathlib import Path
import numpy as np
import nurbspy as nrb

EDITABLE = True  # True allows point/weight editing; False only views the curve.
OUTPUT = Path(__file__).resolve().parent / "output" / Path(__file__).stem

# A sequence of Points avoids array-orientation ambiguity.
points = [
    nrb.Point(-2.0, 0.0, 0.0),
    nrb.Point(-1.0, 1.8, 0.5),
    nrb.Point(0.0, -1.0, 1.2),
    nrb.Point(1.0, 1.5, 0.4),
    nrb.Point(2.0, 0.0, 0.0),
]
# Larger positive weights pull the curve towards the corresponding control point.
# degree=3 chooses cubic basis functions; it is independent of drawing resolution.
curve = nrb.NurbsCurve(points, weights=np.array([1, 1, 0.6, 1.8, 1]), degree=3)

# Change these settings here, or use the Display menu in the viewer.
DISPLAY = dict(
    view="3d",  # Perspective view; "xy", "yz", "zx" look straight at those planes.
    quantity="solid",  # One colour; use "curvature" to colour by curve curvature.
    colormap="viridis",  # Colour palette for quantities; unused with "solid".
    frames=False,  # Show tangent, normal, and binormal direction vectors if True.
    frame_count=7,  # Number of vector frames along the curve, when frames=True.
    frame_scale=0.08,  # Vector length as a fraction of curve length: 0.08 = 8%.
)


def main():
    """Open the selected mode and return accepted edits, if any."""
    if not EDITABLE:
        curve.show(title="NURBS curve viewer", display=DISPLAY)
        return None
    edited = curve.edit(title="NURBS curve editor", display=DISPLAY)
    if edited is not None:
        print("Edited control points:\n", edited.P.T)
        print("Edited weights:\n", edited.W)
        # To save accepted edits beside this demo, uncomment:
        # OUTPUT.mkdir(parents=True, exist_ok=True)
        # edited.to_yaml(OUTPUT / "edited_curve.yaml")
    return edited


if __name__ == "__main__":
    main()
