# Manual viewer/editor checks

Run from the `nurbspy` repository root. Install the standard dependencies:

```powershell
poetry install
```

If you also use JAX or Numba, include their extras in the install command.

## Desktop

```powershell
poetry run python -m demos.03_viewer.01_curve
poetry run python -m demos.03_viewer.02_surface
```

Run each script with `EDITABLE = True` and `EDITABLE = False` to cover editor
and read-only modes. Change the global `RESOLUTION` and `DISPLAY` settings as needed.

- [ ] Windows open without launching a browser or server. View-only windows have no Edit or Reset.
- [ ] Geometry starts in solid colour. Points/net are orange; editors highlight the selected point in green, while viewers retain its normal colour and size. Points remain a sensible, similar size in 3D and planar views while zooming.
- [ ] Arcball, Orbit, and Trackball feel correct. Switching modes preserves the camera. Fit and ViewCube faces/edges/corners, Home, and ISO behave as expected.
- [ ] XY, YZ, and ZX views pan/zoom correctly; optional ticks follow the view. The grid stays behind geometry and points.
- [ ] Display/Edit buttons open and close the right pane. Switching panes retains settings. Escape closes an open pane; a narrow window keeps controls accessible.
- [ ] Select points by clicking and by dropdown. Drag with view-plane, coordinate-plane, and axis constraints; verify only the permitted coordinates change. Planar dragging preserves the hidden coordinate.
- [ ] Enter X/Y/Z coordinates with Enter and by leaving the field. Try blank input. Change weights with the slider and numeric field; invalid values are rejected.
- [ ] In 3D, choose Normal direction, wait for readiness, and drag. The direction stays fixed during the drag and refreshes afterward. Undefined normals produce a useful message.
- [ ] Try colour pickers, scalar quantities, colormaps, limits, and curve Frenet frames. Return to solid colour; the geometry and camera remain usable.
- [ ] On a surface, try Display mesh and STL export mesh. Download an STL after editing and open it in your usual mesh viewer/slicer. Check shape, scale, and sampling; open patches need not be closed solids.
- [ ] Shift-click adds/removes points; Shift-drag adds a rectangle selection without moving the camera. Box select replaces the selection. Dragging a selected point translates the group with its relative positions intact. A background click clears selection, while background dragging navigates.
- [ ] Coordinates and weight edit only the active point. Selection count and green highlights agree, including when selecting no points or using planar views.
- [ ] Undo/Redo buttons and Ctrl+Z/Ctrl+Shift+Z (Command on macOS) reverse one completed drag, slider gesture, or numeric edit per step. Reset is undoable. A new edit after Undo clears Redo. History survives object switches; camera and selection changes do not add history. Input fields retain native text undo. Escape cancels a drag/rectangle before closing the pane.
- [ ] Reset restores the starting geometry. Apply returns edited coordinates/weights to the script; Cancel/window close returns no edited geometry. The original object stays unchanged.

## Embedded applications

```powershell
poetry run python -m demos.03_viewer.03_embedded
```

- [ ] Open port 8061 by default. Test both `GEOMETRY` settings (`"curve"`, `"surface"`) with both `EDITABLE` settings in this script. Repeat the main navigation/display/edit checks; appearance matches desktop. The global `PORT` selects the server port.
- [ ] Separate tabs maintain independent geometry. Resize the page; the viewport and panes remain usable. No application-analysis readout appears.
- [ ] Stop each server with Ctrl+C. Closing desktop windows ends their calls cleanly.

## Your normal workflow

- [ ] Run one of your own Point-based curves/surfaces through `show()` and `edit()`. Repeat with a JAX input if you use that backend; accepted edits currently return NumPy geometry.
- [ ] Save an accepted edit with `to_yaml()`, load it with the desired backend's `from_yaml()`, and confirm geometry/weights agree.
- [ ] Run an existing Matplotlib `.plot()` or diagnostic script to check compatibility. Try a representative larger model to judge editing responsiveness.

Record any issue with the script, view/navigation mode, reproduction steps, and expected result.
