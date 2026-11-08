# Manual viewer/editor checks

Run from the `nurbspy` repository root. Install the standard dependencies:

```powershell
poetry install
```

If you also use JAX or Numba, include their extras in the install command.

## Desktop

```powershell
poetry run python demos/demos_viewer/demo_curve_view.py
poetry run python demos/demos_viewer/demo_surface_view.py
poetry run python demos/demos_viewer/demo_curve_edit.py
poetry run python demos/demos_viewer/demo_surface_edit.py
```

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
- [ ] Reset restores the starting geometry. Apply returns edited coordinates/weights to the script; Cancel/window close returns no edited geometry. The original object stays unchanged. Undo/redo is intentionally absent.

## Embedded applications

```powershell
poetry run python demos/demos_viewer/demo_embedded_curve.py
poetry run python demos/demos_viewer/demo_embedded_surface.py
poetry run python demos/demos_viewer/demo_embedded_surface_view.py
```

- [ ] Open ports 8061, 8062, and 8063 respectively. Repeat the main navigation/display/edit checks; appearance matches desktop. Port 8063 is read-only.
- [ ] Separate tabs maintain independent geometry. Resize the page; the viewport and panes remain usable. No application-analysis readout appears.
- [ ] Stop each server with Ctrl+C. Closing desktop windows ends their calls cleanly.

## Your normal workflow

- [ ] Run one of your own Point-based curves/surfaces through `show()` and `edit()`. Repeat with a JAX input if you use that backend; accepted edits currently return NumPy geometry.
- [ ] Save an accepted edit with `to_yaml()`, load it with the desired backend's `from_yaml()`, and confirm geometry/weights agree.
- [ ] Run an existing Matplotlib `.plot()` or diagnostic script to check compatibility. Try a representative larger model to judge editing responsiveness.

Record any issue with the script, view/navigation mode, reproduction steps, and expected result.
