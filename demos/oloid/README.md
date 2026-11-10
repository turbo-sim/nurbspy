# Oloid construction animation

Run with the installed nurbspy package:

```powershell
poetry run python demos/oloid/demo_oloid_animation.py
```

The visualizer opens first with the complete oloid. Close the window to start
the PNG and GIF export. Set `EXPORT_ANIMATION = False` near the top of the script
to use only the visualizer. `SHOW_VISUALIZER = False` permits unattended export.

The default loop is 40 seconds at 30 fps. Three GIFs share the same surface motion:

- `output/oloid_overview.gif`: three continuous camera revolutions.
- `output/oloid_following.gif`: a camera following the generating line.
- `output/oloid.gif`: both views side by side, overview on the left.

Each of the two development cycles has the following timing. The second cycle
starts at 20 seconds:

| Time within cycle | Surface motion | Duration |
| --- | --- | ---: |
| 0-1 s | Pause at the line-only state | 1 s |
| 1-10 s | Develop | 9 s |
| 10-11 s | Pause at the fully formed oloid | 1 s |
| 11-20 s | Undevelop | 9 s |

Within each motion interval, normalized time $x\in[0,1]$ gives the revealed
fraction $d(x)=(1-\cos(\pi x))/2$ for development, or
$d(x)=(1+\cos(\pi x))/2$ for undevelopment. The overview camera rotates at
constant angular speed throughout the loop, including during the pauses. Its
distance, height and target stay fixed. The second development is seen from the
opposite side.

The following camera targets the generating line's midpoint. It stays on the
outward side of the local surface, perpendicular to the line with a 15-degree
oblique offset towards the already generated surface. Its fixed distance leaves
the whole oloid in frame. Its direction comes from geometry, rather than time
derivatives, so stopping or reversing the line never flips the camera. This view
pauses with the line and hides the floor grid to keep the construction clear.

Cosine easing gives zero surface speed at both the line-only and fully formed
states. Both cameras and the geometry return to their starting states at the
loop join. The exporter writes 1201 PNGs per view and 1201 combined PNGs. The
last PNG is a closing sample for checking the loop; GIFs exclude that duplicate,
preserving the configured hold duration and total playback time of 40 seconds.
GIF encoders may merge identical adjacent samples while preserving their timing.

Global settings control `LOOP_SECONDS`, `DEVELOPMENT_CYCLES`, `CAMERA_ORBITS`,
`HOLD_SECONDS`, `FOLLOW_DISTANCE`, `FOLLOW_ANGLE_DEGREES`, `FIT_TOLERANCE`,
rendering resolution and image dimensions. `LOOP_SECONDS` includes all holds.
Each individual PNG defaults to 1000×750 and GIF to 800×600. Combined PNGs are
2000×750 and the combined GIF is 1600×600.
Each exported PNG reports phase, percentage, elapsed time and ETA. GIF encoding
and frame composition also report progress. One browser is reused for each view.

Outputs are independent of the current working directory:

- `output/overview/frames/frame_00000.png`, …
- `output/following/frames/frame_00000.png`, …
- `output/frames/frame_00000.png`, … (combined)
- `output/oloid_overview.gif`, `output/oloid_following.gif`, `output/oloid.gif`
- `output/loop_view.yaml`, `output/loop_following_view.yaml`
- `output/completed_view.yaml`, `output/completed_following_view.yaml`

Earlier animation output was preserved under `output/previous_root_run/` and
`output/previous_viewer_run/` when reorganizing the original demos.

## Smooth surface approximation

The construction uses equal-radius perpendicular circles with centers one radius
apart. For unit radius, an analytical ruling joins

$$A=(\cos\theta-\tfrac12,\sin\theta,0),\qquad
B=(\tfrac12+\cos\phi,0,\pm\sin\phi),\qquad
\cos\phi=\frac{\cos\theta}{1-\cos\theta}.$$

The range is $\theta\in[\pi/3,5\pi/3]$. Both boundary curves are approximated
together by cubic B-splines with shared knots. Interpolation nodes are added
adaptively until the maximum Euclidean boundary error on an independent 8193-point
check grid is at most `FIT_TOLERANCE` (default $10^{-5}$ times the circle radius).
This is a sampled error check, rather than a certified continuous error bound.

The default fit uses 51 controls per boundary, giving $51\times2$ controls per
ruled patch and 204 surface controls in total, compared with the previous 516.
The maximum checked boundary error is approximately $9.41\times10^{-6}$.
The patches are cubic across the sweep and linear along each straight ruling.
The analytical ruling length is $\sqrt{3}$; the fitted endpoints approximate it
within twice the boundary tolerance on the check grid.

Shared seam endpoints are preserved exactly. The negative patch is traversed in
reverse so the moving line is continuous at the shared seam. Partial patches are
obtained by knot insertion and restriction of the full fitted curves, without
refitting. The highlighted line uses those same curves, so it meets the surface.

Reference circles are exact rational NURBS. “Develop/undevelop” here means
revealing/removing the ruled surface, not flattening it into a planar net.
GIF encoding uses Pillow, already installed with nurbspy's Matplotlib dependency.
