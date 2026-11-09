# Exporting the curve and surface demos

Run these commands with nurbspy installed. The demo scripts open one editor each;
export automation lives here so newcomers see only geometry and public APIs.

```powershell
python docs/export_demo_figures.py
python docs/export_demo_figures.py --only 01_curves/01_bezier_planar 02_surfaces/02_bilinear_spatial
```

Each selected script builds its figure through `main(show=False)`. The exporter
creates both files next to that demo's source folder, for example:

```text
demos/01_curves/output/01_bezier_planar/01_bezier_planar.png
demos/01_curves/output/01_bezier_planar/01_bezier_planar.html
```

PNG export fits the camera, and HTML uses the same figure state afterward. Both
retain the reference grid and the default Auto display quality. PNG dimensions
default to 1600 by 1200 pixels; `--width` and `--height` change the image dimensions
without setting a numeric geometry sampling resolution.

Open the HTML file directly in a browser to rotate, zoom, inspect, and change
display settings without a Python server or internet connection. Editing still
uses the Python demo. The HTML includes the initial construction and precomputed
scientific fields and quality choices.

The two STL demos also write their STL files while constructing their figures.
Generated output folders are ignored by Git. The separate
`generate_example_images.py` command continues to maintain the older Matplotlib
illustrations from `demos/documentation/` for the existing documentation pages.

PNG export uses a headless browser. See the [Figure guide](source/figures.md) for
browser installation and the public `savefig()` / `to_html()` methods.

## Documentation gallery

The separate [interactive gallery](source/gallery.md) includes only
`01_curves` and `02_surfaces`. Every HTML documentation build prepares its assets:

- Current PNG/HTML pairs are reused from the demo output folders.
- Missing or outdated pairs are generated through `main(show=False)`.
- Images, HTML, and Python source are copied into the build's `_gallery/` folder.
- Preview cards load an interactive frame only when clicked; opening another
  unloads the previous frame.

A small `.figure-export.json` file beside each pair records source, renderer, and
image-size fingerprints. Existing unstamped exports are adopted when they postdate
their sources and have the expected image dimensions. Deleting the build folder
does not discard this cache. Run the export command above to force regeneration.
Non-HTML documentation builds skip figure generation.

The gallery has its own navigation entry. The existing examples pages remain
separate. CI installs Chromium before building so missing exports can be generated
without opening any desktop editor.
