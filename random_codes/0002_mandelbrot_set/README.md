# Mandelbrot Set Renderer

An interactive Mandelbrot set visualization built with **NumPy** (vectorized
escape-time computation) and **Matplotlib** (rendering and export). Produces
two high-resolution PNGs and opens them in zoomable/pannable Matplotlib
windows.

## What it is

The renderer computes the escape-time fractal over a grid of complex numbers
using `z = z² + c`, fully vectorized in NumPy — there is no per-pixel Python
loop. Escape counts are made **continuous (smooth)** with the standard
logarithmic smoothing formula `n + 1 − log2(log|z|)`, which removes the visible
color banding of integer iteration counts. Points that never escape are drawn
black via the colormap's bad-color (`set_bad('black')`).

Two views are rendered:

- **Full set** — the whole Mandelbrot set (`mandelbrot_full.png`).
- **Zoom** — Seahorse Valley near the boundary of the main cardioid
  (`mandelbrot_zoom.png`).

Each figure is exported at **1500 × 1200 px, 150 dpi** (10 × 8 inches) and is
also kept open interactively, with the Matplotlib toolbar available for further
zoom and pan.

## Files

| File | Purpose |
| --- | --- |
| `mandelbrot_set.py` | All rendering code, configuration constants and the `main()` entry point. |
| `mandelbrot_full.png` | Pre-rendered image of the full set. |
| `mandelbrot_zoom.png` | Pre-rendered image of the Seahorse Valley zoom. |
| `README.md` | This file. |

## Configuration

All knobs live at the top of `mandelbrot_set.py`:

| Constant | Default | Meaning |
| --- | --- | --- |
| `FULL_X_RANGE` / `FULL_Y_RANGE` | `(-2.6, 1.0)` / `(-1.5, 1.5)` | View window for the full set. |
| `ZOOM_X_RANGE` / `ZOOM_Y_RANGE` | `(-0.7495, -0.7445)` / `(0.0995, 0.1045)` | View window for the Seahorse Valley zoom. |
| `MAX_ITERATIONS` | `3500` | Escape-time iteration cap. |
| `ESCAPE_RADIUS` | `128.0` | Bailout radius. |
| `COLORMAP` | `'inferno'` | Matplotlib colormap for escaped points. |
| `FIGURE_WIDTH_INCHES` / `FIGURE_HEIGHT_INCHES` | `10.0` / `8.0` | Figure size. |
| `IMAGE_DPI` | `150` | Export resolution (→ 1500 × 1200 px). |
| `FULL_OUTPUT_PATH` / `ZOOM_OUTPUT_PATH` | `mandelbrot_full.png` / `mandelbrot_zoom.png` | PNG output paths. |

## Dependencies

```sh
pip install numpy matplotlib
```

The folder has no pinned requirements of its own; the repository-level
`requirements.txt` / `pyproject.toml` are empty placeholders.

## How to run

From inside `0002_mandelbrot_set/`:

```sh
python3 mandelbrot_set.py
```

This renders both views, saves the two PNGs next to the script, then calls
`plt.show()` to display the interactive figures.

**Headless machines:** the script needs a display for `plt.show()`. On a
server or CI box, select a non-interactive backend before running, e.g.:

```sh
MPLBACKEND=Agg python3 mandelbrot_set.py
```

The PNGs are still written; only the interactive windows are suppressed.

## Code layout

- `create_complex_plane()` — builds the sampled complex grid with `linspace`
  and `meshgrid`.
- `compute_smooth_iterations()` — the vectorized escape-time loop with
  continuous iteration counts; returns smoothed values and an "inside" mask.
- `normalize_smooth_iterations()` — log-normalizes escaped values and marks
  interior points as `NaN` so they render black.
- `render_view()` — assembles a figure (image, axes labels, colorbar, title)
  for one `ViewConfig`.
- `export_figure()` — saves a figure as a tight-bounds PNG.
- `main()` — defines the two views and renders/exports/shows them.

## Notes

- Rendering is memory-bound: the plane is held as a `float64`/`complex128`
  array whose size scales with pixels × iterations. Reduce figure size or
  `MAX_ITERATIONS` if memory is constrained.
- Because iteration runs until all points escape or the cap is reached, deep
  zooms with a high `MAX_ITERATIONS` take noticeably longer.
