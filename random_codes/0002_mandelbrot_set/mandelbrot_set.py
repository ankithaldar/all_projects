#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Interactive Mandelbrot set visualization with smooth coloring.

This module computes and renders the Mandelbrot set using NumPy
vectorization and Matplotlib. It produces two interactive views:
the full set and a configurable zoomed region. The escape-time
algorithm uses a continuous iteration count with logarithmic
smoothing to reduce color banding.
'''

from dataclasses import dataclass
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------
# User-configurable view and export settings
# ---------------------------------------------------------------------
FULL_X_RANGE = (-2.6, 1.0)
FULL_Y_RANGE = (-1.5, 1.5)

# Zoomed region: Seahorse Valley near the main cardioid boundary.
ZOOM_X_RANGE = (-0.7495, -0.7445)
ZOOM_Y_RANGE = (0.0995, 0.1045)

MAX_ITERATIONS = 3500
ESCAPE_RADIUS = 128.0
COLORMAP = 'inferno'

FIGURE_WIDTH_INCHES = 10.0
FIGURE_HEIGHT_INCHES = 8.0
IMAGE_DPI = 150

FULL_OUTPUT_PATH = 'mandelbrot_full.png'
ZOOM_OUTPUT_PATH = 'mandelbrot_zoom.png'


@dataclass(frozen=True)
class ViewConfig:
  '''Configuration for one rectangular view of the complex plane.'''

  name: str
  x_range: Tuple[float, float]
  y_range: Tuple[float, float]
  width_inches: float
  height_inches: float
  dpi: int
  output_path: str

  def pixel_size(self) -> Tuple[int, int]:
    '''Return the image size in pixels for this view.'''
    width_pixels = int(round(self.width_inches * self.dpi))
    height_pixels = int(round(self.height_inches * self.dpi))
    return width_pixels, height_pixels


def create_complex_plane(
  x_range: Tuple[float, float],
  y_range: Tuple[float, float],
  width_pixels: int,
  height_pixels: int,
) -> np.ndarray:
  '''Create a rectangular grid of complex numbers.

  Args:
    x_range: Minimum and maximum real-axis values.
    y_range: Minimum and maximum imaginary-axis values.
    width_pixels: Number of horizontal sample points.
    height_pixels: Number of vertical sample points.

  Returns:
    A two-dimensional NumPy array of complex values.
  '''
  # Sample the real and imaginary axes uniformly.
  x_values = np.linspace(
    x_range[0],
    x_range[1],
    width_pixels,
    dtype=np.float64,
  )
  y_values = np.linspace(
    y_range[0],
    y_range[1],
    height_pixels,
    dtype=np.float64,
  )

  # Build a Cartesian grid and convert it to complex numbers.
  real_grid, imaginary_grid = np.meshgrid(x_values, y_values)
  return real_grid + 1j * imaginary_grid


def compute_smooth_iterations(
  complex_plane: np.ndarray,
  max_iterations: int,
  escape_radius: float,
) -> Tuple[np.ndarray, np.ndarray]:
  '''Compute continuous escape-time values for every point.

  Args:
    complex_plane: Two-dimensional array of complex c values.
    max_iterations: Maximum number of escape-time iterations.
    escape_radius: Bailout radius for declaring escape.

  Returns:
    A tuple containing:
      smooth_values: Continuous iteration counts.
      inside_mask: Boolean mask for points that never escaped.
  '''
  # Flatten the plane so active orbits can be updated efficiently.
  shape = complex_plane.shape
  c_values = complex_plane.ravel()
  z_values = np.zeros_like(c_values, dtype=np.complex128)

  # Points that never escape keep the maximum iteration value.
  smooth_values = np.full(
    c_values.shape,
    float(max_iterations),
    dtype=np.float64,
  )
  active = np.ones(c_values.shape, dtype=bool)
  escape_radius_squared = escape_radius * escape_radius

  # Iterate all active orbits together. This avoids Python pixel loops.
  for iteration in range(1, max_iterations + 1):
    if not np.any(active):
      break

    # Find active points and apply z = z^2 + c only to those points.
    active_indices = np.flatnonzero(active)
    z_active = z_values[active_indices]
    z_active = z_active * z_active + c_values[active_indices]
    z_values[active_indices] = z_active

    # Use squared magnitude to avoid an unnecessary square root.
    magnitude_squared = (
      z_active.real * z_active.real + z_active.imag * z_active.imag
    )
    escaped_local = magnitude_squared > escape_radius_squared

    if np.any(escaped_local):
      escaped_indices = active_indices[escaped_local]
      escaped_magnitude = np.sqrt(magnitude_squared[escaped_local])

      # Continuous iteration count:
      # n + 1 - log2(log(|z|)) gives sub-iteration precision.
      with np.errstate(divide='ignore', invalid='ignore'):
        fractional_part = np.log2(
          np.log(np.maximum(escaped_magnitude, np.e))
        )

      smooth_values[escaped_indices] = (
        float(iteration) + 1.0 - fractional_part
      )
      active[escaped_indices] = False

  return smooth_values.reshape(shape), active.reshape(shape)


def normalize_smooth_iterations(
  smooth_values: np.ndarray,
  inside_mask: np.ndarray,
  max_iterations: int,
) -> np.ndarray:
  '''Map smooth iteration counts to a stable color range.

  Args:
    smooth_values: Continuous escape-time values.
    inside_mask: Boolean mask for points inside the Mandelbrot set.
    max_iterations: Maximum number of escape-time iterations.

  Returns:
    Normalized values for Matplotlib. Interior points are NaN so
    they can be drawn with the colormap bad-color.
  '''
  # Clip escaped values to the configured iteration range.
  clipped_values = np.clip(
    smooth_values,
    0.0,
    float(max_iterations),
  )

  # Logarithmic normalization spreads low iteration values out,
  # which helps reveal detail near the boundary.
  normalized = np.log1p(clipped_values) / np.log1p(float(max_iterations))

  # Matplotlib can draw NaN values using colormap.set_bad().
  normalized[inside_mask] = np.nan
  return normalized


def render_view(
  view: ViewConfig,
  max_iterations: int,
  escape_radius: float,
  colormap_name: str,
) -> plt.Figure:
  '''Render one Mandelbrot view into an interactive Matplotlib figure.

  Args:
    view: View configuration.
    max_iterations: Maximum number of escape-time iterations.
    escape_radius: Bailout radius for declaring escape.
    colormap_name: Matplotlib colormap name.

  Returns:
    The Matplotlib figure containing the rendered view.
  '''
  # Convert physical figure size and DPI into image resolution.
  width_pixels, height_pixels = view.pixel_size()

  # Build the complex plane and compute escape-time values.
  complex_plane = create_complex_plane(
    view.x_range,
    view.y_range,
    width_pixels,
    height_pixels,
  )
  smooth_values, inside_mask = compute_smooth_iterations(
    complex_plane,
    max_iterations,
    escape_radius,
  )
  normalized_values = normalize_smooth_iterations(
    smooth_values,
    inside_mask,
    max_iterations,
  )

  # Use a colormap copy so the interior can be drawn as black.
  colormap = plt.get_cmap(colormap_name).copy()
  colormap.set_bad('black')

  # Create the interactive figure. The toolbar provides zoom and pan.
  figure, axis = plt.subplots(
    figsize=(view.width_inches, view.height_inches),
    dpi=view.dpi,
  )
  image = axis.imshow(
    normalized_values,
    cmap=colormap,
    extent=(
      view.x_range[0],
      view.x_range[1],
      view.y_range[0],
      view.y_range[1],
    ),
    origin='lower',
    interpolation='nearest',
  )
  axis.set_aspect('equal')
  axis.set_title(f'{view.name}: use the toolbar to zoom and pan')
  axis.set_xlabel('Real axis')
  axis.set_ylabel('Imaginary axis')
  figure.colorbar(image, ax=axis, label='Log-smoothed escape time')
  figure.tight_layout()

  return figure


def export_figure(figure: plt.Figure, output_path: str, dpi: int) -> None:
  '''Export a figure as a high-resolution PNG file.

  Args:
    figure: Matplotlib figure to save.
    output_path: Destination PNG path.
    dpi: Output resolution in dots per inch.
  '''
  figure.savefig(
    output_path,
    dpi=dpi,
    bbox_inches='tight',
    facecolor='white',
  )


def main() -> None:
  '''Create the configured views, export PNG files, and show plots.'''
  full_view = ViewConfig(
    name='Full Mandelbrot Set',
    x_range=FULL_X_RANGE,
    y_range=FULL_Y_RANGE,
    width_inches=FIGURE_WIDTH_INCHES,
    height_inches=FIGURE_HEIGHT_INCHES,
    dpi=IMAGE_DPI,
    output_path=FULL_OUTPUT_PATH,
  )
  zoom_view = ViewConfig(
    name='Zoomed Region',
    x_range=ZOOM_X_RANGE,
    y_range=ZOOM_Y_RANGE,
    width_inches=FIGURE_WIDTH_INCHES,
    height_inches=FIGURE_HEIGHT_INCHES,
    dpi=IMAGE_DPI,
    output_path=ZOOM_OUTPUT_PATH,
  )

  # Render, export, and keep each figure open for interactive use.
  for view in (full_view, zoom_view):
    figure = render_view(
      view,
      MAX_ITERATIONS,
      ESCAPE_RADIUS,
      COLORMAP,
    )
    export_figure(figure, view.output_path, view.dpi)

  # Display all figures with interactive Matplotlib windows.
  plt.show()


if __name__ == '__main__':
  main()
