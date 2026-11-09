#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''PIL/matplotlib renderers: layered ASCII map image and notebook animation.'''

from __future__ import annotations

import importlib.resources
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont

from world_of_supply.geography import RailroadCell
from world_of_supply.rendering.sprites import railroad_glyph
from world_of_supply.rendering.status import WorldStatusFormatter
from world_of_supply.world import World

_ASSETS = Path(str(importlib.resources.files('world_of_supply') / 'assets'))

_FACILITY_GLYPHS = {
    'SteelFactoryCell': 'S',
    'LumberFactoryCell': 'L',
    'ToyFactoryCell': 'T',
    'WarehouseCell': 'W',
    'RetailerCell': 'R',
}

_LAYER_COLORS = ('#80A7FB', '#FFCB6B', '#C3E88D')


def _load_font(name: str, size: int) -> ImageFont.FreeTypeFont:
  '''Load a bundled TrueType font from package assets.

  Args:
    name: File name inside the assets directory.
    size: Font size in pixels.

  Returns:
    ImageFont.FreeTypeFont: Loaded font.
  '''
  return ImageFont.truetype(str(_ASSETS / name), size)


def _blank_layer(world: World) -> list[list[str]]:
  '''Create one blank character layer sized to the world grid.

  Args:
    world: World snapshot providing the grid dimensions.

  Returns:
    list[list[str]]: Blank layer indexed as ``layer[y][x]``.
  '''
  return [[' '] * world.size_x for _ in range(world.size_y)]


def _blank_layers(world: World) -> list[list[list[str]]]:
  '''Create the three blank layers used by the ASCII renderer.

  Args:
    world: World snapshot providing the grid dimensions.

  Returns:
    list[list[list[str]]]: Blank ``[railroads, vehicles,
      facilities]`` layers.
  '''
  return [_blank_layer(world) for _ in range(3)]


def _facility_glyph(cell) -> str | None:
  '''Look up the display glyph for one grid cell.

  Args:
    cell: Grid cell snapshot.

  Returns:
    str | None: Facility glyph, or ``None`` for other cells.
  '''
  return _FACILITY_GLYPHS.get(type(cell).__name__)


def _paint_enroute_trucks(cell, vehicles: list[list[str]]) -> None:
  '''Paint en-route trucks from one facility onto the vehicle layer.

  Args:
    cell: Grid cell snapshot, possibly holding a distribution unit.
    vehicles: Vehicle layer mutated in place.

  Returns:
    None: The vehicle layer is mutated in place.
  '''
  if getattr(cell, 'distribution', None) is None:
    return
  for truck in cell.distribution.fleet:
    if truck.is_enroute():
      tx, ty = truck.current_location()
      vehicles[ty][tx] = '*'


def _layer_text(layer: list[list[str]]) -> str:
  '''Join one character layer into display text.

  Args:
    layer: Character layer indexed as ``layer[y][x]``.

  Returns:
    str: Layer rows joined by newlines.
  '''
  return '\n'.join(''.join(row) for row in layer)


def _measure_text(text: str, font) -> tuple[int, int]:
  '''Measure display text with the given font.

  Args:
    text: Text to measure.
    font: Font used to measure the text.

  Returns:
    tuple[int, int]: ``(width, height)`` of the text bounding box.
  '''
  probe = ImageDraw.Draw(Image.new('RGB', (10, 10)))
  left, top, right, bottom = probe.multiline_textbbox(
      (0, 0), text, font=font
  )
  return right - left, bottom - top


def _canvas_size(
    map_w: int,
    map_h: int,
    margin_side: int,
) -> tuple[int, int]:
  '''Compute the output image size for one map.

  Args:
    map_w: Measured map width in pixels.
    map_h: Measured map height in pixels.
    margin_side: Horizontal empty border in pixels.

  Returns:
    tuple[int, int]: ``(image_width, image_height)`` in pixels.
  '''
  img_w = map_w + 2 * margin_side
  img_h = int(map_h * 4.0)
  return img_w, img_h


def _draw_layers(
    canvas,
    layers: list[list[list[str]]],
    margin_side: int,
    margin_top: int,
    font,
) -> None:
  '''Draw each ASCII layer onto the canvas in its layer color.

  Args:
    canvas: PIL draw object for the output image.
    layers: Character layers in draw order.
    margin_side: Horizontal empty border in pixels.
    margin_top: Top border in pixels.
    font: Monospace font for the grid.

  Returns:
    None: The canvas is mutated in place.
  '''
  for layer, color in zip(layers, _LAYER_COLORS):
    canvas.multiline_text(
        (margin_side, margin_top),
        _layer_text(layer),
        font=font,
        fill=color,
    )


def _paste_logo(img, img_w: int, img_h: int) -> None:
  '''Paste the project logo at the top center of the image.

  Args:
    img: Output image mutated in place.
    img_w: Output image width in pixels.
    img_h: Output image height in pixels.

  Returns:
    None: The image is mutated in place.
  '''
  logo = Image.open(_ASSETS / 'world-of-supply-logo.png').convert('RGBA')
  logo.thumbnail((img_w / 5, img_h / 10), Image.LANCZOS)
  img.paste(logo, (int(img_w / 2 - img_w / 10), 0), mask=logo)


def _status_columns_chunk(status: list, column: int, n_rows: int) -> list:
  '''Select one status column chunk for display.

  Args:
    status: Full status entry list.
    column: Column index to select.
    n_rows: Number of rows per column.

  Returns:
    list: Status entries for the requested column.
  '''
  return status[column * n_rows:(column + 1) * n_rows]


def _status_column_x(
    img_w: int,
    status_columns: int,
    column_width: float,
    column: int,
) -> float:
  '''Compute the x position of one status column.

  Args:
    img_w: Output image width in pixels.
    status_columns: Number of status panel columns.
    column_width: Width allocated to one column in pixels.
    column: Column index.

  Returns:
    float: X position of the column in pixels.
  '''
  offset = status_columns * column_width / 2
  return img_w / 2 - offset + column_width * column


def _draw_status_column(canvas, chunk: list, x_position, y_position, font):
  '''Dump one status column as YAML text onto the canvas.

  Args:
    canvas: PIL draw object for the output image.
    chunk: Status entries for one column.
    x_position: X position of the column in pixels.
    y_position: Y position of the column in pixels.
    font: Font for the status panels.

  Returns:
    None: The canvas is mutated in place.
  '''
  status_text = yaml.dump(chunk).replace('\'', '')
  canvas.multiline_text(
      (x_position, y_position),
      status_text,
      font=font,
      fill='#BBBBBB',
  )


def _draw_status_panels(
    canvas,
    world: World,
    img_w: int,
    map_h: int,
    status_font,
    status_columns: int,
) -> None:
  '''Draw all YAML status panels below the map.

  Args:
    canvas: PIL draw object for the output image.
    world: World snapshot to summarize.
    img_w: Output image width in pixels.
    map_h: Measured map height in pixels.
    status_font: Font for the status panels.
    status_columns: Number of status panel columns.

  Returns:
    None: The canvas is mutated in place.
  '''
  status = WorldStatusFormatter().status(world)
  n_rows = -(-len(status) // status_columns)
  column_width = img_w / status_columns * 0.9
  for column in range(status_columns):
    chunk = _status_columns_chunk(status, column, n_rows)
    x_position = _status_column_x(
        img_w, status_columns, column_width, column
    )
    _draw_status_column(
        canvas, chunk, x_position, map_h * 1.1, status_font
    )


def _animation_dimensions(image_array: np.ndarray) -> tuple[int, int]:
  '''Read the frame dimensions from an image array.

  Args:
    image_array: Array shaped ``(num_images, height, width,
      channels)``.

  Returns:
    tuple[int, int]: ``(height_px, width_px)`` of one frame.
  '''
  height_px, width_px = image_array[0].shape[:2]
  return height_px, width_px


def _animation_figure_size(image_array: np.ndarray) -> tuple[float, float]:
  '''Compute the matplotlib figure size for an image array.

  Args:
    image_array: Array shaped ``(num_images, height, width,
      channels)``.

  Returns:
    tuple[float, float]: ``(width_inches, height_inches)`` at the
      fixed animation DPI.
  '''
  dpi = 72.0
  height_px, width_px = _animation_dimensions(image_array)
  return width_px / dpi, height_px / dpi


def _make_animation(figure, image, image_array: np.ndarray, animation):
  '''Build the looping matplotlib animation for a frame sequence.

  Args:
    figure: Matplotlib figure displaying the frames.
    image: Matplotlib image artist updated per frame.
    image_array: Array shaped ``(num_images, height, width,
      channels)``.
    animation: Matplotlib animation module used to build the
      animation.

  Returns:
    animation.FuncAnimation: Looping frame animation.
  '''

  def animate(frame: int):
    '''Swap in the frame pixel data.

    Args:
      frame: Frame index.

    Returns:
      tuple: Artist objects for blitting.
    '''
    image.set_array(image_array[frame])
    return (image,)

  return animation.FuncAnimation(
      figure,
      animate,
      frames=len(image_array),
      interval=200,
      repeat_delay=1,
      repeat=True,
  )


class AsciiWorldRenderer:
  '''Composites a layered ASCII snapshot of the world into an image.

  Attributes:
    margin_side: Horizontal empty border in pixels.
    margin_top: Top border in pixels.
    background: Canvas background color.
    status_columns: Number of status panel columns.
    map_font: Monospace font for the grid.
    status_font: Font for the YAML status panels.
  '''

  def __init__(
      self,
      margin_side: int = 150,
      margin_top: int = 20,
      map_font_size: int = 24,
      status_font_size: int = 11,
      background: str = '#263238',
      status_columns: int = 3,
  ) -> None:
    '''Configure canvas geometry and style.

    Args:
      margin_side: Horizontal empty border in pixels.
      margin_top: Top border in pixels.
      map_font_size: Monospace font size for the grid.
      status_font_size: Font size of YAML status panels.
      background: Canvas background color.
      status_columns: Number of status panel columns.

    Returns:
      None: The renderer is mutated in place.
    '''
    self.margin_side = margin_side
    self.margin_top = margin_top
    self.background = background
    self.status_columns = status_columns
    self.map_font = _load_font('FiraMono-Bold.ttf', map_font_size)
    self.status_font = _load_font('monaco.ttf', status_font_size)

  def _ascii_layers(self, world: World) -> list[list[list[str]]]:
    '''Build the stacked character layers of the map.

    Args:
      world: World snapshot.

    Returns:
      list[list[list[str]]]: [railroads, vehicles, facilities]
      layers, each indexed as ``layer[y][x]``.
    '''
    railroads, vehicles, facilities = _blank_layers(world)
    for x in range(world.size_x):
      for y in range(world.size_y):
        cell = world.grid[x][y]
        if isinstance(cell, RailroadCell):
          railroads[y][x] = self._railroad_sprite(x, y, world)
          continue
        glyph = _facility_glyph(cell)
        if glyph is not None:
          facilities[y][x] = glyph
          _paint_enroute_trucks(cell, vehicles)
    return [railroads, vehicles, facilities]

  @staticmethod
  def _railroad_sprite(x: int, y: int, world: World) -> str:
    '''Pick a box-drawing sprite based on neighboring railroad cells.

    Args:
      x: Cell x position.
      y: Cell y position.
      world: World providing the railroad predicate.

    Returns:
      str: Box-drawing character.
    '''

    def is_railroad(nx: int, ny: int) -> bool:
      '''Bound-safe railroad predicate for neighbors.

      Args:
        nx: Neighbor x coordinate.
        ny: Neighbor y coordinate.

      Returns:
        bool: True when the neighbor is a railroad cell.
      '''
      if not (0 <= nx < world.size_x and 0 <= ny < world.size_y):
        return False
      return isinstance(world.grid[nx][ny], RailroadCell)

    return railroad_glyph(x, y, is_railroad)

  def render(self, world: World) -> Image.Image:
    '''Render the full world image including status panels.

    Args:
      world: World snapshot.

    Returns:
      Image.Image: Composited RGB image.
    '''
    layers = self._ascii_layers(world)
    map_w, map_h = _measure_text(_layer_text(layers[0]), self.map_font)
    img_w, img_h = _canvas_size(map_w, map_h, self.margin_side)
    img = Image.new('RGB', (img_w, img_h), color=self.background)
    canvas = ImageDraw.Draw(img)
    _draw_layers(
        canvas, layers, self.margin_side, self.margin_top, self.map_font
    )
    _paste_logo(img, img_w, img_h)
    _draw_status_panels(
        canvas, world, img_w, map_h, self.status_font, self.status_columns
    )
    return img


class NotebookAnimator:
  '''Plays rendered frames as an inline HTML5 video inside notebooks.

  The animator is stateless: each call to
  :meth:`plot_sequence_images` builds a fresh figure and animation
  from the supplied frame array.
  '''

  @staticmethod
  def plot_sequence_images(image_array: np.ndarray) -> None:
    '''Display a sequence of images as a matplotlib animation.

    Args:
      image_array: Array shaped ``(num_images, height, width,
        channels)``.

    Returns:
      None: The animation is displayed; no value is returned.

    Raises:
      ImportError: If IPython is unavailable outside notebook
        environments.
    '''
    from IPython.display import HTML, display
    from matplotlib import animation, pyplot as plt

    width_inches, height_inches = _animation_figure_size(image_array)
    figure = plt.figure(
        figsize=(width_inches, height_inches), dpi=72.0
    )
    image = plt.figimage(image_array[0])
    anim = _make_animation(figure, image, image_array, animation)
    display(HTML(anim.to_html5_video()))
