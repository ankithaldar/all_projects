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


class AsciiWorldRenderer:
  '''Composites a layered ASCII snapshot of the world into an image.'''

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
      list[list[list[str]]]: [railroads, vehicles, facilities] layers,
      each indexed as ``layer[y][x]``.
    '''
    railroads = [[' '] * world.size_x for _ in range(world.size_y)]
    vehicles = [[' '] * world.size_x for _ in range(world.size_y)]
    facilities = [[' '] * world.size_x for _ in range(world.size_y)]
    for x in range(world.size_x):
      for y in range(world.size_y):
        cell = world.grid[x][y]
        if isinstance(cell, RailroadCell):
          railroads[y][x] = self._railroad_sprite(x, y, world)
          continue
        glyph = _FACILITY_GLYPHS.get(type(cell).__name__)
        if glyph is not None:
          facilities[y][x] = glyph
          if getattr(cell, 'distribution', None) is not None:
            for truck in cell.distribution.fleet:
              if truck.is_enroute():
                tx, ty = truck.current_location()
                vehicles[ty][tx] = '*'
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
    map_text = '\n'.join(''.join(row) for row in layers[0])
    probe = ImageDraw.Draw(Image.new('RGB', (10, 10)))
    left, top, right, bottom = probe.multiline_textbbox((0, 0), map_text, font=self.map_font)
    map_w, map_h = right - left, bottom - top

    img_w = map_w + 2 * self.margin_side
    img_h = int(map_h * 4.0)
    img = Image.new('RGB', (img_w, img_h), color=self.background)
    canvas = ImageDraw.Draw(img)

    for layer, color in zip(layers, _LAYER_COLORS):
      text = '\n'.join(''.join(row) for row in layer)
      canvas.multiline_text((self.margin_side, self.margin_top), text, font=self.map_font, fill=color)

    logo = Image.open(_ASSETS / 'world-of-supply-logo.png').convert('RGBA')
    logo.thumbnail((img_w / 5, img_h / 10), Image.LANCZOS)
    img.paste(logo, (int(img_w / 2 - img_w / 10), 0), mask=logo)

    status = WorldStatusFormatter().status(world)
    n_rows = -(-len(status) // self.status_columns)
    column_width = img_w / self.status_columns * 0.9
    for column in range(self.status_columns):
      chunk = status[column * n_rows:(column + 1) * n_rows]
      x_position = img_w / 2 - self.status_columns * column_width / 2 + column_width * column
      status_text = yaml.dump(chunk).replace('\'', '')
      canvas.multiline_text(
          (x_position, map_h * 1.1),
          status_text,
          font=self.status_font,
          fill='#BBBBBB',
      )
    return img


class NotebookAnimator:
  '''Plays rendered frames as an inline HTML5 video inside notebooks.'''

  @staticmethod
  def plot_sequence_images(image_array: np.ndarray) -> None:
    '''Display a sequence of images as a matplotlib animation.

    Args:
      image_array: Array shaped ``(num_images, height, width, channels)``.

    Raises:
      ImportError: If IPython is unavailable outside notebook environments.
    '''
    from IPython.display import HTML, display
    from matplotlib import animation, pyplot as plt

    dpi = 72.0
    height_px, width_px = image_array[0].shape[:2]
    figure = plt.figure(figsize=(width_px / dpi, height_px / dpi), dpi=dpi)
    image = plt.figimage(image_array[0])

    def animate(frame: int):
      '''Swap in the frame pixel data.

      Args:
        frame: Frame index.

      Returns:
        tuple: Artist objects for blitting.
      '''
      image.set_array(image_array[frame])
      return (image,)

    anim = animation.FuncAnimation(
        figure, animate, frames=len(image_array), interval=200, repeat_delay=1, repeat=True
    )
    display(HTML(anim.to_html5_video()))
