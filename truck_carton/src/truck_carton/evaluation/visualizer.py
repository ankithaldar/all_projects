#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Visualization utilities for packing results.'''

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import hsv_to_rgb
from mpl_toolkits.mplot3d.art3d import (
  Poly3DCollection,
)

from truck_carton.logging_config import resolve_contained_path

if TYPE_CHECKING:
  from PIL import Image, ImageDraw, ImageFont

  from truck_carton.domain.models import (
    Carton,
    EpisodeData,
    PlacementInfo,
    Truck,
  )
  from truck_carton.packing.space3d import Space3D


class PackingVisualizer:
  '''Renders 3D visualizations of truck packing
  results using matplotlib.'''

  def render_truck(
    self,
    truck: Truck,
    unused_space: Space3D,
    placed_cartons: dict[int, PlacementInfo],
    all_cartons: list[Carton],
    title: str | None = None,
  ) -> plt.Figure:
    '''Render truck.'''
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    carton_lookup = {c.carton_id: c for c in all_cartons}
    truck_cids = [
      cid
      for cid, info in placed_cartons.items()
      if info.truck_id == truck.truck_id
    ]

    store_ids = sorted(
      {carton_lookup[cid].destination_store_id for cid in truck_cids}
    )
    color_map = {}
    for idx, sid in enumerate(store_ids):
      hue = idx / max(len(store_ids), 1)
      color_map[sid] = hsv_to_rgb([hue, 0.7, 0.9])

    for cid in truck_cids:
      info = placed_cartons[cid]
      carton = carton_lookup[cid]
      x, y, z = info.position
      dl, dw, dh = info.oriented_dims
      color = color_map.get(
        carton.destination_store_id,
        [0.5, 0.5, 0.5],
      )
      alpha = 0.3 if carton.is_fragile else 0.6

      self._draw_box(
        ax,
        x,
        y,
        z,
        dl,
        dw,
        dh,
        color,
        alpha,
      )

      ax.text(
        x + dl / 2,
        y + dw / 2,
        z + dh / 2,
        (f'C{cid}\nS{carton.destination_store_id}\nP{carton.priority}'),
        ha='center',
        va='center',
        fontsize=6,
      )

    ax.set_xlim(0, truck.length)
    ax.set_ylim(0, truck.width)
    ax.set_zlim(0, truck.height)
    ax.set_xlabel('Length (X) - Door at 0')
    ax.set_ylabel('Width (Y)')
    ax.set_zlabel('Height (Z)')
    ax.set_title(
      title
      or (
        f'Truck {truck.truck_id} ({truck.length}x{truck.width}x{truck.height})'
      )
    )

    return fig

  def render_all_trucks(
    self,
    episode_data: EpisodeData,
    spaces: list[Space3D],
    placed_cartons: dict[int, PlacementInfo],
  ) -> list[plt.Figure]:
    '''Render all trucks.'''
    figs = []
    for truck, space in zip(episode_data.trucks, spaces, strict=False):
      fig = self.render_truck(
        truck,
        space,
        placed_cartons,
        episode_data.cartons,
        title=(f'Truck {truck.truck_id} | Route: {truck.route}'),
      )
      figs.append(fig)
    return figs

  @staticmethod
  def _draw_box(
    ax: plt.Axes,
    x: int,
    y: int,
    z: int,
    dx: int,
    dy: int,
    dz: int,
    color: list[float] | tuple[float, ...],
    alpha: float,
  ) -> None:
    '''Draw box.'''
    xx = [x, x + dx]
    yy = [y, y + dy]
    zz = [z, z + dz]

    for zi in zz:
      xs = [xx[0], xx[1], xx[1], xx[0], xx[0]]
      ys = [yy[0], yy[0], yy[1], yy[1], yy[0]]
      ax.plot(
        xs,
        ys,
        [zi] * 5,
        color='black',
        linewidth=0.5,
      )

    for xi in xx:
      for yi in yy:
        ax.plot(
          [xi, xi],
          [yi, yi],
          zz,
          color='black',
          linewidth=0.5,
        )

    v = np.array(
      [
        [x, y, z],
        [x + dx, y, z],
        [x + dx, y + dy, z],
        [x, y + dy, z],
        [x, y, z + dz],
        [x + dx, y, z + dz],
        [x + dx, y + dy, z + dz],
        [x, y + dy, z + dz],
      ]
    )

    faces = [
      [v[j] for j in [0, 1, 5, 4]],
      [v[j] for j in [2, 3, 7, 6]],
      [v[j] for j in [0, 3, 7, 4]],
      [v[j] for j in [1, 2, 6, 5]],
      [v[j] for j in [0, 1, 2, 3]],
      [v[j] for j in [4, 5, 6, 7]],
    ]

    collection = Poly3DCollection(
      faces,
      alpha=alpha,
      facecolor=color,
      edgecolor='black',
      linewidth=0.3,
    )
    ax.add_collection3d(collection)

class GridRenderer:
  '''PIL-based rendering engine with ASCII grid,
  Unicode road sprites, status dashboard, and
  frame animation support.'''

  BG_COLOR = '#263238'
  ROAD_COLOR = '#80A7FB'
  FACILITY_COLOR = '#C3E88D'
  TRUCK_COLOR = '#FFCB6B'
  STATUS_COLOR = '#BBBBBB'
  ALERT_COLOR = '#FF5370'

  _TRUCK_STATE_NAMES = {
    0: 'ROUTING',
    1: 'LOADING',
    2: 'AT_DEPOT',
  }

  MAX_FRAMES = 2000

  def __init__(
    self,
    cell_w: int = 24,
    cell_h: int = 18,
    max_frames: int | None = None,
  ) -> None:
    '''Initialize the instance.'''
    self._cell_w = cell_w
    self._cell_h = cell_h
    self._max_frames = max_frames or self.MAX_FRAMES
    self._frames: list = []

  def render(self, snapshot: dict) -> Image:
    '''Render one frame from an env snapshot.'''
    from PIL import Image, ImageDraw, ImageFont

    gw = snapshot.get('grid_world')
    if gw is None:
      img = Image.new('RGB', (200, 100), self.BG_COLOR)
      return img
    rows, cols = gw.rows, gw.cols

    map_w = cols * self._cell_w
    map_h = rows * self._cell_h
    status_h = 220
    margin = 20
    img_w = map_w + 2 * margin
    img_h = map_h + status_h + 2 * margin

    img = Image.new('RGB', (img_w, img_h), self.BG_COLOR)
    draw = ImageDraw.Draw(img)

    try:
      font = ImageFont.truetype('arial', 13)
      small = ImageFont.truetype('arial', 10)
    except OSError:
      font = ImageFont.load_default()
      small = font

    # Layer 1: roads
    grid = gw.grid
    for r in range(rows):
      for c in range(cols):
        x = margin + c * self._cell_w
        y = margin + r * self._cell_h
        cell = int(grid[r, c])

        if cell == 0:  # TERRAIN
          continue
        if cell == 1:  # ROAD
          sprite = self._road_sprite(r, c, grid)
          draw.text(
            (x + 4, y),
            sprite,
            fill=self.ROAD_COLOR,
            font=font,
          )

    # Layer 2: facilities
    warehouses = snapshot.get('warehouses', [])
    stores = snapshot.get('stores', [])
    depot = gw.depot_position

    dr, dc = depot
    dx = margin + dc * self._cell_w
    dy = margin + dr * self._cell_h
    draw.rectangle(
      [dx, dy, dx + self._cell_w - 1, dy + self._cell_h - 1],
      fill='#455A64',
    )
    draw.text(
      (dx + 4, dy),
      'D',
      fill=self.FACILITY_COLOR,
      font=font,
    )

    for wh in warehouses:
      wr, wc = wh.position
      wx = margin + wc * self._cell_w
      wy = margin + wr * self._cell_h
      draw.rectangle(
        [wx, wy, wx + self._cell_w - 1, wy + self._cell_h - 1],
        fill='#1A237E',
      )
      draw.text(
        (wx + 2, wy),
        f'W{wh.warehouse_id}',
        fill=self.FACILITY_COLOR,
        font=font,
      )

    for st in stores:
      sr, sc = st.position
      sx = margin + sc * self._cell_w
      sy = margin + sr * self._cell_h
      draw.rectangle(
        [sx, sy, sx + self._cell_w - 1, sy + self._cell_h - 1],
        fill='#1B5E20',
      )
      draw.text(
        (sx + 2, sy),
        f'S{st.store_id}',
        fill=self.FACILITY_COLOR,
        font=font,
      )

    # Layer 3: trucks
    truck_colors = [
      '#FFCB6B',
      '#FF5370',
      '#C792EA',
      '#89DDFF',
      '#F78C6C',
    ]
    trucks = snapshot.get('trucks', [])
    for i, truck in enumerate(trucks):
      tr, tc = truck.position
      tx = margin + tc * self._cell_w
      ty = margin + tr * self._cell_h
      color = truck_colors[i % len(truck_colors)]
      draw.text(
        (tx + 8, ty),
        '*',
        fill=color,
        font=font,
      )

    # Layer 4: status panel
    panel_y = margin + map_h + 10
    self._draw_status(
      draw,
      snapshot,
      margin,
      panel_y,
      img_w,
      font,
      small,
    )

    return img

  def _road_sprite(self, r: int, c: int, grid: np.ndarray) -> str:
    '''Road sprite.'''
    rows, cols = grid.shape

    def _traversable(nr: int, nc: int) -> bool:
      '''Traversable.'''
      if nr < 0 or nr >= rows:
        return False
      if nc < 0 or nc >= cols:
        return False
      return int(grid[nr, nc]) != 0

    top = _traversable(r - 1, c)
    bottom = _traversable(r + 1, c)
    left = _traversable(r, c - 1)
    right = _traversable(r, c + 1)

    if (top or bottom) and not right and not left:
      return '\u2502'  # │
    if (right or left) and not top and not bottom:
      return '\u2500'  # ─
    if bottom and right and not top and not left:
      return '\u250c'  # ┌
    if bottom and left and not top and not right:
      return '\u2510'  # ┐
    if top and right and not bottom and not left:
      return '\u2514'  # └
    if top and left and not bottom and not right:
      return '\u2518'  # ┘
    if top and bottom and right and not left:
      return '\u251c'  # ├
    if top and bottom and left and not right:
      return '\u2524'  # ┤
    if bottom and right and left and not top:
      return '\u252c'  # ┬
    if top and right and left and not bottom:
      return '\u2534'  # ┴
    if top and bottom and right and left:
      return '\u253c'  # ┼
    return '\u00b7'  # · fallback

  def _draw_status(
    self,
    draw: ImageDraw,
    snapshot: dict,
    x: int,
    y: int,
    unused_img_w: int,
    unused_font: ImageFont,
    small: ImageFont,
  ) -> None:
    '''Draw status.'''
    trucks = snapshot.get('trucks', [])
    cargo = snapshot.get('truck_cargo', [])
    travel = snapshot.get('truck_travel', [])
    wh_cartons = snapshot.get('warehouse_cartons', {})
    delivered = snapshot.get('delivered', set())
    total = snapshot.get('total_cartons', 0)
    step = snapshot.get('step', 0)
    reward = snapshot.get('cumulative_reward', 0.0)

    # Global status
    lines = [
      f'Step: {step}'
      f'  Delivered: {len(delivered)}/{total}'
      f'  Reward: {reward:.1f}',
    ]

    # Warehouse inventory
    wh_parts = []
    for wh_id, cids in sorted(wh_cartons.items()):
      wh_parts.append(f'W{wh_id}:{len(cids)}')
    if wh_parts:
      lines.append('Warehouses: ' + '  '.join(wh_parts))

    # Per-truck status
    for i, truck in enumerate(trucks):
      state_name = self._TRUCK_STATE_NAMES.get(int(truck.state), '?')
      n_cargo = len(cargo[i]) if i < len(cargo) else 0
      dist = travel[i] if i < len(travel) else 0.0
      lines.append(
        f'T{truck.truck_id}: {state_name}'
        f'  pos={truck.position}'
        f'  cargo={n_cargo}'
        f'  travel={dist:.0f}'
      )

    for i, line in enumerate(lines):
      draw.text(
        (x, y + i * 16),
        line,
        fill=self.STATUS_COLOR,
        font=small,
      )

  def capture_frame(self, snapshot: dict) -> None:
    '''Capture frame.'''
    if len(self._frames) >= self._max_frames:
      self._frames.pop(0)
    self._frames.append(self.render(snapshot))

  def clear_frames(self) -> None:
    '''Clear frames.'''
    self._frames = []

  def save_gif(
    self,
    path: str,
    duration: int = 300,
    base_dir: str | Path | None = None,
  ) -> None:
    '''Save gif.

    path must resolve inside base_dir (default cwd);
    traversal and symlink escapes raise ValueError (S-02).
    '''
    resolved = resolve_contained_path(path, base_dir=base_dir)
    if not self._frames:
      return
    resolved.parent.mkdir(parents=True, exist_ok=True)
    self._frames[0].save(
      str(resolved),
      save_all=True,
      append_images=self._frames[1:],
      duration=duration,
      loop=0,
    )
