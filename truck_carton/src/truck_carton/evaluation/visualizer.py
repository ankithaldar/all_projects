from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import hsv_to_rgb
from mpl_toolkits.mplot3d.art3d import (
  Poly3DCollection,
)

if TYPE_CHECKING:
  from truck_carton.domain.models import (
    Carton,
    EpisodeData,
    GridWorld,
    PlacementInfo,
    Truck,
    Warehouse,
  )
  from truck_carton.packing.space3d import Space3D


class PackingVisualizer:
  """Renders 3D visualizations of truck packing
  results using matplotlib."""

  def render_truck(
    self,
    truck: Truck,
    space: Space3D,
    placed_cartons: dict[int, PlacementInfo],
    all_cartons: list[Carton],
    title: str | None = None,
  ) -> plt.Figure:
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    carton_lookup = {
      c.carton_id: c for c in all_cartons
    }
    truck_cids = [
      cid
      for cid, info in placed_cartons.items()
      if info.truck_id == truck.truck_id
    ]

    store_ids = sorted({
      carton_lookup[cid].destination_store_id
      for cid in truck_cids
    })
    color_map = {}
    for idx, sid in enumerate(store_ids):
      hue = idx / max(len(store_ids), 1)
      color_map[sid] = hsv_to_rgb(
        [hue, 0.7, 0.9]
      )

    for cid in truck_cids:
      info = placed_cartons[cid]
      carton = carton_lookup[cid]
      x, y, z = info.position
      dl, dw, dh = info.oriented_dims
      color = color_map.get(
        carton.destination_store_id,
        [0.5, 0.5, 0.5],
      )
      alpha = (
        0.3 if carton.is_fragile else 0.6
      )

      self._draw_box(
        ax, x, y, z, dl, dw, dh,
        color, alpha,
      )

      ax.text(
        x + dl / 2,
        y + dw / 2,
        z + dh / 2,
        (
          f'C{cid}\n'
          f'S{carton.destination_store_id}\n'
          f'P{carton.priority}'
        ),
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
        f'Truck {truck.truck_id}'
        f' ({truck.length}x{truck.width}'
        f'x{truck.height})'
      )
    )

    return fig

  def render_all_trucks(
    self,
    episode_data: EpisodeData,
    spaces: list[Space3D],
    placed_cartons: dict[int, PlacementInfo],
  ) -> list[plt.Figure]:
    figs = []
    for truck, space in zip(
      episode_data.trucks, spaces
    ):
      fig = self.render_truck(
        truck,
        space,
        placed_cartons,
        episode_data.cartons,
        title=(
          f'Truck {truck.truck_id}'
          f' | Route: {truck.route}'
        ),
      )
      figs.append(fig)
    return figs

  @staticmethod
  def _draw_box(
    ax: plt.Axes,
    x: int, y: int, z: int,
    dx: int, dy: int, dz: int,
    color: list[float] | tuple[float, ...],
    alpha: float,
  ) -> None:
    xx = [x, x + dx]
    yy = [y, y + dy]
    zz = [z, z + dz]

    for zi in zz:
      xs = [
        xx[0], xx[1], xx[1], xx[0], xx[0]
      ]
      ys = [
        yy[0], yy[0], yy[1], yy[1], yy[0]
      ]
      ax.plot(
        xs, ys, [zi] * 5,
        color='black', linewidth=0.5,
      )

    for xi in xx:
      for yi in yy:
        ax.plot(
          [xi, xi], [yi, yi], zz,
          color='black', linewidth=0.5,
        )

    v = np.array([
      [x, y, z],
      [x + dx, y, z],
      [x + dx, y + dy, z],
      [x, y + dy, z],
      [x, y, z + dz],
      [x + dx, y, z + dz],
      [x + dx, y + dy, z + dz],
      [x, y + dy, z + dz],
    ])

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

  # Keep the simple matplotlib grid renderer
  # for backward compatibility.
  def render_grid(
    self,
    grid_world: GridWorld,
    warehouses: list[Warehouse],
    stores: list,
    trucks: list[Truck],
    title: str | None = None,
  ) -> plt.Figure:
    """Render the 2D grid with facilities and
    truck positions."""
    from truck_carton.domain.models import CellType

    fig, ax = plt.subplots(figsize=(10, 10))

    grid = grid_world.grid.astype(float)
    cmap = plt.cm.Pastel1
    ax.imshow(
      grid, cmap=cmap, origin='upper',
      vmin=0, vmax=4,
    )

    dp = grid_world.depot_position
    ax.plot(
      dp[1], dp[0], 's',
      color='black', markersize=14,
      label='Depot',
    )
    ax.annotate(
      'D', (dp[1], dp[0]),
      ha='center', va='center',
      color='white', fontweight='bold',
    )

    for wh in warehouses:
      r, c = wh.position
      ax.plot(
        c, r, '^',
        color='blue', markersize=12,
      )
      ax.annotate(
        f'W{wh.warehouse_id}', (c, r - 0.4),
        ha='center', fontsize=8,
      )

    for store in stores:
      r, c = store.position
      ax.plot(
        c, r, 'o',
        color='green', markersize=12,
      )
      ax.annotate(
        f'S{store.store_id}', (c, r - 0.4),
        ha='center', fontsize=8,
      )

    colors = ['red', 'orange', 'purple',
              'cyan', 'magenta']
    for i, truck in enumerate(trucks):
      r, c = truck.position
      color = colors[i % len(colors)]
      ax.plot(
        c, r, 'D',
        color=color, markersize=10,
        label=f'Truck {truck.truck_id}',
      )

    ax.set_xlim(-0.5, grid_world.cols - 0.5)
    ax.set_ylim(
      grid_world.rows - 0.5, -0.5
    )
    ax.set_title(
      title or 'Grid World Layout'
    )
    ax.legend(
      loc='upper right', fontsize=8
    )
    ax.grid(True, alpha=0.3)

    return fig


class GridRenderer:
  """PIL-based rendering engine with ASCII grid,
  Unicode road sprites, status dashboard, and
  frame animation support."""

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
    self._cell_w = cell_w
    self._cell_h = cell_h
    self._max_frames = max_frames or self.MAX_FRAMES
    self._frames: list = []

  def render(self, snapshot: dict) -> 'Image':
    """Render one frame from an env snapshot."""
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

    img = Image.new(
      'RGB', (img_w, img_h), self.BG_COLOR
    )
    draw = ImageDraw.Draw(img)

    try:
      font = ImageFont.truetype(
        'arial', 13
      )
      small = ImageFont.truetype(
        'arial', 10
      )
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
          sprite = self._road_sprite(
            r, c, grid
          )
          draw.text(
            (x + 4, y), sprite,
            fill=self.ROAD_COLOR, font=font,
          )

    # Layer 2: facilities
    warehouses = snapshot.get('warehouses', [])
    stores = snapshot.get('stores', [])
    depot = gw.depot_position

    dr, dc = depot
    dx = margin + dc * self._cell_w
    dy = margin + dr * self._cell_h
    draw.rectangle(
      [dx, dy, dx + self._cell_w - 1,
       dy + self._cell_h - 1],
      fill='#455A64',
    )
    draw.text(
      (dx + 4, dy), 'D',
      fill=self.FACILITY_COLOR, font=font,
    )

    for wh in warehouses:
      wr, wc = wh.position
      wx = margin + wc * self._cell_w
      wy = margin + wr * self._cell_h
      draw.rectangle(
        [wx, wy, wx + self._cell_w - 1,
         wy + self._cell_h - 1],
        fill='#1A237E',
      )
      draw.text(
        (wx + 2, wy),
        f'W{wh.warehouse_id}',
        fill=self.FACILITY_COLOR, font=font,
      )

    for st in stores:
      sr, sc = st.position
      sx = margin + sc * self._cell_w
      sy = margin + sr * self._cell_h
      draw.rectangle(
        [sx, sy, sx + self._cell_w - 1,
         sy + self._cell_h - 1],
        fill='#1B5E20',
      )
      draw.text(
        (sx + 2, sy),
        f'S{st.store_id}',
        fill=self.FACILITY_COLOR, font=font,
      )

    # Layer 3: trucks
    truck_colors = [
      '#FFCB6B', '#FF5370', '#C792EA',
      '#89DDFF', '#F78C6C',
    ]
    trucks = snapshot.get('trucks', [])
    for i, truck in enumerate(trucks):
      tr, tc = truck.position
      tx = margin + tc * self._cell_w
      ty = margin + tr * self._cell_h
      color = truck_colors[
        i % len(truck_colors)
      ]
      draw.text(
        (tx + 8, ty), '*',
        fill=color, font=font,
      )

    # Layer 4: status panel
    panel_y = margin + map_h + 10
    self._draw_status(
      draw, snapshot, margin, panel_y,
      img_w, font, small,
    )

    return img

  def _road_sprite(
    self, r: int, c: int, grid: np.ndarray
  ) -> str:
    rows, cols = grid.shape

    def _traversable(nr: int, nc: int) -> bool:
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
      return '\u250C'  # ┌
    if bottom and left and not top and not right:
      return '\u2510'  # ┐
    if top and right and not bottom and not left:
      return '\u2514'  # └
    if top and left and not bottom and not right:
      return '\u2518'  # ┘
    if top and bottom and right and not left:
      return '\u251C'  # ├
    if top and bottom and left and not right:
      return '\u2524'  # ┤
    if bottom and right and left and not top:
      return '\u252C'  # ┬
    if top and right and left and not bottom:
      return '\u2534'  # ┴
    if top and bottom and right and left:
      return '\u253C'  # ┼
    return '\u00B7'  # · fallback

  def _draw_status(
    self,
    draw: 'ImageDraw',
    snapshot: dict,
    x: int,
    y: int,
    img_w: int,
    font: 'ImageFont',
    small: 'ImageFont',
  ) -> None:
    trucks = snapshot.get('trucks', [])
    cargo = snapshot.get('truck_cargo', [])
    travel = snapshot.get('truck_travel', [])
    wh_cartons = snapshot.get(
      'warehouse_cartons', {}
    )
    delivered = snapshot.get('delivered', set())
    total = snapshot.get('total_cartons', 0)
    step = snapshot.get('step', 0)
    reward = snapshot.get(
      'cumulative_reward', 0.0
    )

    # Global status
    lines = [
      f'Step: {step}'
      f'  Delivered: {len(delivered)}/{total}'
      f'  Reward: {reward:.1f}',
    ]

    # Warehouse inventory
    wh_parts = []
    for wh_id, cids in sorted(
      wh_cartons.items()
    ):
      wh_parts.append(
        f'W{wh_id}:{len(cids)}'
      )
    if wh_parts:
      lines.append(
        'Warehouses: ' + '  '.join(wh_parts)
      )

    # Per-truck status
    for i, truck in enumerate(trucks):
      state_name = self._TRUCK_STATE_NAMES.get(
        int(truck.state), '?'
      )
      n_cargo = (
        len(cargo[i]) if i < len(cargo) else 0
      )
      dist = (
        travel[i] if i < len(travel) else 0.0
      )
      bar = self._progress_bar(
        len(delivered), total
      )
      lines.append(
        f'T{truck.truck_id}: {state_name}'
        f'  pos={truck.position}'
        f'  cargo={n_cargo}'
        f'  travel={dist:.0f}'
      )

    for i, line in enumerate(lines):
      draw.text(
        (x, y + i * 16), line,
        fill=self.STATUS_COLOR, font=small,
      )

  @staticmethod
  def _progress_bar(
    done: int, total: int, width: int = 15
  ) -> str:
    if total == 0:
      filled = 0
    else:
      filled = round(
        min(done, total) / total * width
      )
    bar = '=' * filled + '-' * (width - filled)
    return f'[{bar}] {done}/{total}'

  def capture_frame(
    self, snapshot: dict
  ) -> None:
    if len(self._frames) >= self._max_frames:
      self._frames.pop(0)
    self._frames.append(self.render(snapshot))

  def clear_frames(self) -> None:
    self._frames = []

  def save_gif(
    self,
    path: str,
    duration: int = 300,
  ) -> None:
    if not self._frames:
      return
    self._frames[0].save(
      path,
      save_all=True,
      append_images=self._frames[1:],
      duration=duration,
      loop=0,
    )

  def play_animation(self) -> None:
    """Play frames as HTML5 video in Jupyter."""
    from matplotlib import animation

    if not self._frames:
      return

    frames_np = [
      np.array(f) for f in self._frames
    ]
    dpi = 72.0
    h, w = frames_np[0].shape[:2]
    fig = plt.figure(
      figsize=(w / dpi, h / dpi), dpi=dpi
    )
    im = plt.figimage(frames_np[0])

    def _animate(i: int):
      im.set_array(frames_np[i])
      return (im,)

    anim = animation.FuncAnimation(
      fig, _animate,
      frames=len(frames_np),
      interval=300,
      repeat=True,
    )

    try:
      from IPython.display import HTML, display
      display(HTML(anim.to_html5_video()))
    except ImportError:
      plt.show()
