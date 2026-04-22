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
    PlacementInfo,
    Truck,
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
