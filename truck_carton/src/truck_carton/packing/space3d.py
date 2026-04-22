from __future__ import annotations

import numpy as np


class Space3D:
  """3D occupancy grid for a single truck interior."""

  def __init__(
    self, length: int, width: int, height: int
  ) -> None:
    self.length = length
    self.width = width
    self.height = height
    self.grid = np.zeros(
      (length, width, height), dtype=np.int16
    )

  @property
  def shape(self) -> tuple[int, int, int]:
    return (self.length, self.width, self.height)

  def can_place(
    self,
    x: int, y: int, z: int,
    dl: int, dw: int, dh: int,
  ) -> bool:
    if (
      x + dl > self.length
      or y + dw > self.width
      or z + dh > self.height
    ):
      return False
    region = self.grid[
      x:x + dl, y:y + dw, z:z + dh
    ]
    return bool(np.all(region == 0))

  def place(
    self,
    carton_id: int,
    x: int, y: int, z: int,
    dl: int, dw: int, dh: int,
  ) -> None:
    self.grid[
      x:x + dl, y:y + dw, z:z + dh
    ] = carton_id

  def remove(self, carton_id: int) -> None:
    self.grid[self.grid == carton_id] = 0

  def get_occupancy_ratio(self) -> float:
    total = self.grid.size
    if total == 0:
      return 0.0
    return float(np.count_nonzero(self.grid)) / total

  def get_height_map(self) -> np.ndarray:
    occupied = self.grid != 0
    height_map = np.zeros(
      (self.length, self.width), dtype=np.int32
    )
    for z in range(self.height - 1, -1, -1):
      mask = occupied[:, :, z] & (height_map == 0)
      height_map[mask] = z + 1
    return height_map

  def get_normalized_grid(
    self, max_carton_id: int
  ) -> np.ndarray:
    if max_carton_id == 0:
      return np.zeros_like(
        self.grid, dtype=np.float32
      )
    return self.grid.astype(np.float32) / max_carton_id

  def get_occupied_volume(self) -> int:
    return int(np.count_nonzero(self.grid))

  def copy(self) -> Space3D:
    new = Space3D(self.length, self.width, self.height)
    new.grid = self.grid.copy()
    return new
