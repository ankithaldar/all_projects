#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Three-dimensional occupancy space for packing.'''

from __future__ import annotations

import numpy as np

from truck_carton.config import EnvironmentConfig


def _validate_place_args(
  x: int, y: int, z: int, dl: int, dw: int, dh: int,
) -> None:
  '''Reject negative coords and non-positive dims.'''
  for name, value in (
    ('x', x), ('y', y), ('z', z),
    ('dl', dl), ('dw', dw), ('dh', dh),
  ):
    if isinstance(value, bool) or not isinstance(
      value, (int, np.integer),
    ):
      raise ValueError(f'{name} must be an integer (got {value!r})')
  if x < 0 or y < 0 or z < 0:
    raise ValueError(
      f'coordinates must be >= 0 (got {(x, y, z)})'
    )
  if dl <= 0 or dw <= 0 or dh <= 0:
    raise ValueError(
      f'dims must be > 0 (got {(dl, dw, dh)})'
    )


class Space3D:
  '''3D occupancy grid for a single truck interior.'''

  def __init__(self, length: int, width: int, height: int) -> None:
    '''Initialize the instance.'''
    caps = EnvironmentConfig()
    for name, value, cap in (
      ('length', length, caps.max_truck_length),
      ('width', width, caps.max_truck_width),
      ('height', height, caps.max_truck_height),
    ):
      if isinstance(value, bool) or not isinstance(
        value, (int, np.integer),
      ):
        raise ValueError(f'{name} must be an integer')
      if value <= 0 or value > cap:
        raise ValueError(f'{name} must be in [1, {cap}] (got {value})')
    self.length = length
    self.width = width
    self.height = height
    self.grid = np.zeros((length, width, height), dtype=np.int16)

  @property
  def shape(self) -> tuple[int, int, int]:
    '''Shape.'''
    return (self.length, self.width, self.height)

  def can_place(
    self,
    x: int,
    y: int,
    z: int,
    dl: int,
    dw: int,
    dh: int,
  ) -> bool:
    '''Can place.'''
    _validate_place_args(x, y, z, dl, dw, dh)
    if x + dl > self.length or y + dw > self.width or z + dh > self.height:
      return False
    region = self.grid[x : x + dl, y : y + dw, z : z + dh]
    return bool(np.all(region == 0))

  def place(
    self,
    carton_id: int,
    x: int,
    y: int,
    z: int,
    dl: int,
    dw: int,
    dh: int,
  ) -> None:
    '''Place a carton, rejecting overlap.

    Gravity/support is enforced by
    PlacementValidator; place() is the
    low-level primitive it builds on and
    guarantees no silent overwrite.'''
    if carton_id <= 0:
      raise ValueError(
        f'carton_id must be > 0 (got {carton_id})'
        f'; 0 is reserved for empty cells'
      )
    _validate_place_args(x, y, z, dl, dw, dh)
    if (
      x + dl > self.length
      or y + dw > self.width
      or z + dh > self.height
    ):
      raise ValueError('placement exceeds space bounds')
    region = self.grid[x : x + dl, y : y + dw, z : z + dh]
    if np.any(region != 0):
      raise ValueError('placement overlaps existing carton')
    self.grid[x : x + dl, y : y + dw, z : z + dh] = carton_id

  def remove(self, carton_id: int) -> None:
    '''Remove.'''
    self.grid[self.grid == carton_id] = 0

  def get_occupancy_ratio(self) -> float:
    '''Get occupancy ratio.'''
    total = self.grid.size
    if total == 0:
      return 0.0
    return float(np.count_nonzero(self.grid)) / total

  def get_height_map(self) -> np.ndarray:
    '''Get height map.'''
    occupied = self.grid != 0
    height_map = np.zeros((self.length, self.width), dtype=np.int32)
    for z in range(self.height - 1, -1, -1):
      mask = occupied[:, :, z] & (height_map == 0)
      height_map[mask] = z + 1
    return height_map

  def get_normalized_grid(self, max_carton_id: int) -> np.ndarray:
    '''Get normalized grid.'''
    if max_carton_id == 0:
      return np.zeros_like(self.grid, dtype=np.float32)
    return self.grid.astype(np.float32) / max_carton_id

  def get_occupied_volume(self) -> int:
    '''Get occupied volume.'''
    return int(np.count_nonzero(self.grid))

  def copy(self) -> Space3D:
    '''Copy.'''
    new = Space3D(self.length, self.width, self.height)
    new.grid = self.grid.copy()
    return new
