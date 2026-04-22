from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from truck_carton.domain.models import Carton, PlacementInfo
from truck_carton.packing.rotation import (
  Rotation,
  apply_rotation,
  get_valid_rotations,
)
from truck_carton.packing.space3d import Space3D


@dataclass
class PlacementCandidate:
  truck_id: int
  x: int
  y: int
  z: int
  rotation: Rotation
  oriented_dims: tuple[int, int, int]


class PlacementValidator:
  """Validates carton placements against physical
  constraints and enumerates valid positions."""

  def __init__(
    self,
    space: Space3D,
    placed_cartons: dict[int, PlacementInfo],
  ) -> None:
    self._space = space
    self._placed = placed_cartons
    self._carton_lookup: dict[int, Carton] = {}

  def check_bounds(
    self,
    x: int, y: int, z: int,
    dl: int, dw: int, dh: int,
  ) -> bool:
    return (
      x >= 0
      and y >= 0
      and z >= 0
      and x + dl <= self._space.length
      and y + dw <= self._space.width
      and z + dh <= self._space.height
    )

  def check_no_overlap(
    self,
    x: int, y: int, z: int,
    dl: int, dw: int, dh: int,
  ) -> bool:
    return self._space.can_place(x, y, z, dl, dw, dh)

  def check_gravity_support(
    self,
    x: int, y: int, z: int,
    dl: int, dw: int, dh: int,
  ) -> bool:
    if z == 0:
      return True
    below = self._space.grid[
      x:x + dl, y:y + dw, z - 1
    ]
    return bool(np.all(below != 0))

  def check_fragility(
    self,
    carton: Carton,
    x: int, y: int, z: int,
    dl: int, dw: int, dh: int,
  ) -> bool:
    if not carton.is_fragile:
      return self._check_no_fragile_below(
        x, y, z, dl, dw
      )
    return self._check_no_nonfragile_above(
      x, y, z, dl, dw, dh
    )

  def _check_no_fragile_below(
    self,
    x: int, y: int, z: int,
    dl: int, dw: int,
  ) -> bool:
    if z == 0:
      return True
    column = self._space.grid[
      x:x + dl, y:y + dw, :z
    ]
    unique_ids = np.unique(column)
    unique_ids = unique_ids[unique_ids != 0]
    for cid in unique_ids:
      carton_below = self._carton_lookup.get(
        int(cid)
      )
      if (
        carton_below is not None
        and carton_below.is_fragile
      ):
        return False
    return True

  def _check_no_nonfragile_above(
    self,
    x: int, y: int, z: int,
    dl: int, dw: int, dh: int,
  ) -> bool:
    top_z = z + dh
    if top_z >= self._space.height:
      return True
    above = self._space.grid[
      x:x + dl, y:y + dw, top_z:
    ]
    unique_ids = np.unique(above)
    unique_ids = unique_ids[unique_ids != 0]
    for cid in unique_ids:
      carton_above = self._carton_lookup.get(
        int(cid)
      )
      if (
        carton_above is not None
        and not carton_above.is_fragile
      ):
        return False
    return True

  def check_weight(
    self,
    carton_weight: float,
    current_weight: float,
    max_weight: float,
  ) -> bool:
    return current_weight + carton_weight <= max_weight

  def validate_placement(
    self,
    carton: Carton,
    x: int, y: int, z: int,
    dl: int, dw: int, dh: int,
    current_weight: float,
    max_weight: float,
    all_cartons: list[Carton] | None = None,
  ) -> tuple[bool, list[str]]:
    if all_cartons:
      self._carton_lookup = {
        c.carton_id: c for c in all_cartons
      }
    violations: list[str] = []
    if not self.check_bounds(x, y, z, dl, dw, dh):
      violations.append('out_of_bounds')
    if not self.check_no_overlap(x, y, z, dl, dw, dh):
      violations.append('overlap')
    if not self.check_gravity_support(
      x, y, z, dl, dw, dh
    ):
      violations.append('no_support')
    if not self.check_fragility(
      carton, x, y, z, dl, dw, dh
    ):
      violations.append('fragility_violation')
    if not self.check_weight(
      carton.weight, current_weight, max_weight
    ):
      violations.append('weight_exceeded')
    return (len(violations) == 0, violations)

  def find_valid_positions(
    self,
    carton: Carton,
    current_weight: float,
    max_weight: float,
    all_cartons: list[Carton] | None = None,
  ) -> list[tuple[int, int, int, Rotation]]:
    self._carton_lookup = {}
    if all_cartons:
      self._carton_lookup = {
        c.carton_id: c for c in all_cartons
      }

    if not self.check_weight(
      carton.weight, current_weight, max_weight
    ):
      return []

    height_map = self._space.get_height_map()
    candidates: list[
      tuple[int, int, int, Rotation]
    ] = []

    for rotation in get_valid_rotations(
      carton.is_fragile
    ):
      dl, dw, dh = apply_rotation(
        carton.length,
        carton.width,
        carton.height,
        rotation,
      )

      if (
        dl > self._space.length
        or dw > self._space.width
        or dh > self._space.height
      ):
        continue

      for x in range(
        self._space.length - dl + 1
      ):
        for y in range(
          self._space.width - dw + 1
        ):
          footprint = height_map[
            x:x + dl, y:y + dw
          ]
          z = int(footprint.max())

          if footprint.min() != z:
            continue

          if z + dh > self._space.height:
            continue

          if not self.check_fragility(
            carton, x, y, z, dl, dw, dh
          ):
            continue

          candidates.append(
            (x, y, z, rotation)
          )

    return candidates
