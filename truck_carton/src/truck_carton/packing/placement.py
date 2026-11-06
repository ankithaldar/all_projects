#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Placement validation and candidate search.'''

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from truck_carton.domain.models import Carton, PlacementInfo
from truck_carton.packing.rotation import (
  Rotation,
  apply_rotation,
  get_valid_rotations,
)
from truck_carton.packing.space3d import Space3D


def _require_int_args(args: tuple) -> None:
  '''Reject bool/non-int coords and dims with ValueError.'''
  for value in args:
    if isinstance(value, bool) or not isinstance(
      value, (int, np.integer),
    ):
      raise ValueError(f'must be an integer (got {value!r})')


@dataclass
class PlacementCandidate:
  '''Placement Candidate.'''

  truck_id: int
  x: int
  y: int
  z: int
  rotation: Rotation
  oriented_dims: tuple[int, int, int]


class PlacementValidator:
  '''Validates carton placements against physical
  constraints and enumerates valid positions.'''

  def __init__(
    self,
    space: Space3D,
    placed_cartons: dict[int, PlacementInfo],
  ) -> None:
    '''Initialize the instance.

    placed_cartons is accepted for caller
    compatibility but not retained; it was
    dead state (validation reads Space3D).'''
    _ = placed_cartons
    self._space = space
    self._carton_lookup: dict[int, Carton] = {}

  def check_bounds(
    self,
    x: int,
    y: int,
    z: int,
    dl: int,
    dw: int,
    dh: int,
  ) -> bool:
    '''Check bounds.'''
    _require_int_args((x, y, z, dl, dw, dh))
    return (
      dl > 0
      and dw > 0
      and dh > 0
      and x >= 0
      and y >= 0
      and z >= 0
      and x + dl <= self._space.length
      and y + dw <= self._space.width
      and z + dh <= self._space.height
    )

  def check_no_overlap(
    self,
    x: int,
    y: int,
    z: int,
    dl: int,
    dw: int,
    dh: int,
  ) -> bool:
    '''Check no overlap.'''
    if not self.check_bounds(x, y, z, dl, dw, dh):
      return False
    return self._space.can_place(x, y, z, dl, dw, dh)

  def check_gravity_support(
    self,
    x: int,
    y: int,
    z: int,
    dl: int,
    dw: int,
    unused_dh: int,
  ) -> bool:
    '''Check gravity support.'''
    if not self.check_bounds(x, y, z, dl, dw, unused_dh):
      return False
    if z == 0:
      return True
    below = self._space.grid[x : x + dl, y : y + dw, z - 1]
    return bool(np.all(below != 0))

  def check_fragility(
    self,
    carton: Carton,
    x: int,
    y: int,
    z: int,
    dl: int,
    dw: int,
    dh: int,
  ) -> bool:
    '''Check fragility.'''
    if not self.check_bounds(x, y, z, dl, dw, dh):
      return False
    if not carton.is_fragile:
      return self._check_no_fragile_below(x, y, z, dl, dw)
    return self._check_no_nonfragile_above(x, y, z, dl, dw, dh)

  def _check_no_fragile_below(
    self,
    x: int,
    y: int,
    z: int,
    dl: int,
    dw: int,
  ) -> bool:
    '''Check no fragile below.'''
    if z == 0:
      return True
    column = self._space.grid[x : x + dl, y : y + dw, :z]
    unique_ids = np.unique(column)
    unique_ids = unique_ids[unique_ids != 0]
    for cid in unique_ids:
      carton_below = self._carton_lookup.get(int(cid))
      if carton_below is not None and carton_below.is_fragile:
        return False
    return True

  def _check_no_nonfragile_above(
    self,
    x: int,
    y: int,
    z: int,
    dl: int,
    dw: int,
    dh: int,
  ) -> bool:
    '''Check no nonfragile above.'''
    top_z = z + dh
    if top_z >= self._space.height:
      return True
    above = self._space.grid[x : x + dl, y : y + dw, top_z:]
    unique_ids = np.unique(above)
    unique_ids = unique_ids[unique_ids != 0]
    for cid in unique_ids:
      carton_above = self._carton_lookup.get(int(cid))
      if carton_above is not None and not carton_above.is_fragile:
        return False
    return True

  def check_weight(
    self,
    carton_weight: float,
    current_weight: float,
    max_weight: float,
  ) -> bool:
    '''Check weight.'''
    for value in (carton_weight, current_weight, max_weight):
      if not math.isfinite(float(value)):
        return False
    if carton_weight < 0 or current_weight < 0 or max_weight < 0:
      return False
    return current_weight + carton_weight <= max_weight

  def validate_placement(
    self,
    carton: Carton,
    x: int,
    y: int,
    z: int,
    dl: int,
    dw: int,
    dh: int,
    current_weight: float,
    max_weight: float,
    all_cartons: list[Carton] | None = None,
  ) -> tuple[bool, list[str]]:
    '''Validate placement.

    Warning: omitting all_cartons resets the
    lookup and SKIPS fragility validation
    (fail-open). Unknown ids count as neither
    fragile nor non-fragile, so a non-fragile
    carton above a fragile one (and the
    reverse) passes silently. Callers needing
    fragility MUST pass all_cartons; the env
    always does. Bounds, overlap, gravity,
    and weight checks still run.

    Args:
      carton: Carton to validate.
      x, y, z: Placement position.
      dl, dw, dh: Oriented dimensions.
      current_weight: Current load weight.
      max_weight: Maximum allowed weight.
      all_cartons: Cartons for fragility
       lookup; omit only if fragility is
       not needed (see warning above).

    Returns:
      Tuple of (valid, violations).
    '''
    _require_int_args(
      (carton.length, carton.width, carton.height)
    )
    self._carton_lookup = {}
    if all_cartons:
      self._carton_lookup = {c.carton_id: c for c in all_cartons}
    violations: list[str] = []
    if not self.check_bounds(x, y, z, dl, dw, dh):
      violations.append('out_of_bounds')
    else:
      if not self.check_no_overlap(x, y, z, dl, dw, dh):
        violations.append('overlap')
      if not self.check_gravity_support(x, y, z, dl, dw, dh):
        violations.append('no_support')
      if not self.check_fragility(carton, x, y, z, dl, dw, dh):
        violations.append('fragility_violation')
    if not self.check_weight(carton.weight, current_weight, max_weight):
      violations.append('weight_exceeded')
    return (len(violations) == 0, violations)

  def find_valid_positions(
    self,
    carton: Carton,
    current_weight: float,
    max_weight: float,
    all_cartons: list[Carton] | None = None,
  ) -> list[tuple[int, int, int, Rotation]]:
    '''Find valid positions.

    Warning: omitting all_cartons resets the
    lookup and SKIPS fragility validation
    (fail-open), like validate_placement.
    Callers needing fragility MUST pass
    all_cartons; the env always does.

    Args:
      carton: Carton to place.
      current_weight: Current load weight.
      max_weight: Maximum allowed weight.
      all_cartons: Cartons for fragility
       lookup; omit only if fragility is
       not needed (see warning above).

    Returns:
      List of (x, y, z, rotation) positions.
    '''
    _require_int_args(
      (carton.length, carton.width, carton.height)
    )
    self._carton_lookup = {}
    if all_cartons:
      self._carton_lookup = {c.carton_id: c for c in all_cartons}

    if carton.length <= 0 or carton.width <= 0 or carton.height <= 0:
      return []
    if not self.check_weight(carton.weight, current_weight, max_weight):
      return []

    height_map = self._space.get_height_map()
    candidates: list[tuple[int, int, int, Rotation]] = []

    for rotation in get_valid_rotations(carton.is_fragile):
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

      for x in range(self._space.length - dl + 1):
        for y in range(self._space.width - dw + 1):
          footprint = height_map[x : x + dl, y : y + dw]
          z = int(footprint.max())

          if footprint.min() != z:
            continue

          if z + dh > self._space.height:
            continue

          if not self.check_fragility(carton, x, y, z, dl, dw, dh):
            continue

          candidates.append((x, y, z, rotation))

    return candidates
