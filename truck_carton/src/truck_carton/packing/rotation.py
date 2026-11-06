#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Carton rotation definitions and helpers.'''

from enum import IntEnum


class Rotation(IntEnum):
  '''Rotation.'''

  LWH = 0
  LHW = 1
  WLH = 2
  WHL = 3
  HLW = 4
  HWL = 5


_ROTATION_PERMUTATIONS: dict[Rotation, tuple[int, int, int]] = {
  Rotation.LWH: (0, 1, 2),
  Rotation.LHW: (0, 2, 1),
  Rotation.WLH: (1, 0, 2),
  Rotation.WHL: (1, 2, 0),
  Rotation.HLW: (2, 0, 1),
  Rotation.HWL: (2, 1, 0),
}

FRAGILE_ROTATIONS: tuple[Rotation, ...] = (
  Rotation.LWH,
  Rotation.WLH,
)
ALL_ROTATIONS: tuple[Rotation, ...] = tuple(Rotation)


def apply_rotation(
  length: int, w: int, h: int, rotation: Rotation
) -> tuple[int, int, int]:
  '''Apply rotation.'''
  if isinstance(rotation, bool) or not isinstance(rotation, int):
    raise ValueError(f'invalid rotation: {rotation!r}')
  dims = (length, w, h)
  try:
    perm = _ROTATION_PERMUTATIONS[rotation]
  except (KeyError, TypeError):
    raise ValueError(f'invalid rotation: {rotation!r}') from None
  return (dims[perm[0]], dims[perm[1]], dims[perm[2]])


def get_valid_rotations(
  is_fragile: bool,
) -> tuple[Rotation, ...]:
  '''Get valid rotations.'''
  return FRAGILE_ROTATIONS if is_fragile else ALL_ROTATIONS
