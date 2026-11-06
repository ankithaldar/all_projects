#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Final microfix tests for F-09 and curriculum stage.'''

import pytest

from truck_carton.config import AppConfig
from truck_carton.domain.models import (
  Carton,
  PlacementInfo,
)
from truck_carton.env.packing_env import (
  TruckCartonPackingEnv,
)
from truck_carton.packing.placement import (
  PlacementValidator,
)
from truck_carton.packing.rotation import Rotation
from truck_carton.packing.space3d import Space3D


def _validator_with_fragile_below():
  '''Validator with fragile carton at origin.'''
  space = Space3D(4, 4, 4)
  fragile = Carton(1, 2, 2, 1, 5.0, True, 3, 0)
  space.place(1, 0, 0, 0, 2, 2, 1)
  placed = {
    1: PlacementInfo(0, (0, 0, 0), (2, 2, 1), Rotation.LWH),
  }
  validator = PlacementValidator(space, placed)
  return validator, fragile


def test_f09_omit_skips_fragility_pass_flags():
  '''Omitting all_cartons fails open; passing
  it flags non-fragile above fragile.'''
  validator, fragile = _validator_with_fragile_below()
  non_fragile = Carton(2, 2, 2, 1, 10.0, False, 1, 0)
  omitted = validator.validate_placement(
    non_fragile, 0, 0, 1, 2, 2, 1, 5.0, 500.0,
  )
  assert omitted == (True, [])
  flagged = validator.validate_placement(
    non_fragile,
    0,
    0,
    1,
    2,
    2,
    1,
    5.0,
    500.0,
    all_cartons=[fragile, non_fragile],
  )
  assert flagged == (False, ['fragility_violation'])


def test_f09_omit_skips_above_check_too():
  '''Omitting all_cartons also skips the
  non-fragile-above check for fragile.'''
  space = Space3D(4, 4, 4)
  non_fragile = Carton(1, 2, 2, 1, 10.0, False, 1, 0)
  space.place(1, 0, 0, 1, 2, 2, 1)
  placed = {
    1: PlacementInfo(0, (0, 0, 1), (2, 2, 1), Rotation.LWH),
  }
  validator = PlacementValidator(space, placed)
  fragile = Carton(2, 2, 2, 1, 5.0, True, 3, 0)
  omitted = validator.validate_placement(
    fragile, 0, 0, 0, 2, 2, 1, 10.0, 500.0,
  )
  assert omitted == (True, [])
  flagged = validator.validate_placement(
    fragile,
    0,
    0,
    0,
    2,
    2,
    1,
    10.0,
    500.0,
    all_cartons=[fragile, non_fragile],
  )
  assert flagged == (False, ['fragility_violation'])


def test_f09_validate_placement_docstring_warns():
  '''Docstring warns omitting all_cartons
  skips fragility validation (fail-open).'''
  doc = PlacementValidator.validate_placement.__doc__ or ''
  assert 'all_cartons' in doc
  assert 'SKIPS fragility' in doc
  assert 'fail-open' in doc
  assert 'MUST pass all_cartons' in doc


@pytest.mark.parametrize('bad', [True, False, 1.9, '3', None])
def test_set_curriculum_stage_rejects_non_int(bad):
  '''set_curriculum_stage rejects non-int.'''
  env = TruckCartonPackingEnv(config=AppConfig())
  with pytest.raises(ValueError):
    env.set_curriculum_stage(bad)


@pytest.mark.parametrize('bad', [True, False, 1.9, '3', None])
def test_constructor_rejects_non_int_stage(bad):
  '''Constructor rejects non-int stage.'''
  with pytest.raises(ValueError):
    TruckCartonPackingEnv(
      config=AppConfig(), curriculum_stage=bad,
    )


def test_set_curriculum_stage_clamps_like_ctor():
  '''Setter clamps out-of-range like ctor.'''
  last = len(AppConfig().curriculum.stages) - 1
  env = TruckCartonPackingEnv(config=AppConfig())
  env.set_curriculum_stage(-5)
  assert env.curriculum_stage == 0
  env.set_curriculum_stage(999)
  assert env.curriculum_stage == last
  env.set_curriculum_stage(2)
  assert env.curriculum_stage == 2
  ctor_low = TruckCartonPackingEnv(
    config=AppConfig(), curriculum_stage=-5,
  )
  ctor_high = TruckCartonPackingEnv(
    config=AppConfig(), curriculum_stage=999,
  )
  assert ctor_low.curriculum_stage == 0
  assert ctor_high.curriculum_stage == last
