#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Adversarial cycle-3 regression tests.'''

import inspect

import numpy as np
import pytest

import truck_carton.domain.data_generator as data_mod
from truck_carton.config import (
  AppConfig,
  CurriculumConfig,
  CurriculumStage,
)
from truck_carton.curriculum.manager import (
  CurriculumManager,
)
from truck_carton.domain.data_generator import (
  DataGenerator,
)
from truck_carton.domain.models import (
  Carton,
  PlacementInfo,
)
from truck_carton.packing.placement import (
  PlacementValidator,
)
from truck_carton.packing.rotation import (
  Rotation,
  apply_rotation,
)
from truck_carton.packing.space3d import Space3D
from truck_carton.reward.calculator import (
  EnvironmentState,
)
from truck_carton.reward.completion import (
  CompletionReward,
)


def _stage(window: int = 2) -> CurriculumStage:
  '''Stage helper.'''
  return CurriculumStage(
    'toy', 2, 2, 10, 1, 5, 5, 0.5, window,
  )


def _state(placed, cartons, total):
  '''State helper.'''
  return EnvironmentState(
    trucks=[],
    spaces=[],
    placed_cartons=placed,
    unplaced_cartons=[],
    all_cartons=cartons,
    stores=[],
    current_weights=[],
    step_carton=None,
    step_placement=None,
    is_terminal=False,
    total_cartons=total,
  )


# F-02
def test_space3d_place_rejects_overlap():
  '''Place must not silently overwrite.'''
  s = Space3D(4, 4, 4)
  s.place(5, 0, 0, 0, 2, 2, 2)
  with pytest.raises(ValueError):
    s.place(9, 0, 0, 0, 1, 1, 1)
  assert s.grid[0, 0, 0] == 5


def test_space3d_place_rejects_partial_overlap():
  '''Place must not overwrite on partial
  overlap; gravity remains the validator's
  job (see placement.py).'''
  s = Space3D(4, 4, 4)
  s.place(5, 0, 0, 0, 2, 2, 2)
  with pytest.raises(ValueError):
    s.place(9, 1, 1, 1, 2, 2, 1)
  assert s.grid[0, 0, 0] == 5


# F-09
def test_validate_placement_no_stale_lookup():
  '''Second call without all_cartons must not
  reuse stale lookup.'''
  space = Space3D(4, 4, 4)
  fragile = Carton(1, 2, 2, 1, 5.0, True, 3, 0)
  non_fragile = Carton(2, 2, 2, 1, 10.0, False, 1, 0)
  space.place(1, 0, 0, 0, 2, 2, 1)
  placed = {
    1: PlacementInfo(0, (0, 0, 0), (2, 2, 1), Rotation.LWH),
  }
  v = PlacementValidator(space, placed)
  valid, violations = v.validate_placement(
    non_fragile, 0, 0, 1, 2, 2, 1, 5.0, 500.0,
    all_cartons=[fragile, non_fragile],
  )
  assert valid is False
  assert 'fragility_violation' in violations
  valid2, violations2 = v.validate_placement(
    non_fragile, 0, 0, 1, 2, 2, 1, 5.0, 500.0,
  )
  assert valid2 is True
  assert 'fragility_violation' not in violations2


# F-10
def test_apply_rotation_bad_rotation():
  '''Bad rotation raises ValueError not KeyError.'''
  with pytest.raises(ValueError):
    apply_rotation(1, 2, 3, 99)
  with pytest.raises(ValueError):
    apply_rotation(1, 2, 3, -1)


# F-07
def test_completion_excludes_orphan():
  '''Orphan placed ids do not count.'''
  placed = {
    99: PlacementInfo(0, (0, 0, 0), (1, 1, 1), Rotation.LWH),
  }
  state = _state(placed, [], 1)
  assert CompletionReward().compute(state) == 0.0


def test_completion_counts_only_known():
  '''Known placed ids still count.'''
  cartons = [
    Carton(1, 1, 1, 1, 1.0, False, 1, 0),
    Carton(2, 1, 1, 1, 1.0, False, 1, 0),
  ]
  placed = {
    1: PlacementInfo(0, (0, 0, 0), (1, 1, 1), Rotation.LWH),
    99: PlacementInfo(0, (1, 0, 0), (1, 1, 1), Rotation.LWH),
  }
  state = _state(placed, cartons, 2)
  assert CompletionReward().compute(state) == 0.25


# F-06
def test_curriculum_rejects_nonfinite_reward():
  '''record_episode rejects inf/nan.'''
  cm = CurriculumManager(CurriculumConfig(stages=(_stage(), _stage())))
  with pytest.raises(ValueError):
    cm.record_episode(float('inf'))
  with pytest.raises(ValueError):
    cm.record_episode(float('nan'))


# F-11
def test_curriculum_rejects_empty_stages():
  '''Empty stages rejected.'''
  with pytest.raises(ValueError):
    CurriculumManager(CurriculumConfig(stages=()))


def test_curriculum_rejects_bad_window():
  '''Window < 1 rejected.'''
  with pytest.raises(ValueError):
    CurriculumManager(
      CurriculumConfig(stages=(_stage(window=0),))
    )
  with pytest.raises(ValueError):
    CurriculumManager(
      CurriculumConfig(stages=(_stage(window=-1),))
    )


# F-08
def test_generator_no_duplicate_sentinel_stores():
  '''Small grid must not emit duplicate
  (0, 0) store positions.'''
  gen = DataGenerator(AppConfig(), np.random.default_rng(42))
  data = gen.generate(1, 8, 1, 1, 5, 5)
  positions = [s.position for s in data.stores]
  assert len(positions) == len(set(positions))
  assert positions.count((0, 0)) <= 1
  store_ids = {s.store_id for s in data.stores}
  for carton in data.cartons:
    assert carton.destination_store_id in store_ids


def test_generator_no_phantom_store_zero():
  '''Zero stores with cartons must not
  emit phantom store 0 destinations.'''
  gen = DataGenerator(AppConfig(), np.random.default_rng(42))
  with pytest.raises(ValueError):
    gen.generate(1, 0, 3, 1, 5, 5)


# Reviewer #8
def test_no_road_network_strategy_protocol():
  '''Protocol removed, direct MSTRoadStrategy.'''
  assert not hasattr(data_mod, 'RoadNetworkStrategy')
  params = inspect.signature(
    data_mod.DataGenerator.__init__,
  ).parameters
  assert 'road_strategy' not in params


# Reviewer #10
def test_validator_has_no_dead_placed_attr():
  '''PlacementValidator must not keep
  dead _placed attribute.'''
  v = PlacementValidator(Space3D(4, 4, 4), {})
  assert not hasattr(v, '_placed')
