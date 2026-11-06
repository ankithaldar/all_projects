#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Security input validation tests S-03/S-04/S-05.'''

import math

import numpy as np
import pytest

from truck_carton.config import AppConfig, RewardWeights
from truck_carton.domain.data_generator import (
  DataGenerator,
)
from truck_carton.domain.models import (
  Carton,
  PlacementInfo,
  Store,
  Truck,
  Warehouse,
)
from truck_carton.packing.placement import (
  PlacementValidator,
)
from truck_carton.packing.rotation import Rotation
from truck_carton.packing.space3d import Space3D
from truck_carton.reward.calculator import (
  EnvironmentState,
  RewardCalculator,
)
from truck_carton.reward.displacement import (
  DisplacementReward,
)
from truck_carton.reward.fragility import (
  FragilityReward,
)
from truck_carton.reward.grouping import GroupingReward
from truck_carton.reward.priority import PriorityReward
from truck_carton.reward.travel_distance import (
  TravelDistanceReward,
)
from truck_carton.reward.utilization import (
  UtilizationReward,
)
from truck_carton.reward.weight import WeightReward


def _state(placed=None, cartons=None, weights=None, **kw):
  '''Make reward state with orphan-safe defaults.'''
  args = {
    'trucks': [Truck(0, 4, 4, 4, 500.0, [0, 1])],
    'spaces': [Space3D(4, 4, 4)],
    'placed_cartons': placed or {},
    'unplaced_cartons': [],
    'all_cartons': cartons or [],
    'stores': [Store(0, 0), Store(1, 1)],
    'current_weights': weights if weights is not None else [0.0],
    'step_carton': None,
    'step_placement': None,
    'is_terminal': False,
    'total_cartons': kw.pop('total', 4),
  }
  args.update(kw)
  return EnvironmentState(**args)


def _orphan_state():
  '''State whose placed id 99 is absent from cartons.'''
  placed = {
    99: PlacementInfo(0, (0, 0, 0), (1, 1, 1), Rotation.LWH),
  }
  return _state(placed=placed, cartons=[], total=1)


# S-04 Space3D bounds.


@pytest.mark.parametrize('coord', [( -1, 0, 0), (0, -1, 0), (0, 0, -1)])
def test_can_place_negative_coord(coord):
  '''Negative coords raise ValueError, not True.'''
  s = Space3D(4, 4, 4)
  with pytest.raises(ValueError):
    s.can_place(*coord, 1, 1, 1)


@pytest.mark.parametrize('dims', [(0, 1, 1), (1, 0, 1), (1, 1, -1)])
def test_can_place_nonpositive_dims(dims):
  '''Non-positive dims raise ValueError.'''
  s = Space3D(4, 4, 4)
  with pytest.raises(ValueError):
    s.can_place(0, 0, 0, *dims)


def test_place_negative_coord_raises_no_mutation():
  '''place() with negative coord raises, writes nothing.'''
  s = Space3D(4, 4, 4)
  with pytest.raises(ValueError):
    s.place(7, -1, 0, 0, 1, 1, 1)
  assert s.get_occupied_volume() == 0


def test_place_nonpositive_dims_raises():
  '''place() with zero dims raises ValueError.'''
  s = Space3D(4, 4, 4)
  with pytest.raises(ValueError):
    s.place(7, 0, 0, 0, 0, 1, 1)
  assert s.get_occupied_volume() == 0


def test_space3d_rejects_zero_and_absurd_dims():
  '''Constructor rejects zero and over-cap dims.'''
  with pytest.raises(ValueError):
    Space3D(0, 4, 4)
  with pytest.raises(ValueError):
    Space3D(17, 1, 1)
  with pytest.raises(ValueError):
    Space3D(1, 9, 1)
  with pytest.raises(ValueError):
    Space3D(1, 1, 9)


# S-04 domain dataclass validation.


def test_carton_rejects_bad_dims_weights_ids():
  '''Carton rejects non-positive dims, bad weights/ids.'''
  base = dict(
    carton_id=1, length=1, width=1, height=1, weight=1.0,
    is_fragile=False, priority=1, destination_store_id=0,
  )
  for patch in (
    {'length': 0}, {'width': -1}, {'height': 0},
    {'weight': -1.0}, {'weight': float('nan')},
    {'weight': float('inf')}, {'carton_id': 0},
    {'carton_id': -1}, {'destination_store_id': -1},
    {'origin_warehouse_id': -1},
  ):
    with pytest.raises(ValueError):
      Carton(**{**base, **patch})


def test_truck_rejects_bad_dims_weights_ids():
  '''Truck rejects non-positive dims, bad weights/ids.'''
  base = dict(
    truck_id=0, length=4, width=4, height=4,
    max_weight=500.0, route=[0],
  )
  for patch in (
    {'length': 0}, {'width': -2}, {'height': 0},
    {'max_weight': -1.0}, {'max_weight': float('nan')},
    {'max_weight': float('inf')}, {'truck_id': -1},
  ):
    with pytest.raises(ValueError):
      Truck(**{**base, **patch})


def test_warehouse_store_reject_negative_ids():
  '''Warehouse/Store reject negative ids.'''
  with pytest.raises(ValueError):
    Warehouse(warehouse_id=-1, position=(0, 0))
  with pytest.raises(ValueError):
    Store(store_id=-1, route_position=0)
  with pytest.raises(ValueError):
    Store(store_id=0, route_position=-1)


def test_placement_info_rejects_bad_values():
  '''PlacementInfo rejects negative pos, zero dims.'''
  with pytest.raises(ValueError):
    PlacementInfo(-1, (0, 0, 0), (1, 1, 1), Rotation.LWH)
  with pytest.raises(ValueError):
    PlacementInfo(0, (-1, 0, 0), (1, 1, 1), Rotation.LWH)
  with pytest.raises(ValueError):
    PlacementInfo(0, (0, 0, 0), (0, 1, 1), Rotation.LWH)


# S-04 validator.


def test_validator_negative_dims_rejected_no_crash():
  '''Validator rejects negative dims, does not raise.'''
  v = PlacementValidator(Space3D(4, 4, 4), {})
  assert v.check_bounds(0, 0, 0, -2, 2, 2) is False
  assert v.check_bounds(0, 0, 0, 0, 2, 2) is False
  carton = Carton(1, 1, 1, 1, 1.0, False, 1, 0)
  valid, violations = v.validate_placement(
    carton, 0, 0, 0, -2, 2, 2, 0.0, 500.0,
  )
  assert valid is False
  assert 'out_of_bounds' in violations


def test_validator_bad_weights_rejected():
  '''Validator rejects negative/NaN weights.'''
  v = PlacementValidator(Space3D(4, 4, 4), {})
  assert v.check_weight(-1.0, 0.0, 500.0) is False
  assert v.check_weight(float('nan'), 0.0, 500.0) is False
  assert v.check_weight(1.0, float('nan'), 500.0) is False
  assert v.check_weight(1.0, 0.0, float('inf')) is False


def test_find_valid_positions_zero_dims_returns_empty():
  '''Mutated zero-dim carton yields [], not a crash.'''
  v = PlacementValidator(Space3D(4, 4, 4), {})
  carton = Carton(1, 1, 1, 1, 1.0, False, 1, 0)
  carton.length = 0
  assert not v.find_valid_positions(carton, 0.0, 500.0)


# S-03 generator caps.


def _gen():
  '''Make generator.'''
  return DataGenerator(AppConfig(), np.random.default_rng(42))


@pytest.mark.parametrize('kw', [
  {'num_trucks': -1}, {'num_stores': -1}, {'num_cartons': -1},
  {'num_warehouses': -1}, {'num_trucks': 13}, {'num_stores': 9},
  {'num_cartons': 121}, {'num_warehouses': 7},
  {'grid_rows': 0}, {'grid_cols': 0}, {'grid_rows': -5},
  {'grid_rows': 161}, {'grid_cols': 161}, {'grid_rows': 2},
])
def test_generate_rejects_absurd_inputs(kw):
  '''generate() rejects negative/over-cap inputs.'''
  args = dict(num_trucks=1, num_stores=1, num_cartons=1)
  args.update(kw)
  with pytest.raises(ValueError):
    _gen().generate(**args)


def test_generate_at_caps_still_works():
  '''Counts at (small-grid) caps do not raise.'''
  data = _gen().generate(
    num_trucks=1, num_stores=1, num_cartons=1,
    num_warehouses=1, grid_rows=5, grid_cols=5,
  )
  assert len(data.trucks) == 1


# S-05 reward robustness.


@pytest.mark.parametrize('component', [
  DisplacementReward, GroupingReward, PriorityReward,
  FragilityReward,
])
def test_orphan_cid_skipped_by_components(component):
  '''Orphan placed cid does not crash lookup components.'''
  result = component().compute(_orphan_state())
  assert math.isfinite(result)


def test_orphan_cid_calculator_finite():
  '''Calculator skips orphan cid, returns finite total.'''
  calc = RewardCalculator(RewardWeights())
  total, breakdown = calc.compute(_orphan_state())
  assert math.isfinite(total)
  assert all(math.isfinite(v) for v in breakdown.values())


def test_mismatched_weights_spaces_lengths():
  '''Short current_weights does not IndexError.'''
  state = _state(weights=[])
  assert math.isfinite(UtilizationReward().compute(state))
  assert math.isfinite(WeightReward().compute(state))
  calc = RewardCalculator(RewardWeights())
  total, _ = calc.compute(state)
  assert math.isfinite(total)


def test_nan_inf_weights_do_not_poison():
  '''NaN/inf current weights are skipped, not poison.'''
  for bad in (float('nan'), float('inf'), float('-inf')):
    state = _state(weights=[bad])
    util = UtilizationReward().compute(state)
    assert math.isfinite(util)
    assert 0.0 <= util <= 1.0
    assert math.isfinite(WeightReward().compute(state))
  calc = RewardCalculator(RewardWeights())
  state = _state(
    weights=[float('nan')], total_travel_distance=float('nan'),
  )
  total, breakdown = calc.compute(state)
  assert math.isfinite(total)
  assert all(math.isfinite(v) for v in breakdown.values())


def test_negative_weight_does_not_poison_utilization():
  '''Negative current weight yields >=0 utilization.'''
  state = _state(weights=[-999.0])
  util = UtilizationReward().compute(state)
  assert math.isfinite(util)
  assert util >= 0.0


def test_nan_travel_distance_finite():
  '''NaN travel distance yields finite reward.'''
  state = _state(total_travel_distance=float('nan'))
  result = TravelDistanceReward().compute(state)
  assert math.isfinite(result)
  assert 0.0 <= result <= 1.0
