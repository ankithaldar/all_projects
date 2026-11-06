#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Equivalence of metrics vs reward components.'''

from truck_carton.domain.models import (
  Carton,
  EpisodeData,
  PlacementInfo,
  Store,
  Truck,
)
from truck_carton.evaluation.metrics import (
  EpisodeMetrics,
  MetricsCollector,
)
from truck_carton.packing.rotation import Rotation
from truck_carton.packing.space3d import Space3D
from truck_carton.reward.calculator import EnvironmentState
from truck_carton.reward.displacement import DisplacementReward
from truck_carton.reward.fragility import FragilityReward
from truck_carton.reward.grouping import GroupingReward
from truck_carton.reward.priority import PriorityReward
from truck_carton.reward.support import SupportReward
from truck_carton.reward.weight import WeightReward


def _crafted() -> tuple[EpisodeData, list[Space3D],
                       dict[int, PlacementInfo], list[float],
                       EnvironmentState]:
  '''Crafted episode exercising every divergence.'''
  trucks = [Truck(0, 8, 4, 4, 10.0, [0])]
  stores = [Store(0, 0), Store(1, 1)]
  cartons = [
    Carton(1, 1, 1, 1, 1.0, False, 3, 0),
    Carton(2, 1, 1, 1, 1.0, False, 1, 0),
    Carton(3, 1, 1, 1, 1.0, False, 1, 1),
    Carton(4, 1, 1, 1, 1.0, True, 1, 0),
    Carton(5, 1, 1, 1, 1.0, False, 1, 0),
    Carton(6, 1, 1, 1, 1.0, False, 1, 0),
  ]
  placed = {
    1: PlacementInfo(0, (3, 0, 0), (1, 1, 1), Rotation.LWH),
    2: PlacementInfo(0, (0, 0, 0), (1, 1, 1), Rotation.LWH),
    3: PlacementInfo(0, (1, 0, 0), (1, 1, 1), Rotation.LWH),
    4: PlacementInfo(0, (0, 1, 0), (1, 1, 1), Rotation.LWH),
    5: PlacementInfo(0, (0, 1, 1), (1, 1, 1), Rotation.LWH),
    6: PlacementInfo(0, (2, 1, 2), (1, 1, 1), Rotation.LWH),
  }
  space = Space3D(8, 4, 4)
  for cid, info in placed.items():
    x, y, z = info.position
    dl, dw, dh = info.oriented_dims
    space.place(cid, x, y, z, dl, dw, dh)
  spaces = [space]
  weights = [15.0]
  episode = EpisodeData(
    trucks=trucks, stores=stores, cartons=cartons,
  )
  state = EnvironmentState(
    trucks=trucks,
    spaces=spaces,
    placed_cartons=placed,
    unplaced_cartons=[],
    all_cartons=cartons,
    stores=stores,
    current_weights=weights,
    step_carton=None,
    step_placement=None,
    is_terminal=False,
    total_cartons=len(cartons),
  )
  return episode, spaces, placed, weights, state


def _metrics(state_tuple) -> EpisodeMetrics:
  '''Compute metrics for crafted tuple.'''
  episode, spaces, placed, weights, _ = state_tuple
  return MetricsCollector().compute(
    episode_data=episode,
    spaces=spaces,
    placed_cartons=placed,
    current_weights=weights,
    total_reward=0.0,
    curriculum_stage=0,
  )


def test_metrics_match_reward_on_crafted_episode():
  '''Metrics equal reward components on same episode.'''
  crafted = _crafted()
  state = crafted[4]
  m = _metrics(crafted)
  assert m.avg_displacement_per_stop == (
    DisplacementReward().compute(state)
  )
  assert m.grouping_compliance_rate == (
    GroupingReward().compute(state)
  )
  assert m.fragility_violation_rate == (
    FragilityReward().compute(state)
  )
  assert m.support_violation_rate == (
    SupportReward().compute(state)
  )
  assert m.weight_violation_rate == WeightReward().compute(state)
  assert m.priority_accessibility_score == (
    PriorityReward().compute(state)
  )


def test_displacement_excludes_off_route_carton():
  '''Off-route carton must not inflate displacement.'''
  crafted = _crafted()
  state = crafted[4]
  m = _metrics(crafted)
  assert DisplacementReward().compute(state) == 0.0
  assert m.avg_displacement_per_stop == 0.0


def test_priority_is_graded_not_binary():
  '''Behind high-priority carton scores graded.'''
  crafted = _crafted()
  state = crafted[4]
  graded = PriorityReward().compute(state)
  assert 0.0 < graded < 1.0
  assert _metrics(crafted).priority_accessibility_score == graded


def test_weight_is_graded_excess_ratio():
  '''Overweight scores excess ratio not binary.'''
  crafted = _crafted()
  state = crafted[4]
  assert WeightReward().compute(state) == 0.5
  assert _metrics(crafted).weight_violation_rate == 0.5


def test_grouping_fragility_support_match_reward():
  '''Grouping/fragility/support match crafted.'''
  crafted = _crafted()
  state = crafted[4]
  m = _metrics(crafted)
  assert m.grouping_compliance_rate == (
    GroupingReward().compute(state)
  )
  assert m.fragility_violation_rate == (
    FragilityReward().compute(state)
  )
  assert m.support_violation_rate == (
    SupportReward().compute(state)
  )
  assert m.fragility_violation_rate > 0.0
  assert m.support_violation_rate > 0.0


def test_empty_episode_matches_reward():
  '''Empty episode matches reward zeros.'''
  trucks = [Truck(0, 4, 4, 4, 500.0, [0])]
  stores = [Store(0, 0)]
  episode = EpisodeData(trucks=trucks, stores=stores, cartons=[])
  spaces = [Space3D(4, 4, 4)]
  state = EnvironmentState(
    trucks=trucks,
    spaces=spaces,
    placed_cartons={},
    unplaced_cartons=[],
    all_cartons=[],
    stores=stores,
    current_weights=[0.0],
    step_carton=None,
    step_placement=None,
    is_terminal=False,
    total_cartons=0,
  )
  m = MetricsCollector().compute(
    episode_data=episode,
    spaces=spaces,
    placed_cartons={},
    current_weights=[0.0],
    total_reward=0.0,
    curriculum_stage=0,
  )
  assert m.avg_displacement_per_stop == (
    DisplacementReward().compute(state)
  )
  assert m.grouping_compliance_rate == (
    GroupingReward().compute(state)
  )
  assert m.priority_accessibility_score == (
    PriorityReward().compute(state)
  )
  assert m.fragility_violation_rate == (
    FragilityReward().compute(state)
  )
  assert m.support_violation_rate == (
    SupportReward().compute(state)
  )
  assert m.weight_violation_rate == WeightReward().compute(state)
