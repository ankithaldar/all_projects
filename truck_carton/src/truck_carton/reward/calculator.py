#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Reward calculation combining individual components.'''

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol

from truck_carton.config import RewardWeights
from truck_carton.domain.models import (
  Carton,
  PlacementInfo,
  Store,
  Truck,
)
from truck_carton.packing.space3d import Space3D
from truck_carton.reward.completion import CompletionReward
from truck_carton.reward.displacement import DisplacementReward
from truck_carton.reward.fragility import FragilityReward
from truck_carton.reward.grouping import GroupingReward
from truck_carton.reward.priority import PriorityReward
from truck_carton.reward.support import SupportReward
from truck_carton.reward.travel_distance import (
  TravelDistanceReward,
)
from truck_carton.reward.utilization import UtilizationReward
from truck_carton.reward.weight import WeightReward


@dataclass
class EnvironmentState:
  '''Snapshot passed to each reward component.'''

  trucks: list[Truck]
  spaces: list[Space3D]
  placed_cartons: dict[int, PlacementInfo]
  unplaced_cartons: list[Carton]
  all_cartons: list[Carton]
  stores: list[Store]
  current_weights: list[float]
  step_carton: Carton | None
  step_placement: PlacementInfo | None
  is_terminal: bool
  total_cartons: int
  total_travel_distance: float = 0.0
  max_possible_distance: float = 1.0
  delivered_cartons: set[int] = field(default_factory=set)


class RewardComponent(Protocol):
  '''Reward Component.'''

  def compute(self, state: EnvironmentState) -> float:
    '''Compute.'''


class RewardCalculator:
  '''Orchestrates all reward components into a
  weighted sum.'''

  def __init__(self, weights: RewardWeights) -> None:
    '''Initialize the instance.'''
    self._components: list[tuple[str, float, RewardComponent]] = [
      (
        'utilization',
        weights.alpha_utilization,
        UtilizationReward(),
      ),
      (
        'displacement',
        weights.beta_displacement,
        DisplacementReward(),
      ),
      (
        'grouping',
        weights.gamma_grouping,
        GroupingReward(),
      ),
      (
        'fragility',
        weights.delta_fragility,
        FragilityReward(),
      ),
      (
        'support',
        weights.epsilon_support,
        SupportReward(),
      ),
      (
        'weight',
        weights.zeta_weight,
        WeightReward(),
      ),
      (
        'completion',
        weights.eta_completion,
        CompletionReward(),
      ),
      (
        'priority',
        weights.theta_priority,
        PriorityReward(),
      ),
      (
        'travel_distance',
        weights.iota_travel_distance,
        TravelDistanceReward(),
      ),
    ]

  def compute(self, state: EnvironmentState) -> tuple[float, dict[str, float]]:
    '''Compute.

    Defined S-05 behavior: orphan placed cids are skipped by
    lookup components; non-finite component outputs and
    non-finite weights are skipped (treated as 0.0), so NaN/inf
    never poisons the returned total or breakdown.'''
    breakdown: dict[str, float] = {}
    total = 0.0
    for name, weight, component in self._components:
      raw = component.compute(state)
      if not math.isfinite(raw):
        raw = 0.0
      breakdown[name] = raw
      if math.isfinite(weight):
        total += weight * raw
    return total, breakdown
