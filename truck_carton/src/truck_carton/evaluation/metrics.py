#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Metrics collection for truck-carton episodes.

Single source of truth: displacement, grouping, fragility,
support, weight and priority metrics delegate to the
reward components in truck_carton.reward, so evaluation
cannot diverge from training reward. Intentional changes
from the former duplicated implementations, proven by
tests/test_metrics_reward_equivalence.py:
- displacement: route-excluded and normalized by
  placed count (was raw per-stop average, counted
  off-route cartons).
- priority: graded accessibility (was binary 0/1).
- weight: graded excess ratio (was binary per truck).
- grouping/priority on empty episodes: 0.0 (was 1.0);
  reward defines no placements as 0.0.
Grouping/fragility/support algorithms were already
equivalent on non-empty episodes; delegation removes
the duplicate copies. Other fields (utilization,
completion, travel, delivery) are evaluation-only
and remain computed here.
'''

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from truck_carton.domain.models import (
  EpisodeData,
  PlacementInfo,
)
from truck_carton.packing.space3d import Space3D
from truck_carton.reward.calculator import EnvironmentState
from truck_carton.reward.displacement import DisplacementReward
from truck_carton.reward.fragility import FragilityReward
from truck_carton.reward.grouping import GroupingReward
from truck_carton.reward.priority import PriorityReward
from truck_carton.reward.support import SupportReward
from truck_carton.reward.weight import WeightReward


@dataclass
class EpisodeMetrics:
  '''All evaluation metrics for one episode.'''

  volumetric_utilization_per_truck: list[float]
  fleet_volumetric_utilization: float
  weight_utilization_per_truck: list[float]
  avg_displacement_per_stop: float
  grouping_compliance_rate: float
  fragility_violation_rate: float
  support_violation_rate: float
  weight_violation_rate: float
  priority_accessibility_score: float
  total_reward: float
  num_placed: int
  num_total: int
  completion_rate: float
  curriculum_stage: int
  total_travel_distance: float = 0.0
  avg_travel_per_truck: float = 0.0
  delivery_completion_rate: float = 0.0


class MetricsCollector:
  '''Computes all evaluation metrics from a
  completed episode.'''

  def compute(
    self,
    episode_data: EpisodeData,
    spaces: list[Space3D],
    placed_cartons: dict[int, PlacementInfo],
    current_weights: list[float],
    total_reward: float,
    curriculum_stage: int,
    truck_travel: list[float] | None = None,
    num_delivered: int = 0,
  ) -> EpisodeMetrics:
    '''Compute.'''
    trucks = episode_data.trucks
    cartons = episode_data.cartons

    vol_util = [s.get_occupancy_ratio() for s in spaces]
    wt_util = [
      current_weights[i] / max(t.max_weight, 1e-9) for i, t in enumerate(trucks)
    ]

    state = EnvironmentState(
      trucks=trucks,
      spaces=spaces,
      placed_cartons=placed_cartons,
      unplaced_cartons=[
        c for c in cartons if c.carton_id not in placed_cartons
      ],
      all_cartons=cartons,
      stores=episode_data.stores,
      current_weights=current_weights,
      step_carton=None,
      step_placement=None,
      is_terminal=False,
      total_cartons=len(cartons),
    )

    num_placed = len(placed_cartons)
    num_total = len(cartons)

    travel = truck_travel or []
    total_travel = sum(travel)
    num_trucks = max(len(trucks), 1)

    return EpisodeMetrics(
      volumetric_utilization_per_truck=vol_util,
      fleet_volumetric_utilization=(
        float(np.mean(vol_util)) if vol_util else 0.0
      ),
      weight_utilization_per_truck=wt_util,
      avg_displacement_per_stop=(
        DisplacementReward().compute(state)
      ),
      grouping_compliance_rate=GroupingReward().compute(state),
      fragility_violation_rate=FragilityReward().compute(state),
      support_violation_rate=SupportReward().compute(state),
      weight_violation_rate=WeightReward().compute(state),
      priority_accessibility_score=(
        PriorityReward().compute(state)
      ),
      total_reward=total_reward,
      num_placed=num_placed,
      num_total=num_total,
      completion_rate=(num_placed / max(num_total, 1)),
      curriculum_stage=curriculum_stage,
      total_travel_distance=total_travel,
      avg_travel_per_truck=(total_travel / num_trucks),
      delivery_completion_rate=(num_delivered / max(num_total, 1)),
    )
