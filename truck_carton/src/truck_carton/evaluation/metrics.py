from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from truck_carton.domain.models import (
  Carton,
  EpisodeData,
  PlacementInfo,
  Store,
  Truck,
)
from truck_carton.packing.space3d import Space3D


@dataclass
class EpisodeMetrics:
  """All evaluation metrics for one episode."""

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
  """Computes all evaluation metrics from a
  completed episode."""

  def compute(
    self,
    episode_data: EpisodeData,
    spaces: list[Space3D],
    placed_cartons: dict[int, PlacementInfo],
    current_weights: list[float],
    total_reward: float,
    curriculum_stage: int,
  ) -> EpisodeMetrics:
    trucks = episode_data.trucks
    cartons = episode_data.cartons
    stores = episode_data.stores
    carton_lookup = {
      c.carton_id: c for c in cartons
    }

    vol_util = [
      s.get_occupancy_ratio() for s in spaces
    ]
    wt_util = [
      current_weights[i]
      / max(t.max_weight, 1e-9)
      for i, t in enumerate(trucks)
    ]

    avg_disp = self._compute_avg_displacement(
      trucks, spaces, placed_cartons,
      stores, carton_lookup,
    )
    grouping = self._compute_grouping(
      placed_cartons, carton_lookup
    )
    fragility = (
      self._compute_fragility_violations(
        spaces, placed_cartons, carton_lookup
      )
    )
    support = self._compute_support_violations(
      spaces, placed_cartons
    )
    weight_viol = (
      self._compute_weight_violations(
        trucks, current_weights
      )
    )
    priority = self._compute_priority_score(
      placed_cartons, carton_lookup
    )

    num_placed = len(placed_cartons)
    num_total = len(cartons)

    return EpisodeMetrics(
      volumetric_utilization_per_truck=vol_util,
      fleet_volumetric_utilization=(
        float(np.mean(vol_util))
        if vol_util
        else 0.0
      ),
      weight_utilization_per_truck=wt_util,
      avg_displacement_per_stop=avg_disp,
      grouping_compliance_rate=grouping,
      fragility_violation_rate=fragility,
      support_violation_rate=support,
      weight_violation_rate=weight_viol,
      priority_accessibility_score=priority,
      total_reward=total_reward,
      num_placed=num_placed,
      num_total=num_total,
      completion_rate=(
        num_placed / max(num_total, 1)
      ),
      curriculum_stage=curriculum_stage,
    )

  def _compute_avg_displacement(
    self,
    trucks: list[Truck],
    spaces: list[Space3D],
    placed: dict[int, PlacementInfo],
    stores: list[Store],
    carton_lookup: dict[int, Carton],
  ) -> float:
    store_positions = {
      s.store_id: s.route_position
      for s in stores
    }
    total_disp = 0
    total_stops = 0

    for truck_idx, truck in enumerate(trucks):
      truck_cids = [
        cid for cid, info in placed.items()
        if info.truck_id == truck_idx
      ]
      if not truck_cids:
        continue

      route = sorted(
        truck.route,
        key=lambda sid: store_positions.get(
          sid, 0
        ),
      )
      unloaded: set[int] = set()

      for store_id in route:
        store_cids = [
          cid for cid in truck_cids
          if cid not in unloaded
          and carton_lookup[cid]
          .destination_store_id == store_id
        ]
        if not store_cids:
          continue

        max_x = max(
          placed[cid].position[0]
          + placed[cid].oriented_dims[0]
          for cid in store_cids
        )
        blockers = sum(
          1 for cid in truck_cids
          if cid not in unloaded
          and cid not in store_cids
          and placed[cid].position[0] < max_x
        )
        total_disp += blockers
        total_stops += 1
        unloaded.update(store_cids)

    return total_disp / max(total_stops, 1)

  def _compute_grouping(
    self,
    placed: dict[int, PlacementInfo],
    carton_lookup: dict[int, Carton],
  ) -> float:
    groups: dict[
      tuple[int, int], list[int]
    ] = {}
    for cid, info in placed.items():
      key = (
        info.truck_id,
        carton_lookup[cid]
        .destination_store_id,
      )
      groups.setdefault(key, []).append(cid)

    scores: list[float] = []
    for cids in groups.values():
      if len(cids) < 2:
        scores.append(1.0)
        continue

      min_pos = np.array(
        [float('inf')] * 3
      )
      max_pos = np.array(
        [float('-inf')] * 3
      )
      vol = 0
      for cid in cids:
        info = placed[cid]
        p = np.array(
          info.position, dtype=np.float64
        )
        d = np.array(
          info.oriented_dims,
          dtype=np.float64,
        )
        min_pos = np.minimum(min_pos, p)
        max_pos = np.maximum(
          max_pos, p + d
        )
        vol += int(np.prod(d))

      bbox = float(
        np.prod(max_pos - min_pos)
      )
      scores.append(
        min(vol / max(bbox, 1.0), 1.0)
      )

    return (
      float(np.mean(scores))
      if scores
      else 1.0
    )

  def _compute_fragility_violations(
    self,
    spaces: list[Space3D],
    placed: dict[int, PlacementInfo],
    carton_lookup: dict[int, Carton],
  ) -> float:
    violations = 0
    for cid, info in placed.items():
      c = carton_lookup[cid]
      if c.is_fragile:
        continue
      space = spaces[info.truck_id]
      x, y, z = info.position
      dl, dw, _ = info.oriented_dims
      if z == 0:
        continue
      below = space.grid[
        x:x + dl, y:y + dw, :z
      ]
      unique = (
        set(np.unique(below).tolist())
        - {0}
      )
      for uid in unique:
        if (
          uid in carton_lookup
          and carton_lookup[uid].is_fragile
        ):
          violations += 1
          break

    return violations / max(len(placed), 1)

  def _compute_support_violations(
    self,
    spaces: list[Space3D],
    placed: dict[int, PlacementInfo],
  ) -> float:
    violations = 0
    for cid, info in placed.items():
      x, y, z = info.position
      dl, dw, _ = info.oriented_dims
      if z == 0:
        continue
      space = spaces[info.truck_id]
      below = space.grid[
        x:x + dl, y:y + dw, z - 1
      ]
      if not np.all(below != 0):
        violations += 1

    return violations / max(len(placed), 1)

  def _compute_weight_violations(
    self,
    trucks: list[Truck],
    weights: list[float],
  ) -> float:
    violations = sum(
      1 for i, t in enumerate(trucks)
      if i < len(weights)
      and weights[i] > t.max_weight
    )
    return violations / max(len(trucks), 1)

  def _compute_priority_score(
    self,
    placed: dict[int, PlacementInfo],
    carton_lookup: dict[int, Carton],
  ) -> float:
    groups: dict[
      tuple[int, int], list[int]
    ] = {}
    for cid, info in placed.items():
      key = (
        info.truck_id,
        carton_lookup[cid]
        .destination_store_id,
      )
      groups.setdefault(key, []).append(cid)

    scores: list[float] = []
    for cids in groups.values():
      high = [
        c for c in cids
        if carton_lookup[c].priority == 3
      ]
      other = [
        c for c in cids
        if carton_lookup[c].priority < 3
      ]
      if not high or not other:
        scores.append(1.0)
        continue

      hp_min_x = min(
        placed[c].position[0] for c in high
      )
      other_min_x = min(
        placed[c].position[0]
        for c in other
      )
      scores.append(
        1.0
        if hp_min_x <= other_min_x
        else 0.0
      )

    return (
      float(np.mean(scores))
      if scores
      else 1.0
    )
