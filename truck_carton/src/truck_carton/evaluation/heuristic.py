"""Heuristic baseline agent for truck-carton packing.

Provides a deterministic rule-based policy for
comparison with RL-trained models. Uses domain
knowledge to make packing and routing decisions
without any learning.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from truck_carton.domain.models import CellType

if TYPE_CHECKING:
  from truck_carton.domain.models import Carton
  from truck_carton.env.action import (
    ActionManager,
    RoutingCandidate,
  )
  from truck_carton.env.packing_env import (
    TruckCartonPackingEnv,
  )
  from truck_carton.packing.placement import (
    PlacementCandidate,
  )


class HeuristicAgent:
  """Rule-based agent using domain heuristics.

  Packing strategy (higher score = better):
    1. Prefer lowest z (bottom-up stacking)
    2. Prefer corner/wall positions for stability
    3. Prefer positions that group same-store cartons
    4. Place fragile cartons high (near the top)
    5. Place high-priority near door (low x)

  Routing strategy:
    1. Visit nearest warehouse with cartons
    2. Deliver to stores with most matching cargo
    3. Go to depot when no useful destinations
  """

  def __init__(self, config: object) -> None:
    self._max_candidates = getattr(
      getattr(config, 'env', config),
      'max_candidates', 500,
    )

  def predict(
    self,
    env: TruckCartonPackingEnv,
    masks: np.ndarray,
  ) -> int:
    """Select an action using heuristic rules."""
    valid = np.where(masks)[0]
    if len(valid) == 0:
      return 0

    am = env.action_manager
    packing_valid = [
      a for a in valid if a < self._max_candidates
    ]
    routing_valid = [
      a for a in valid if a >= self._max_candidates
    ]

    if packing_valid and am.candidates:
      return self._select_packing(
        packing_valid, am, env
      )
    if routing_valid and am.routing_candidates:
      return self._select_routing(
        routing_valid, am, env
      )
    return int(valid[0])

  def _select_packing(
    self,
    valid_actions: list[int],
    am: ActionManager,
    env: TruckCartonPackingEnv,
  ) -> int:
    """Score each packing candidate and pick best."""
    carton = env.current_carton
    if carton is None:
      return valid_actions[0]

    placed = env.placed_cartons
    carton_lookup = env.carton_lookup
    truck_idx = env.active_truck_idx

    best_score = -float('inf')
    best_action = valid_actions[0]

    for action in valid_actions:
      if action >= len(am.candidates):
        continue
      cand = am.candidates[action]
      score = self._score_candidate(
        cand, carton, placed, carton_lookup,
        truck_idx, env,
      )
      if score > best_score:
        best_score = score
        best_action = action

    return best_action

  def _score_candidate(
    self,
    cand: PlacementCandidate,
    carton: Carton,
    placed: dict[int, object],
    carton_lookup: dict[int, Carton],
    truck_idx: int,
    env: TruckCartonPackingEnv,
  ) -> float:
    """Multi-criteria scoring for a placement."""
    x, y, z = cand.x, cand.y, cand.z
    dl, dw, dh = cand.oriented_dims
    ep = env.episode_data
    if ep is None or cand.truck_id >= len(ep.trucks):
      return 0.0
    truck = ep.trucks[cand.truck_id]
    score = 0.0

    # 1. Bottom-up: prefer lowest z
    score -= z * 10.0

    # 2. Corner/wall bonus
    at_x_wall = (x == 0 or x + dl == truck.length)
    at_y_wall = (y == 0 or y + dw == truck.width)
    if at_x_wall and at_y_wall:
      score += 8.0
    elif at_x_wall or at_y_wall:
      score += 4.0

    # 3. Grouping: prefer adjacent to same-store
    dest = carton.destination_store_id
    neighbor_bonus = 0.0
    for cid, info in placed.items():
      if info.truck_id != cand.truck_id:
        continue
      other = carton_lookup.get(cid)
      if other is None:
        continue
      if other.destination_store_id == dest:
        ox, oy, oz = info.position
        odl, odw, odh = info.oriented_dims
        dx = max(0, max(x, ox) - min(
          x + dl, ox + odl
        ))
        dy = max(0, max(y, oy) - min(
          y + dw, oy + odw
        ))
        dz = max(0, max(z, oz) - min(
          z + dh, oz + odh
        ))
        dist = dx + dy + dz
        if dist <= 1:
          neighbor_bonus += 5.0
        elif dist <= 3:
          neighbor_bonus += 2.0
    score += min(neighbor_bonus, 15.0)

    # 4. Fragile cartons: prefer high z
    if carton.is_fragile:
      score += z * 5.0

    # 5. Priority accessibility: high-priority
    #    cartons near door (low x)
    if carton.priority == 3:
      score -= x * 3.0
    elif carton.priority == 2:
      score -= x * 1.5

    # 6. Compact packing: minimize wasted x-extent
    score -= (x + dl) * 0.5

    # 7. Weight balance: slight preference for
    #    distributing across trucks evenly
    if cand.truck_id == truck_idx:
      score += 1.0

    return score

  def _select_routing(
    self,
    valid_actions: list[int],
    am: ActionManager,
    env: TruckCartonPackingEnv,
  ) -> int:
    """Pick best routing destination."""
    truck_idx = env.active_truck_idx
    cargo = env.truck_cargo[truck_idx]
    carton_lookup = env.carton_lookup
    wh_cartons = env.warehouse_cartons

    best_score = -float('inf')
    best_action = valid_actions[0]
    offset = self._max_candidates

    for action in valid_actions:
      ri = action - offset
      if ri >= len(am.routing_candidates):
        continue
      rc = am.routing_candidates[ri]
      score = self._score_routing(
        rc, cargo, carton_lookup, wh_cartons,
      )
      if score > best_score:
        best_score = score
        best_action = action

    return best_action

  def _score_routing(
    self,
    rc: RoutingCandidate,
    cargo: set[int],
    carton_lookup: dict[int, Carton],
    wh_cartons: dict[int, list[int]],
  ) -> float:
    """Score a routing candidate."""
    score = 0.0
    dist = max(rc.distance, 1.0)

    if rc.location_type == CellType.WAREHOUSE:
      wh_id = rc.location_id
      remaining = len(
        wh_cartons.get(wh_id, [])
      )
      score = remaining * 10.0 / dist

    elif rc.location_type == CellType.STORE:
      matching = sum(
        1 for cid in cargo
        if carton_lookup.get(cid) is not None
        and carton_lookup[cid]
        .destination_store_id == rc.location_id
      )
      score = matching * 15.0 / dist

    elif rc.location_type == CellType.DEPOT:
      if not cargo:
        score = 5.0 / dist
      else:
        score = -100.0

    return score


class RandomAgent:
  """Uniform random baseline for comparison."""

  def __init__(self, seed: int = 42) -> None:
    self._rng = np.random.default_rng(seed)

  def predict(
    self,
    env: TruckCartonPackingEnv,
    masks: np.ndarray,
  ) -> int:
    valid = np.where(masks)[0]
    if len(valid) == 0:
      return 0
    return int(self._rng.choice(valid))
