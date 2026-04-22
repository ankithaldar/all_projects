from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
  from truck_carton.reward.calculator import (
    EnvironmentState,
  )


class PriorityReward:
  """High-priority carton accessibility score."""

  def compute(
    self, state: EnvironmentState
  ) -> float:
    if not state.placed_cartons:
      return 0.0

    carton_lookup = {
      c.carton_id: c for c in state.all_cartons
    }
    scores: list[float] = []

    for truck_idx in range(len(state.trucks)):
      store_groups: dict[int, list[int]] = {}
      for cid, info in (
        state.placed_cartons.items()
      ):
        if info.truck_id != truck_idx:
          continue
        store_id = (
          carton_lookup[cid]
          .destination_store_id
        )
        store_groups.setdefault(
          store_id, []
        ).append(cid)

      for store_id, cids in (
        store_groups.items()
      ):
        high_priority = [
          cid for cid in cids
          if carton_lookup[cid].priority == 3
        ]
        if not high_priority:
          scores.append(1.0)
          continue

        other = [
          cid for cid in cids
          if carton_lookup[cid].priority < 3
        ]
        if not other:
          scores.append(1.0)
          continue

        hp_min_x = min(
          state.placed_cartons[cid]
          .position[0]
          for cid in high_priority
        )
        other_min_x = min(
          state.placed_cartons[cid]
          .position[0]
          for cid in other
        )

        if hp_min_x <= other_min_x:
          scores.append(1.0)
        else:
          max_x = max(
            state.placed_cartons[cid]
            .position[0]
            + state.placed_cartons[cid]
            .oriented_dims[0]
            for cid in cids
          )
          if max_x > 0:
            scores.append(max(
              0.0,
              1.0 - (
                (hp_min_x - other_min_x)
                / max_x
              ),
            ))
          else:
            scores.append(0.0)

    if not scores:
      return 0.0
    return float(
      np.clip(np.mean(scores), 0.0, 1.0)
    )
