from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
  from truck_carton.reward.calculator import (
    EnvironmentState,
  )


class SupportReward:
  """Unsupported carton base cell violation rate."""

  def compute(
    self, state: EnvironmentState
  ) -> float:
    if not state.placed_cartons:
      return 0.0

    violations = 0

    for truck_idx, space in enumerate(
      state.spaces
    ):
      for cid, info in (
        state.placed_cartons.items()
      ):
        if info.truck_id != truck_idx:
          continue

        x, y, z = info.position
        dl, dw, dh = info.oriented_dims

        if z == 0:
          continue

        below = space.grid[
          x:x + dl, y:y + dw, z - 1
        ]
        if not np.all(below != 0):
          violations += 1

    total = max(len(state.placed_cartons), 1)
    return min(violations / total, 1.0)
