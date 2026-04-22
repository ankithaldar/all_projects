from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from truck_carton.reward.calculator import (
    EnvironmentState,
  )


class TravelDistanceReward:
  """Normalized fleet travel distance [0, 1]."""

  def compute(
    self, state: EnvironmentState
  ) -> float:
    if state.max_possible_distance <= 0:
      return 0.0
    return min(
      state.total_travel_distance
      / state.max_possible_distance,
      1.0,
    )
