from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from truck_carton.reward.calculator import (
    EnvironmentState,
  )


class CompletionReward:
  """Placement progress ratio, 1.0 when all placed."""

  def compute(
    self, state: EnvironmentState
  ) -> float:
    if state.total_cartons == 0:
      return 0.0

    fraction = (
      len(state.placed_cartons)
      / state.total_cartons
    )
    all_placed = (
      len(state.placed_cartons)
      == state.total_cartons
    )
    # Scale so 100% placement yields 1.0, with
    # a steeper curve rewarding near-completion.
    if all_placed:
      return 1.0
    return fraction * 0.5
