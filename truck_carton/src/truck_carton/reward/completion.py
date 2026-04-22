from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from truck_carton.reward.calculator import (
        EnvironmentState,
    )


class CompletionReward:
    def compute(
        self, state: EnvironmentState
    ) -> float:
        if state.total_cartons == 0:
            return 0.0

        fraction = (
            len(state.placed_cartons)
            / state.total_cartons
        )
        bonus = 1.0 if (
            len(state.placed_cartons)
            == state.total_cartons
        ) else 0.0
        return fraction + bonus
