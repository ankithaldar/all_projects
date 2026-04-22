from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from truck_carton.reward.calculator import (
        EnvironmentState,
    )


class WeightReward:
    def compute(
        self, state: EnvironmentState
    ) -> float:
        if not state.trucks:
            return 0.0

        penalty = 0.0
        for i, truck in enumerate(state.trucks):
            if i >= len(state.current_weights):
                break
            excess = (
                state.current_weights[i]
                - truck.max_weight
            )
            if excess > 0 and truck.max_weight > 0:
                penalty += excess / truck.max_weight

        return penalty / max(len(state.trucks), 1)
