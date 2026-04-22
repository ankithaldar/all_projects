from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from truck_carton.reward.calculator import (
        EnvironmentState,
    )


class UtilizationReward:
    """Average volumetric and weight fill ratio."""

    def compute(
        self, state: EnvironmentState
    ) -> float:
        if not state.spaces:
            return 0.0

        vol_sum = 0.0
        wt_sum = 0.0
        active = 0

        for i, space in enumerate(state.spaces):
            if i >= len(state.trucks):
                break
            truck = state.trucks[i]
            vol_sum += space.get_occupancy_ratio()
            if truck.max_weight > 0:
                wt_sum += min(
                    state.current_weights[i]
                    / truck.max_weight,
                    1.0,
                )
            active += 1

        if active == 0:
            return 0.0

        vol_util = vol_sum / active
        wt_util = wt_sum / active
        return min((vol_util + wt_util) / 2.0, 1.0)
