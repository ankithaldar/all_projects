from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from truck_carton.reward.calculator import (
        EnvironmentState,
    )


class FragilityReward:
    """Non-fragile-above-fragile violation rate."""

    def compute(
        self, state: EnvironmentState
    ) -> float:
        if not state.placed_cartons:
            return 0.0

        carton_lookup = {
            c.carton_id: c for c in state.all_cartons
        }
        violations = 0

        for truck_idx, space in enumerate(
            state.spaces
        ):
            fragile_ids: set[int] = set()
            for cid, info in (
                state.placed_cartons.items()
            ):
                if (
                    info.truck_id == truck_idx
                    and carton_lookup[cid].is_fragile
                ):
                    fragile_ids.add(cid)

            if not fragile_ids:
                continue

            for cid, info in (
                state.placed_cartons.items()
            ):
                if (
                    info.truck_id != truck_idx
                    or carton_lookup[cid].is_fragile
                ):
                    continue

                x, y, z = info.position
                dl, dw, dh = info.oriented_dims
                below = space.grid[
                    x:x + dl, y:y + dw, :z
                ]
                unique_below = (
                    set(np.unique(below).tolist())
                    - {0}
                )
                if unique_below & fragile_ids:
                    violations += 1

        total = max(len(state.placed_cartons), 1)
        return violations / total
