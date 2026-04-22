from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from truck_carton.reward.calculator import (
        EnvironmentState,
    )


class GroupingReward:
    """Bounding-box tightness of same-store groups."""

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
                if len(cids) < 2:
                    scores.append(1.0)
                    continue

                min_pos = np.array(
                    [float('inf')] * 3
                )
                max_pos = np.array(
                    [float('-inf')] * 3
                )
                total_vol = 0

                for cid in cids:
                    info = state.placed_cartons[cid]
                    pos = np.array(
                        info.position,
                        dtype=np.float64,
                    )
                    dims = np.array(
                        info.oriented_dims,
                        dtype=np.float64,
                    )
                    min_pos = np.minimum(
                        min_pos, pos
                    )
                    max_pos = np.maximum(
                        max_pos, pos + dims
                    )
                    total_vol += int(np.prod(dims))

                bbox_dims = max_pos - min_pos
                bbox_vol = float(np.prod(bbox_dims))
                if bbox_vol > 0:
                    scores.append(
                        min(total_vol / bbox_vol, 1.0)
                    )
                else:
                    scores.append(1.0)

        if not scores:
            return 0.0
        return float(np.mean(scores))
