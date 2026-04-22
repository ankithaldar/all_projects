from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from truck_carton.reward.calculator import (
        EnvironmentState,
    )


class DisplacementReward:
    def compute(
        self, state: EnvironmentState
    ) -> float:
        if not state.placed_cartons:
            return 0.0

        carton_lookup = {
            c.carton_id: c for c in state.all_cartons
        }
        store_positions = {
            s.store_id: s.route_position
            for s in state.stores
        }
        total_displacement = 0
        total_stops = 0

        for truck_idx, truck in enumerate(state.trucks):
            if truck_idx >= len(state.spaces):
                break

            truck_cids = [
                cid
                for cid, info in state.placed_cartons.items()
                if info.truck_id == truck_idx
            ]
            if not truck_cids:
                continue

            route_stores = sorted(
                truck.route,
                key=lambda sid: store_positions.get(
                    sid, 0
                ),
            )
            unloaded: set[int] = set()

            for store_id in route_stores:
                store_cids = [
                    cid
                    for cid in truck_cids
                    if cid not in unloaded
                    and carton_lookup[cid]
                    .destination_store_id == store_id
                ]
                if not store_cids:
                    continue

                max_x_store = 0
                for cid in store_cids:
                    info = state.placed_cartons[cid]
                    x_end = (
                        info.position[0]
                        + info.oriented_dims[0]
                    )
                    max_x_store = max(
                        max_x_store, x_end
                    )

                blockers = 0
                for cid in truck_cids:
                    if (
                        cid in unloaded
                        or cid in store_cids
                    ):
                        continue
                    info = state.placed_cartons[cid]
                    if info.position[0] < max_x_store:
                        blockers += 1

                total_displacement += blockers
                total_stops += 1
                unloaded.update(store_cids)

        if total_stops == 0:
            return 0.0

        avg_disp = total_displacement / total_stops
        max_possible = max(
            len(state.placed_cartons), 1
        )
        return avg_disp / max_possible
