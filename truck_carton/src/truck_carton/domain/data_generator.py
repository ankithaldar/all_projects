from __future__ import annotations

import numpy as np

from truck_carton.config import AppConfig
from truck_carton.domain.models import (
    Carton,
    EpisodeData,
    Store,
    Truck,
)


class DataGenerator:
    """Generates random episode data for each curriculum stage."""

    def __init__(
        self, config: AppConfig, rng: np.random.Generator
    ) -> None:
        self._config = config
        self._rng = rng

    def generate(
        self,
        num_trucks: int,
        num_stores: int,
        num_cartons: int,
    ) -> EpisodeData:
        stores = self._generate_stores(num_stores)
        trucks = self._generate_trucks(num_trucks, stores)
        cartons = self._generate_cartons(num_cartons, stores)
        return EpisodeData(
            trucks=trucks, stores=stores, cartons=cartons
        )

    def _generate_stores(
        self, num_stores: int
    ) -> list[Store]:
        return [
            Store(store_id=i, route_position=i)
            for i in range(num_stores)
        ]

    def _generate_trucks(
        self, num_trucks: int, stores: list[Store]
    ) -> list[Truck]:
        tc = self._config.truck
        trucks: list[Truck] = []
        store_ids = [s.store_id for s in stores]

        for i in range(num_trucks):
            num_stops = self._rng.integers(
                1, len(store_ids) + 1
            )
            route = sorted(
                self._rng.choice(
                    store_ids,
                    size=int(num_stops),
                    replace=False,
                ).tolist()
            )

            trucks.append(Truck(
                truck_id=i,
                length=int(self._rng.integers(
                    tc.length_range[0],
                    tc.length_range[1] + 1,
                )),
                width=int(self._rng.integers(
                    tc.width_range[0],
                    tc.width_range[1] + 1,
                )),
                height=int(self._rng.integers(
                    tc.height_range[0],
                    tc.height_range[1] + 1,
                )),
                max_weight=float(self._rng.uniform(
                    tc.weight_capacity_range[0],
                    tc.weight_capacity_range[1],
                )),
                route=route,
            ))
        return trucks

    def _generate_cartons(
        self, num_cartons: int, stores: list[Store]
    ) -> list[Carton]:
        cc = self._config.carton
        store_ids = [s.store_id for s in stores]
        cartons: list[Carton] = []

        for i in range(num_cartons):
            cartons.append(Carton(
                carton_id=i,
                length=int(self._rng.integers(
                    cc.length_range[0],
                    cc.length_range[1] + 1,
                )),
                width=int(self._rng.integers(
                    cc.width_range[0],
                    cc.width_range[1] + 1,
                )),
                height=int(self._rng.integers(
                    cc.height_range[0],
                    cc.height_range[1] + 1,
                )),
                weight=float(self._rng.uniform(
                    cc.weight_range[0], cc.weight_range[1]
                )),
                is_fragile=bool(
                    self._rng.random() < cc.fragile_probability
                ),
                priority=int(self._rng.integers(1, 4)),
                destination_store_id=int(
                    self._rng.choice(store_ids)
                ),
            ))
        return cartons
