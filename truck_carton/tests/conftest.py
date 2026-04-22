import numpy as np
import pytest

from truck_carton.config import AppConfig
from truck_carton.domain.models import (
  Carton,
  EpisodeData,
  Store,
  Truck,
)
from truck_carton.packing.space3d import Space3D


@pytest.fixture
def config() -> AppConfig:
  return AppConfig()


@pytest.fixture
def rng() -> np.random.Generator:
  return np.random.default_rng(42)


@pytest.fixture
def small_truck() -> Truck:
  return Truck(
    truck_id=0, length=8, width=4,
    height=4, max_weight=500.0, route=[0, 1],
  )


@pytest.fixture
def small_space(small_truck: Truck) -> Space3D:
  return Space3D(
    small_truck.length,
    small_truck.width,
    small_truck.height,
  )


@pytest.fixture
def sample_carton() -> Carton:
  return Carton(
    carton_id=1, length=2, width=2, height=2,
    weight=10.0, is_fragile=False, priority=2,
    destination_store_id=0,
  )


@pytest.fixture
def fragile_carton() -> Carton:
  return Carton(
    carton_id=2, length=2, width=2, height=1,
    weight=5.0, is_fragile=True, priority=3,
    destination_store_id=1,
  )


@pytest.fixture
def stores() -> list[Store]:
  return [
    Store(store_id=0, route_position=0),
    Store(store_id=1, route_position=1),
  ]


@pytest.fixture
def sample_episode(
  small_truck: Truck,
  stores: list[Store],
  sample_carton: Carton,
  fragile_carton: Carton,
) -> EpisodeData:
  truck2 = Truck(
    truck_id=1, length=6, width=4,
    height=4, max_weight=400.0, route=[0],
  )
  cartons = [
    sample_carton,
    fragile_carton,
    Carton(
      carton_id=3, length=1, width=1,
      height=1, weight=3.0, is_fragile=False,
      priority=1, destination_store_id=0,
    ),
    Carton(
      carton_id=4, length=2, width=1,
      height=1, weight=8.0, is_fragile=False,
      priority=3, destination_store_id=1,
    ),
  ]
  return EpisodeData(
    trucks=[small_truck, truck2],
    stores=stores,
    cartons=cartons,
  )
