from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np

from truck_carton.packing.rotation import Rotation


class CellType(IntEnum):
  TERRAIN = 0
  ROAD = 1
  DEPOT = 2
  WAREHOUSE = 3
  STORE = 4


class TruckState(IntEnum):
  ROUTING = 0
  LOADING = 1
  AT_DEPOT = 2


@dataclass
class Warehouse:
  warehouse_id: int
  position: tuple[int, int]


@dataclass
class Store:
  store_id: int
  route_position: int
  position: tuple[int, int] = (0, 0)


@dataclass
class Truck:
  truck_id: int
  length: int
  width: int
  height: int
  max_weight: float
  route: list[int]
  position: tuple[int, int] = (0, 0)
  state: TruckState = TruckState.ROUTING
  visited_stores: list[int] = field(
    default_factory=list
  )


@dataclass
class Carton:
  carton_id: int
  length: int
  width: int
  height: int
  weight: float
  is_fragile: bool
  priority: int
  destination_store_id: int
  origin_warehouse_id: int = 0


@dataclass
class PlacementInfo:
  truck_id: int
  position: tuple[int, int, int]
  oriented_dims: tuple[int, int, int]
  rotation: Rotation


@dataclass
class GridWorld:
  """Procedurally generated 2D grid layout."""

  rows: int
  cols: int
  grid: np.ndarray
  depot_position: tuple[int, int]
  distance_matrix: np.ndarray
  facility_positions: list[tuple[int, int]]
  path_cache: dict = field(default_factory=dict)


@dataclass
class EpisodeData:
  trucks: list[Truck]
  stores: list[Store]
  cartons: list[Carton]
  warehouses: list[Warehouse] = field(
    default_factory=list
  )
  grid_world: GridWorld | None = None


@dataclass
class ObservationContext:
  """All state needed for observation encoding.
  Decouples observer from environment internals."""

  trucks: list[Truck]
  spaces: list
  current_carton: Carton | None
  remaining_cartons: list[Carton]
  packing_candidates: list
  current_weights: list[float]
  num_placed: int
  total_cartons: int
  stage_index: int
  step_count: int
  max_steps: int
  grid_world: GridWorld | None = None
  warehouses: list[Warehouse] = field(
    default_factory=list
  )
  warehouse_cartons: dict[int, list[int]] = field(
    default_factory=dict
  )
  routing_candidates: list = field(
    default_factory=list
  )
  active_truck_idx: int = 0
  total_travel: float = 0.0
  num_delivered: int = 0
