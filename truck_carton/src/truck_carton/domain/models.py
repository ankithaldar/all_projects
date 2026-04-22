from __future__ import annotations

from dataclasses import dataclass

from truck_carton.packing.rotation import Rotation


@dataclass
class Store:
  store_id: int
  route_position: int


@dataclass
class Truck:
  truck_id: int
  length: int
  width: int
  height: int
  max_weight: float
  route: list[int]


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


@dataclass
class PlacementInfo:
  truck_id: int
  position: tuple[int, int, int]
  oriented_dims: tuple[int, int, int]
  rotation: Rotation


@dataclass
class EpisodeData:
  trucks: list[Truck]
  stores: list[Store]
  cartons: list[Carton]
