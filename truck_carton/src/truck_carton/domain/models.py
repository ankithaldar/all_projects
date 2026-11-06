#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Domain models for trucks, cartons, and grid worlds.'''

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np

from truck_carton.config import EnvironmentConfig
from truck_carton.packing.rotation import Rotation


def _positive_dims(name: str, values: tuple) -> None:
  '''Require every dim to be a positive number.'''
  for value in values:
    if isinstance(value, bool) or not isinstance(
      value, (int, float, np.integer, np.floating),
    ):
      raise ValueError(f'{name} dims must be numbers')
    if not math.isfinite(float(value)) or value <= 0:
      raise ValueError(f'{name} dims must be > 0 (got {value})')


def _finite_nonneg_weight(name: str, value: float) -> None:
  '''Require a finite, non-negative weight.'''
  if isinstance(value, bool) or not isinstance(
    value, (int, float, np.integer, np.floating),
  ):
    raise ValueError(f'{name} must be a number')
  if not math.isfinite(float(value)) or value < 0:
    raise ValueError(
      f'{name} must be finite and >= 0 (got {value})'
    )


def _valid_id(name: str, value: int, minimum: int = 0) -> None:
  '''Require an integer id at or above a minimum.'''
  if isinstance(value, bool) or not isinstance(
    value, (int, np.integer),
  ):
    raise ValueError(f'{name} must be an integer id')
  if value < minimum:
    raise ValueError(f'{name} must be >= {minimum} (got {value})')


def _valid_position(name: str, position: tuple) -> None:
  '''Require a 2-tuple of non-negative integers.'''
  if len(position) != 2:
    raise ValueError(f'{name} must have length 2')
  for value in position:
    if isinstance(value, bool) or not isinstance(
      value, (int, np.integer),
    ):
      raise ValueError(f'{name} entries must be integers')
    if value < 0:
      raise ValueError(f'{name} entries must be >= 0')


class CellType(IntEnum):
  '''Cell Type.'''

  TERRAIN = 0
  ROAD = 1
  DEPOT = 2
  WAREHOUSE = 3
  STORE = 4


class TruckState(IntEnum):
  '''Truck State.'''

  ROUTING = 0
  LOADING = 1
  AT_DEPOT = 2


@dataclass
class Warehouse:
  '''Warehouse.'''

  warehouse_id: int
  position: tuple[int, int]

  def __post_init__(self) -> None:
    '''Validate warehouse id and position.'''
    _valid_id('warehouse_id', self.warehouse_id)
    _valid_position('position', self.position)


@dataclass
class Store:
  '''Store.'''

  store_id: int
  route_position: int
  position: tuple[int, int] = (0, 0)

  def __post_init__(self) -> None:
    '''Validate store ids and position.'''
    _valid_id('store_id', self.store_id)
    _valid_id('route_position', self.route_position)
    _valid_position('position', self.position)


@dataclass
class Truck:
  '''Truck.'''

  truck_id: int
  length: int
  width: int
  height: int
  max_weight: float
  route: list[int]
  position: tuple[int, int] = (0, 0)
  state: TruckState = TruckState.ROUTING
  visited_stores: list[int] = field(default_factory=list)

  def __post_init__(self) -> None:
    '''Validate truck dims, weight, and id.'''
    _valid_id('truck_id', self.truck_id)
    _positive_dims('truck', (self.length, self.width, self.height))
    _finite_nonneg_weight('max_weight', self.max_weight)
    _valid_position('position', self.position)


@dataclass
class Carton:
  '''Carton.'''

  carton_id: int
  length: int
  width: int
  height: int
  weight: float
  is_fragile: bool
  priority: int
  destination_store_id: int
  origin_warehouse_id: int = 0

  def __post_init__(self) -> None:
    '''Validate carton dims, weight, and ids.'''
    _valid_id('carton_id', self.carton_id, minimum=1)
    _positive_dims('carton', (self.length, self.width, self.height))
    _finite_nonneg_weight('weight', self.weight)
    _valid_id('destination_store_id', self.destination_store_id)
    _valid_id('origin_warehouse_id', self.origin_warehouse_id)


@dataclass
class PlacementInfo:
  '''Placement Info.'''

  truck_id: int
  position: tuple[int, int, int]
  oriented_dims: tuple[int, int, int]
  rotation: Rotation

  def __post_init__(self) -> None:
    '''Validate placement position and dims.'''
    _valid_id('truck_id', self.truck_id)
    if len(self.position) != 3 or any(
      isinstance(v, bool) or not isinstance(v, (int, np.integer)) or v < 0
      for v in self.position
    ):
      raise ValueError('position must be 3 non-negative integers')
    _positive_dims('oriented_dims', tuple(self.oriented_dims))


@dataclass
class GridWorld:
  '''Procedurally generated 2D grid layout.'''

  rows: int
  cols: int
  grid: np.ndarray
  depot_position: tuple[int, int]
  distance_matrix: np.ndarray
  facility_positions: list[tuple[int, int]]
  path_cache: dict = field(default_factory=dict)

  def __post_init__(self) -> None:
    '''Validate grid size against config caps.'''
    caps = EnvironmentConfig()
    for name, value, cap in (
      ('rows', self.rows, caps.max_grid_rows),
      ('cols', self.cols, caps.max_grid_cols),
    ):
      if isinstance(value, bool) or not isinstance(
        value, (int, np.integer),
      ):
        raise ValueError(f'{name} must be an integer')
      if value <= 0 or value > cap:
        raise ValueError(f'{name} must be in [1, {cap}] (got {value})')
    _valid_position('depot_position', self.depot_position)


@dataclass
class EpisodeData:
  '''Episode Data.'''

  trucks: list[Truck]
  stores: list[Store]
  cartons: list[Carton]
  warehouses: list[Warehouse] = field(default_factory=list)
  grid_world: GridWorld | None = None


@dataclass
class ObservationContext:
  '''All state needed for observation encoding.
  Decouples observer from environment internals.'''

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
  warehouses: list[Warehouse] = field(default_factory=list)
  warehouse_cartons: dict[int, list[int]] = field(default_factory=dict)
  routing_candidates: list = field(default_factory=list)
  active_truck_idx: int = 0
  total_travel: float = 0.0
  num_delivered: int = 0
