#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Candidate index action handling for the environment.'''

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from truck_carton.config import EnvironmentConfig
from truck_carton.domain.models import CellType
from truck_carton.packing.placement import (
  PlacementCandidate,
)
from truck_carton.packing.rotation import (
  apply_rotation,
)

if TYPE_CHECKING:
  from truck_carton.domain.models import (
    Carton,
    GridWorld,
    Store,
    Truck,
    Warehouse,
  )
  from truck_carton.packing.placement import (
    PlacementValidator,
  )
  from truck_carton.packing.space3d import Space3D


@dataclass
class RoutingCandidate:
  '''A single routing destination for a truck.

  Attributes:
    location_type: Facility type of destination.
    location_id: Warehouse or store identifier.
    distance: Travel distance from the active truck.
    position: Grid position of the destination.
  '''

  location_type: CellType
  location_id: int
  distance: float
  position: tuple[int, int] = (0, 0)


class ActionManager:
  '''Computes and decodes packing and routing actions.

  Attributes:
    candidates: Current packing placement candidates.
    routing_candidates: Current routing candidates.
    total_action_size: Packing plus routing action count.
  '''

  def __init__(self, config: EnvironmentConfig) -> None:
    '''Initialize with environment size limits.

    Args:
      config: Environment configuration limits.
    '''
    self._config = config
    self.candidates: list[PlacementCandidate] = []
    self.routing_candidates: list[RoutingCandidate] = []

  @property
  def total_action_size(self) -> int:
    '''Return the unified discrete action space size.

    Returns:
      Packing candidate limit plus routing action limit.
    '''
    return self._config.max_candidates + self._config.max_routing_actions

  def compute_candidates(
    self,
    carton: Carton,
    trucks: list[Truck],
    spaces: list[Space3D],
    current_weights: list[float],
    validators: list[PlacementValidator],
    all_cartons: list[Carton] | None = None,
  ) -> None:
    '''Compute valid packing placements for a carton.

    Args:
      carton: Carton to place.
      trucks: Trucks that may receive the carton.
      spaces: Truck interior spaces matching trucks.
      current_weights: Current load per truck.
      validators: Placement validators matching trucks.
      all_cartons: Optional carton list for fragility checks.
    '''
    self.routing_candidates = []
    found: list[PlacementCandidate] = []
    for idx, truck in enumerate(trucks):
      if idx >= len(spaces) or idx >= len(validators):
        continue
      weight = current_weights[idx] if idx < len(current_weights) else 0.0
      positions = validators[idx].find_valid_positions(
        carton,
        weight,
        truck.max_weight,
        all_cartons,
      )
      for x, y, z, rotation in positions:
        dims = apply_rotation(
          carton.length,
          carton.width,
          carton.height,
          rotation,
        )
        found.append(
          PlacementCandidate(
            truck_id=truck.truck_id,
            x=x,
            y=y,
            z=z,
            rotation=rotation,
            oriented_dims=dims,
          )
        )
        if len(found) >= self._config.max_candidates:
          break
      if len(found) >= self._config.max_candidates:
        break
    self.candidates = found

  def compute_routing_candidates(
    self,
    truck: Truck,
    truck_idx: int,
    warehouses: list[Warehouse],
    stores: list[Store],
    grid_world: GridWorld | None,
    truck_cargo: object = None,
    delivered: set[int] | None = None,
    carton_lookup: dict[int, Carton] | None = None,
    warehouse_cartons: dict[int, list[int]] | None = None,
  ) -> None:
    '''Compute routing destinations for a truck.

    delivered / warehouse_cartons are accepted for
    call-site compatibility (packing_env.py passes
    them positionally) and reserved for future
    filtering; they are currently unused. Removing
    them would break that non-owned caller.

    Args:
      truck: Active truck.
      truck_idx: Index of the active truck.
      warehouses: Candidate warehouse facilities.
      stores: Candidate store facilities.
      grid_world: Grid world for distance lookup.
      truck_cargo: Cargo mapping or sequence per truck.
      delivered: Delivered carton identifiers (unused).
      carton_lookup: Carton lookup for store matching.
      warehouse_cartons: Warehouse carton counts (unused).
    '''
    self.candidates = []
    cargo_ids = self._cargo_ids(truck, truck_idx, truck_cargo)
    lookup = carton_lookup or {}
    found: list[RoutingCandidate] = []
    for warehouse in warehouses:
      dist = self._distance(truck, warehouse.position, grid_world)
      found.append(
        RoutingCandidate(
          location_type=CellType.WAREHOUSE,
          location_id=warehouse.warehouse_id,
          distance=dist,
          position=warehouse.position,
        )
      )
    for store in stores:
      if lookup and cargo_ids:
        matching = any(
          cid in lookup and lookup[cid].destination_store_id == store.store_id
          for cid in cargo_ids
        )
        if not matching:
          continue
      elif cargo_ids:
        continue
      else:
        continue
      dist = self._distance(truck, store.position, grid_world)
      found.append(
        RoutingCandidate(
          location_type=CellType.STORE,
          location_id=store.store_id,
          distance=dist,
          position=store.position,
        )
      )
    depot = grid_world.depot_position if grid_world is not None else (0, 0)
    found.append(
      RoutingCandidate(
        location_type=CellType.DEPOT,
        location_id=0,
        distance=self._distance(truck, depot, grid_world),
        position=depot,
      )
    )
    limit = self._config.max_routing_actions
    self.routing_candidates = found[:limit]

  def _cargo_ids(
    self,
    truck: Truck,
    truck_idx: int,
    truck_cargo: object,
  ) -> list[int]:
    '''Extract cargo identifiers for a truck.

    Args:
      truck: Active truck.
      truck_idx: Index of the active truck.
      truck_cargo: Cargo mapping or sequence.

    Returns:
      Cargo carton identifiers for the truck.
    '''
    if isinstance(truck_cargo, dict):
      ids = truck_cargo.get(truck.truck_id)
      if ids is None:
        ids = truck_cargo.get(truck_idx, [])
      return list(ids)
    if isinstance(truck_cargo, (list, tuple)):
      if truck_idx < len(truck_cargo):
        return list(truck_cargo[truck_idx])
      return []
    return []

  def _distance(
    self,
    truck: Truck,
    position: tuple[int, int],
    grid_world: GridWorld | None,
  ) -> float:
    '''Compute travel distance to a position.

    Args:
      truck: Active truck.
      position: Destination grid position.
      grid_world: Grid world for graph distances.

    Returns:
      Graph distance when available, else Manhattan.
    '''
    if grid_world is not None:
      facilities = grid_world.facility_positions
      try:
        src = facilities.index(truck.position)
        dst = facilities.index(position)
      except ValueError:
        src = -1
        dst = -1
      if src >= 0 and dst >= 0:
        matrix = grid_world.distance_matrix
        if src < matrix.shape[0] and dst < matrix.shape[1]:
          value = float(matrix[src, dst])
          if np.isfinite(value):
            return value
    return float(
      abs(truck.position[0] - position[0])
      + abs(truck.position[1] - position[1])
    )

  def decode_action(self, action: int) -> tuple[str, object] | None:
    '''Decode a discrete action index.

    Args:
      action: Discrete action index.

    Returns:
      Action type and candidate, or None when invalid.
    '''
    idx = int(action)
    if 0 <= idx < len(self.candidates):
      return ('packing', self.candidates[idx])
    offset = self._config.max_candidates
    routing_idx = idx - offset
    if 0 <= routing_idx < len(self.routing_candidates):
      return ('routing', self.routing_candidates[routing_idx])
    return None

  def get_action_mask(self) -> np.ndarray:
    '''Build the boolean mask over all actions.

    Returns:
      Boolean array marking valid action indices.
    '''
    mask = np.zeros(self.total_action_size, dtype=np.bool_)
    for idx in range(len(self.candidates)):
      mask[idx] = True
    offset = self._config.max_candidates
    for idx in range(len(self.routing_candidates)):
      mask[offset + idx] = True
    return mask
