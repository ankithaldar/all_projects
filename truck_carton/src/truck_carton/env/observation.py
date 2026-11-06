#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Dict observation encoding for the environment.

Trusted caller: ObservationContext objects are built by
the environment itself from trusted internal state.
Mutating returned observation arrays or context objects
does not affect the environment, but callers must not
mutate context cartons/trucks shared with the env.
This builder only sanitizes numeric values (NaN/inf).
'''

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
from gymnasium import spaces

from truck_carton.config import EnvironmentConfig

if TYPE_CHECKING:
  from truck_carton.domain.models import (
    Carton,
    ObservationContext,
  )


_INT_CAPS = (
  'max_trucks',
  'max_stores',
  'max_cartons',
  'max_truck_length',
  'max_truck_width',
  'max_truck_height',
  'max_candidates',
  'candidate_feature_dim',
  'max_grid_rows',
  'max_grid_cols',
  'max_warehouses',
  'max_locations',
  'max_routing_actions',
  'routing_feature_dim',
  'num_curriculum_stages',
)
_FLOAT_CAPS = ('max_weight_capacity', 'max_carton_weight')


def validate_environment_config(config: EnvironmentConfig) -> None:
  '''Validate EnvironmentConfig caps are sane.

  Args:
    config: Environment configuration to check.

  Raises:
    ValueError: If any cap is non-positive, non-finite,
      or the wrong type.
  '''
  for name in _INT_CAPS:
    value = getattr(config, name)
    if isinstance(value, bool) or not isinstance(value, int):
      raise ValueError(f'{name} must be an int (got {value!r})')
    if value <= 0:
      raise ValueError(f'{name} must be > 0 (got {value})')
  for name in _FLOAT_CAPS:
    value = getattr(config, name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
      raise ValueError(f'{name} must be numeric (got {value!r})')
    if not math.isfinite(float(value)) or float(value) <= 0:
      raise ValueError(f'{name} must be > 0 (got {value})')
  if config.max_locations < config.max_stores:
    raise ValueError('max_locations must cover max_stores')


def build_observation_space(
  config: EnvironmentConfig,
) -> spaces.Dict:
  '''Build the fixed Dict observation space.

  Args:
    config: Environment configuration limits.

  Returns:
    Dict observation space with Box components.

  Raises:
    ValueError: If config caps are not sane.
  '''
  validate_environment_config(config)
  return spaces.Dict(
    {
      'truck_grids': spaces.Box(
        0.0,
        1.0,
        shape=(
          config.max_trucks,
          config.max_truck_length
          * config.max_truck_width
          * config.max_truck_height,
        ),
        dtype=np.float32,
      ),
      'truck_meta': spaces.Box(
        0.0,
        1.0,
        shape=(config.max_trucks, 7),
        dtype=np.float32,
      ),
      'truck_routes': spaces.Box(
        0.0,
        1.0,
        shape=(config.max_trucks, config.max_stores),
        dtype=np.float32,
      ),
      'height_maps': spaces.Box(
        0.0,
        1.0,
        shape=(
          config.max_trucks,
          config.max_truck_length,
          config.max_truck_width,
        ),
        dtype=np.float32,
      ),
      'current_carton': spaces.Box(0.0, 1.0, shape=(7,), dtype=np.float32),
      'carton_queue': spaces.Box(
        0.0,
        1.0,
        shape=(config.max_cartons, 7),
        dtype=np.float32,
      ),
      'candidates': spaces.Box(
        0.0,
        1.0,
        shape=(
          config.max_candidates,
          config.candidate_feature_dim,
        ),
        dtype=np.float32,
      ),
      'facility_distances': spaces.Box(
        0.0,
        1.0,
        shape=(config.max_locations, config.max_locations),
        dtype=np.float32,
      ),
      'truck_positions': spaces.Box(
        0.0,
        1.0,
        shape=(config.max_trucks, 2),
        dtype=np.float32,
      ),
      'truck_states': spaces.Box(
        0.0,
        1.0,
        shape=(config.max_trucks,),
        dtype=np.float32,
      ),
      'warehouse_counts': spaces.Box(
        0.0,
        1.0,
        shape=(config.max_warehouses,),
        dtype=np.float32,
      ),
      'routing_candidates': spaces.Box(
        0.0,
        1.0,
        shape=(
          config.max_routing_actions,
          config.routing_feature_dim,
        ),
        dtype=np.float32,
      ),
      'global_info': spaces.Box(0.0, 1.0, shape=(6,), dtype=np.float32),
    }
  )


class ObservationBuilder:
  '''Encodes environment state into Dict observations.'''

  def __init__(self, config: EnvironmentConfig) -> None:
    '''Initialize with environment limits.

    Args:
      config: Environment configuration limits.

    Raises:
      ValueError: If config caps are not sane.
    '''
    validate_environment_config(config)
    self._cfg = config

  def build(self, ctx: ObservationContext) -> dict:
    '''Encode an observation context.

    Args:
      ctx: Observation context snapshot.

    Returns:
      Dict observation with fixed shapes and values
      in the unit interval.
    '''
    obs = {
      'truck_grids': self._truck_grids(ctx),
      'truck_meta': self._truck_meta(ctx),
      'truck_routes': self._truck_routes(ctx),
      'height_maps': self._height_maps(ctx),
      'current_carton': self._carton_features(ctx.current_carton),
      'carton_queue': self._carton_queue(ctx),
      'candidates': self._candidates(ctx),
      'facility_distances': self._facility_distances(ctx),
      'truck_positions': self._truck_positions(ctx),
      'truck_states': self._truck_states(ctx),
      'warehouse_counts': self._warehouse_counts(ctx),
      'routing_candidates': self._routing(ctx),
      'global_info': self._global_info(ctx),
    }
    return {
      key: np.clip(
        np.nan_to_num(
          np.asarray(value, dtype=np.float32),
          nan=0.0,
          posinf=1.0,
          neginf=0.0,
        ),
        0.0,
        1.0,
      )
      for key, value in obs.items()
    }

  def _truck_grids(self, ctx: ObservationContext) -> np.ndarray:
    '''Encode flattened truck occupancy grids.

    Args:
      ctx: Observation context snapshot.

    Returns:
      Normalized flattened grids per truck.
    '''
    cfg = self._cfg
    size = cfg.max_truck_length * cfg.max_truck_width * cfg.max_truck_height
    out = np.zeros((cfg.max_trucks, size), dtype=np.float32)
    max_id = 0
    for space in ctx.spaces:
      if space.grid.size:
        max_id = max(max_id, int(space.grid.max()))
    for idx, space in enumerate(ctx.spaces[: cfg.max_trucks]):
      if max_id <= 0:
        break
      flat = space.get_normalized_grid(max_id).ravel()
      out[idx, : min(size, flat.size)] = flat[:size]
    return out

  def _truck_meta(self, ctx: ObservationContext) -> np.ndarray:
    '''Encode per truck metadata features.

    Args:
      ctx: Observation context snapshot.

    Returns:
      Metadata matrix per truck.
    '''
    cfg = self._cfg
    out = np.zeros((cfg.max_trucks, 7), dtype=np.float32)
    for idx, truck in enumerate(ctx.trucks[: cfg.max_trucks]):
      weight = (
        ctx.current_weights[idx] if idx < len(ctx.current_weights) else 0.0
      )
      occupancy = 0.0
      if idx < len(ctx.spaces):
        occupancy = ctx.spaces[idx].get_occupancy_ratio()
      out[idx] = [
        truck.length / cfg.max_truck_length,
        truck.width / cfg.max_truck_width,
        truck.height / cfg.max_truck_height,
        weight / cfg.max_weight_capacity,
        truck.max_weight / cfg.max_weight_capacity,
        occupancy,
        len(truck.visited_stores) / max(cfg.max_stores, 1),
      ]
    return out

  def _truck_routes(self, ctx: ObservationContext) -> np.ndarray:
    '''Encode truck store route membership.

    Args:
      ctx: Observation context snapshot.

    Returns:
      Binary route matrix per truck.
    '''
    cfg = self._cfg
    out = np.zeros(
      (cfg.max_trucks, cfg.max_stores),
      dtype=np.float32,
    )
    for idx, truck in enumerate(ctx.trucks[: cfg.max_trucks]):
      for store in truck.route:
        if 0 <= store < cfg.max_stores:
          out[idx, store] = 1.0
    return out

  def _height_maps(self, ctx: ObservationContext) -> np.ndarray:
    '''Encode per truck height maps.

    Args:
      ctx: Observation context snapshot.

    Returns:
      Normalized height map per truck.
    '''
    cfg = self._cfg
    out = np.zeros(
      (
        cfg.max_trucks,
        cfg.max_truck_length,
        cfg.max_truck_width,
      ),
      dtype=np.float32,
    )
    for idx, space in enumerate(ctx.spaces[: cfg.max_trucks]):
      height_map = space.get_height_map()
      rows = min(cfg.max_truck_length, height_map.shape[0])
      cols = min(cfg.max_truck_width, height_map.shape[1])
      out[idx, :rows, :cols] = height_map[:rows, :cols] / cfg.max_truck_height
    return out

  def _carton_features(self, carton: Carton | None) -> np.ndarray:
    '''Encode a single carton feature vector.

    Args:
      carton: Carton to encode, or None.

    Returns:
      Normalized carton feature vector.
    '''
    cfg = self._cfg
    if carton is None:
      return np.zeros(7, dtype=np.float32)
    return np.array(
      [
        carton.length / cfg.max_truck_length,
        carton.width / cfg.max_truck_width,
        carton.height / cfg.max_truck_height,
        carton.weight / cfg.max_carton_weight,
        1.0 if carton.is_fragile else 0.0,
        carton.priority / 3.0,
        carton.destination_store_id / max(cfg.max_stores - 1, 1),
      ],
      dtype=np.float32,
    )

  def _carton_queue(self, ctx: ObservationContext) -> np.ndarray:
    '''Encode remaining unplaced cartons.

    Args:
      ctx: Observation context snapshot.

    Returns:
      Normalized carton queue matrix.
    '''
    cfg = self._cfg
    out = np.zeros((cfg.max_cartons, 7), dtype=np.float32)
    for idx, carton in enumerate(ctx.remaining_cartons[: cfg.max_cartons]):
      out[idx] = self._carton_features(carton)
    return out

  def _candidates(self, ctx: ObservationContext) -> np.ndarray:
    '''Encode packing candidate features.

    Args:
      ctx: Observation context snapshot.

    Returns:
      Normalized packing candidate matrix.
    '''
    cfg = self._cfg
    out = np.zeros(
      (
        cfg.max_candidates,
        cfg.candidate_feature_dim,
      ),
      dtype=np.float32,
    )
    for idx, cand in enumerate(ctx.packing_candidates[: cfg.max_candidates]):
      truck_idx = cand.truck_id
      occupancy = 0.0
      if 0 <= truck_idx < len(ctx.spaces):
        occupancy = ctx.spaces[truck_idx].get_occupancy_ratio()
      weight = 0.0
      if 0 <= truck_idx < len(ctx.current_weights):
        weight = ctx.current_weights[truck_idx]
      dl, dw, dh = cand.oriented_dims
      volume = float(dl * dw * dh)
      out[idx] = [
        truck_idx / cfg.max_trucks,
        cand.x / cfg.max_truck_length,
        cand.y / cfg.max_truck_width,
        cand.z / cfg.max_truck_height,
        dl / cfg.max_truck_length,
        dw / cfg.max_truck_width,
        dh / cfg.max_truck_height,
        float(cand.rotation) / 5.0,
        occupancy,
        weight / cfg.max_weight_capacity,
        (cand.x + dl) / cfg.max_truck_length,
        (cand.y + dw) / cfg.max_truck_width,
        (cand.z + dh) / cfg.max_truck_height,
        1.0 if cand.z == 0 else 0.0,
        1.0 if cand.x == 0 else 0.0,
        1.0 if cand.y == 0 else 0.0,
        (cand.x + cand.y) / (cfg.max_truck_length + cfg.max_truck_width),
        volume
        / max(
          cfg.max_truck_length * cfg.max_truck_width * cfg.max_truck_height,
          1,
        ),
        dl / max(dw, 1),
        dh / max(cfg.max_truck_height, 1),
      ]
    return out

  def _facility_distances(self, ctx: ObservationContext) -> np.ndarray:
    '''Encode pairwise facility distances.

    Args:
      ctx: Observation context snapshot.

    Returns:
      Normalized facility distance matrix.
    '''
    cfg = self._cfg
    out = np.zeros(
      (cfg.max_locations, cfg.max_locations),
      dtype=np.float32,
    )
    if ctx.grid_world is None:
      return out
    matrix = ctx.grid_world.distance_matrix
    finite = matrix[np.isfinite(matrix)]
    scale = float(finite.max()) if finite.size else 1.0
    scale = max(scale, 1.0)
    rows = min(cfg.max_locations, matrix.shape[0])
    cols = min(cfg.max_locations, matrix.shape[1])
    block = matrix[:rows, :cols].astype(np.float32)
    block = np.where(np.isfinite(block), block / scale, 1.0)
    out[:rows, :cols] = block
    return out

  def _truck_positions(self, ctx: ObservationContext) -> np.ndarray:
    '''Encode normalized truck grid positions.

    Args:
      ctx: Observation context snapshot.

    Returns:
      Normalized position matrix per truck.
    '''
    cfg = self._cfg
    out = np.zeros((cfg.max_trucks, 2), dtype=np.float32)
    for idx, truck in enumerate(ctx.trucks[: cfg.max_trucks]):
      out[idx] = [
        truck.position[0] / cfg.max_grid_rows,
        truck.position[1] / cfg.max_grid_cols,
      ]
    return out

  def _truck_states(self, ctx: ObservationContext) -> np.ndarray:
    '''Encode normalized truck states.

    Args:
      ctx: Observation context snapshot.

    Returns:
      Normalized state vector per truck.
    '''
    cfg = self._cfg
    out = np.zeros(cfg.max_trucks, dtype=np.float32)
    for idx, truck in enumerate(ctx.trucks[: cfg.max_trucks]):
      out[idx] = float(truck.state) / 2.0
    return out

  def _warehouse_counts(self, ctx: ObservationContext) -> np.ndarray:
    '''Encode remaining cartons per warehouse.

    Args:
      ctx: Observation context snapshot.

    Returns:
      Normalized warehouse count vector.
    '''
    cfg = self._cfg
    out = np.zeros(cfg.max_warehouses, dtype=np.float32)
    for warehouse in ctx.warehouses:
      wid = warehouse.warehouse_id
      if 0 <= wid < cfg.max_warehouses:
        out[wid] = len(ctx.warehouse_cartons.get(wid, [])) / cfg.max_cartons
    return out

  def _routing(self, ctx: ObservationContext) -> np.ndarray:
    '''Encode routing candidate features.

    Args:
      ctx: Observation context snapshot.

    Returns:
      Normalized routing candidate matrix.
    '''
    cfg = self._cfg
    out = np.zeros(
      (
        cfg.max_routing_actions,
        cfg.routing_feature_dim,
      ),
      dtype=np.float32,
    )
    scale = 1.0
    if ctx.grid_world is not None:
      finite = ctx.grid_world.distance_matrix[
        np.isfinite(ctx.grid_world.distance_matrix)
      ]
      if finite.size:
        scale = max(float(finite.max()), 1.0)
    for idx, cand in enumerate(
      ctx.routing_candidates[: cfg.max_routing_actions]
    ):
      out[idx] = [
        cand.location_id / cfg.max_locations,
        cand.distance / scale,
        float(cand.location_type) / 4.0,
        cand.position[0] / cfg.max_grid_rows,
        cand.position[1] / cfg.max_grid_cols,
        1.0 if cand.location_type == 2 else 0.0,
        1.0 if cand.location_type == 4 else 0.0,
      ]
    return out

  def _global_info(self, ctx: ObservationContext) -> np.ndarray:
    '''Encode global episode progress features.

    Args:
      ctx: Observation context snapshot.

    Returns:
      Normalized global feature vector.
    '''
    cfg = self._cfg
    total = max(ctx.total_cartons, 1)
    denom = max(cfg.num_curriculum_stages - 1, 1)
    return np.array(
      [
        ctx.num_placed / total,
        ctx.stage_index / denom,
        ctx.num_delivered / total,
        ctx.step_count / max(ctx.max_steps, 1),
        ctx.total_travel / max(total * 10.0, 1.0),
        ctx.active_truck_idx / cfg.max_trucks,
      ],
      dtype=np.float32,
    )
