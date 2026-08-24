#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Observation encoding: raw semantic features plus normalized vectors.'''

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from world_of_supply.rl.agents import consumer_agent_id, producer_agent_id

if TYPE_CHECKING:
  from world_of_supply.facility import FacilityCell
  from world_of_supply.world import World


class ObservationEncoder:
  '''Encodes facility states into raw feature dicts and flat vectors.

  The raw feature dict keeps human-interpretable semantics (used by scripted
  baselines and the ``info`` return); the vector form min-max normalizes the
  concatenation of all features, matching the legacy observation layout while
  guarding against zero-range division.

  Attributes:
    product_ids: Sorted global product ids (feature column order).
    facility_types: Mapping of facility class name to one-hot column.
    max_sources_per_facility: Widest supplier list in the scenario.
    episode_duration: Ticks per episode (for the global-time feature).
  '''

  def __init__(
      self,
      product_ids: list[str],
      facility_types: dict[str, int],
      max_sources_per_facility: int,
      episode_duration: int,
      reference_facility_count: int,
  ) -> None:
    '''Capture the dimensional context of the scenario.

    Args:
      product_ids: Global sorted product ids.
      facility_types: Facility class names to one-hot columns.
      max_sources_per_facility: Maximum number of suppliers any consumer has.
      episode_duration: Episode length in ticks.
      reference_facility_count: Number of facilities (global feature width).
    '''
    self.product_ids = list(product_ids)
    self.facility_types = dict(facility_types)
    self.max_sources_per_facility = max_sources_per_facility
    self.episode_duration = episode_duration
    self.reference_facility_count = reference_facility_count

  @property
  def total_dim(self) -> int:
    '''Return the flattened feature dimension per agent.

    Returns:
      int: Length of every encoded observation vector.
    '''
    n_products = len(self.product_ids)
    return (
        len(self.facility_types)          # facility_type one-hot
        + self.reference_facility_count   # facility_id one-hot
        + 1                               # is_positive_balance
        + n_products                      # bom_inputs
        + n_products                      # bom_outputs
        + 1                               # storage_capacity
        + n_products                      # storage_levels
        + 1                               # storage_utilization
        + 1                               # distributor_in_transit_orders
        + 1                               # distributor_in_transit_orders_qty
        + n_products * self.max_sources_per_facility   # source export mask
        + n_products * self.max_sources_per_facility   # in-transit orders
        + 1                               # global_time
        + self.reference_facility_count   # global storage utilization
    )

  def encode_facility(self, facility: 'FacilityCell', world: 'World') -> dict:
    '''Build the raw semantic feature dict for one facility.

    Args:
      facility: Facility to encode.
      world: World snapshot providing global features.

    Returns:
      dict: Feature name to scalar or flat list of numbers.
    '''
    state: dict = {}
    facility_type = [0] * len(self.facility_types)
    facility_type[self.facility_types[type(facility).__name__]] = 1
    state['facility_type'] = facility_type

    facility_id_one_hot = [0] * self.reference_facility_count
    facility_id_one_hot[facility.id_num - 1] = 1
    state['facility_id'] = facility_id_one_hot
    state['is_positive_balance'] = 1 if facility.economy.total_balance.total() > 0 else 0

    n_products = len(self.product_ids)
    state['bom_inputs'] = [facility.bom.inputs.get(pid, 0) for pid in self.product_ids]
    state['bom_outputs'] = [
        facility.bom.output_lot_size if pid == facility.bom.output_product_id else 0
        for pid in self.product_ids
    ]

    assert facility.storage is not None
    state['storage_capacity'] = facility.storage.max_capacity
    state['storage_levels'] = [facility.storage.stock_levels.get(pid, 0) for pid in self.product_ids]
    state['storage_utilization'] = (
        sum(state['storage_levels']) / state['storage_capacity'] if state['storage_capacity'] else 0.0
    )

    state['distributor_in_transit_orders'] = 0
    state['distributor_in_transit_orders_qty'] = 0
    if facility.distribution is not None:
      queue = facility.distribution.order_queue
      state['distributor_in_transit_orders'] = len(queue)
      state['distributor_in_transit_orders_qty'] = sum(order.quantity for order in queue)

    mask_width = n_products * self.max_sources_per_facility
    state['consumer_source_export_mask'] = [0] * mask_width
    state['consumer_in_transit_orders'] = [0] * mask_width
    if facility.consumer is not None:
      for i_source, source in enumerate(facility.consumer.sources):
        for i_product, product_id in enumerate(self.product_ids):
          index = i_source * n_products + i_product
          if source.bom.output_product_id == product_id:
            state['consumer_source_export_mask'][index] = 1
          booked = facility.consumer.open_orders.get(source.id, {})
          state['consumer_in_transit_orders'][index] = booked.get(product_id, 0)

    state['global_time'] = world.time_step / self.episode_duration
    utilization = []
    for other in world.facilities.values():
      capacity = other.storage.max_capacity if other.storage is not None else 1
      used = other.storage.used_capacity() if other.storage is not None else 0
      utilization.append(used / capacity if capacity else 0.0)
    state['global_storage_utilization'] = utilization
    return state

  def encode_world(self, world: 'World') -> tuple[dict[str, np.ndarray], dict[str, dict]]:
    '''Encode every facility for both of its agents.

    Args:
      world: World snapshot.

    Returns:
      tuple[dict[str, np.ndarray], dict[str, dict]]: Normalized vectors keyed
      by agent id, and the corresponding raw feature dicts.
    '''
    vectors: dict[str, np.ndarray] = {}
    raws: dict[str, dict] = {}
    for facility in world.facilities.values():
      raw = self.encode_facility(facility, world)
      raws[producer_agent_id(facility.id)] = raw
      raws[consumer_agent_id(facility.id)] = raw
      vectors[producer_agent_id(facility.id)] = normalize(self._flatten(raw))
      vectors[consumer_agent_id(facility.id)] = vectors[producer_agent_id(facility.id)]
    return vectors, raws

  def _flatten(self, raw: dict) -> np.ndarray:
    '''Concatenate all feature blocks into one float vector.

    Args:
      raw: Raw feature dict from :meth:`encode_facility`.

    Returns:
      np.ndarray: Flat float32 vector of length :attr:`total_dim`.
    '''
    pieces: list[list[float]] = []
    for value in raw.values():
      pieces.append(value if isinstance(value, list) else [value])
    return np.hstack([np.array(piece, dtype=np.float64) for piece in pieces])


def normalize(vector: np.ndarray) -> np.ndarray:
  '''Min-max normalize a vector into ``[0, 1]`` safely.

  Constant vectors map to zeros instead of producing NaN.

  Args:
    vector: Input vector.

  Returns:
    np.ndarray: Normalized copy as float32.
  '''
  span = float(np.max(vector) - np.min(vector))
  if span == 0.0:
    return np.zeros_like(vector, dtype=np.float32)
  return ((vector - np.min(vector)) / span).astype(np.float32)
