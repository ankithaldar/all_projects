#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Observation encoding: raw semantic features plus normalized vectors.'''

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from world_of_supply.rl.agents import (
    consumer_agent_id,
    facility_type_name,
    producer_agent_id,
)

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

    Returns:
      None: The encoder is mutated in place.
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
    n_sources = self.max_sources_per_facility
    return (
        len(self.facility_types)          # facility_type one-hot
        + self.reference_facility_count   # facility_id one-hot
        + 1                               # is_positive_balance
        + n_products                      # bom_inputs
        + n_products                      # bom_outputs
        + 1                               # storage_capacity
        + n_products                      # storage_levels
        + 1                               # storage_utilization
        + 1                               # in-transit order count
        + 1                               # in-transit order qty
        + n_products * n_sources          # source export mask
        + n_products * n_sources          # in-transit orders
        + 1                               # global_time
        + self.reference_facility_count   # global storage utilization
    )

  def _encode_identity(self, facility: 'FacilityCell') -> dict:
    '''Encode facility identity and solvency features.

    Args:
      facility: Facility to encode.

    Returns:
      dict: Identity feature names to scalars or one-hot lists.
    '''
    state: dict = {}
    facility_type = [0] * len(self.facility_types)
    facility_type[self.facility_types[facility_type_name(facility)]] = 1
    state['facility_type'] = facility_type
    one_hot = [0] * self.reference_facility_count
    one_hot[facility.id_num - 1] = 1
    state['facility_id'] = one_hot
    balance = facility.economy.total_balance.total()
    state['is_positive_balance'] = 1 if balance > 0 else 0
    return state

  def _encode_bom(self, facility: 'FacilityCell') -> dict:
    '''Encode bill-of-materials input and output features.

    Args:
      facility: Facility to encode.

    Returns:
      dict: BOM feature names to flat lists.
    '''
    state: dict = {}
    inputs = [
        facility.bom.inputs.get(pid, 0) for pid in self.product_ids
    ]
    state['bom_inputs'] = inputs
    outputs = []
    for pid in self.product_ids:
      is_output = pid == facility.bom.output_product_id
      size = facility.bom.output_lot_size if is_output else 0
      outputs.append(size)
    state['bom_outputs'] = outputs
    return state

  def _encode_storage(self, facility: 'FacilityCell') -> dict:
    '''Encode storage capacity and utilization features.

    Args:
      facility: Facility to encode.

    Returns:
      dict: Storage feature names to scalars and lists.
    '''
    state: dict = {}
    assert facility.storage is not None
    state['storage_capacity'] = facility.storage.max_capacity
    levels = [
        facility.storage.stock_levels.get(pid, 0)
        for pid in self.product_ids
    ]
    state['storage_levels'] = levels
    capacity = state['storage_capacity']
    state['storage_utilization'] = sum(levels) / capacity if capacity else 0.0
    return state

  def _encode_distribution(self, facility: 'FacilityCell') -> dict:
    '''Encode distributor in-transit order features.

    Args:
      facility: Facility to encode.

    Returns:
      dict: Distribution feature names to scalars.
    '''
    state: dict = {}
    state['distributor_in_transit_orders'] = 0
    state['distributor_in_transit_orders_qty'] = 0
    if facility.distribution is not None:
      queue = facility.distribution.order_queue
      state['distributor_in_transit_orders'] = len(queue)
      state['distributor_in_transit_orders_qty'] = sum(
          order.quantity for order in queue
      )
    return state

  def _encode_consumer(self, facility: 'FacilityCell') -> dict:
    '''Encode consumer source and open-order features.

    Args:
      facility: Facility to encode.

    Returns:
      dict: Consumer feature names to flat lists.
    '''
    state: dict = {}
    n_products = len(self.product_ids)
    mask_width = n_products * self.max_sources_per_facility
    mask = [0] * mask_width
    orders = [0] * mask_width
    if facility.consumer is not None:
      sources = facility.consumer.sources
      open_orders = facility.consumer.open_orders
      for i_source, source in enumerate(sources):
        for i_product, product_id in enumerate(self.product_ids):
          index = i_source * n_products + i_product
          if source.bom.output_product_id == product_id:
            mask[index] = 1
          booked = open_orders.get(source.id, {})
          orders[index] = booked.get(product_id, 0)
    state['consumer_source_export_mask'] = mask
    state['consumer_in_transit_orders'] = orders
    return state

  def _encode_global(self, world: 'World') -> dict:
    '''Encode global time and storage utilization features.

    Args:
      world: World snapshot providing global features.

    Returns:
      dict: Global feature names to scalars and lists.
    '''
    state: dict = {}
    state['global_time'] = world.time_step / self.episode_duration
    utilization = []
    for other in world.facilities.values():
      if other.storage is not None:
        capacity = other.storage.max_capacity
      else:
        capacity = 1
      if other.storage is not None:
        used = other.storage.used_capacity()
      else:
        used = 0
      utilization.append(used / capacity if capacity else 0.0)
    state['global_storage_utilization'] = utilization
    return state

  def encode_facility(self, facility: 'FacilityCell', world: 'World') -> dict:
    '''Build the raw semantic feature dict for one facility.

    Args:
      facility: Facility to encode.
      world: World snapshot providing global features.

    Returns:
      dict: Feature name to scalar or flat list of numbers.
    '''
    state: dict = {}
    state.update(self._encode_identity(facility))
    state.update(self._encode_bom(facility))
    state.update(self._encode_storage(facility))
    state.update(self._encode_distribution(facility))
    state.update(self._encode_consumer(facility))
    state.update(self._encode_global(world))
    return state

  def encode_world(
      self, world: 'World'
  ) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
    '''Encode every facility for both of its agents.

    Args:
      world: World snapshot.

    Returns:
      tuple[dict[str, np.ndarray], dict[str, dict]]: Normalized vectors
      keyed by agent id, and the corresponding raw feature dicts.
    '''
    vectors: dict[str, np.ndarray] = {}
    raws: dict[str, dict] = {}
    for facility in world.facilities.values():
      raw = self.encode_facility(facility, world)
      producer_id = producer_agent_id(facility.id)
      consumer_id = consumer_agent_id(facility.id)
      raws[producer_id] = raw
      raws[consumer_id] = raw
      vector = normalize(self._flatten(raw))
      vectors[producer_id] = vector
      vectors[consumer_id] = vector
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
