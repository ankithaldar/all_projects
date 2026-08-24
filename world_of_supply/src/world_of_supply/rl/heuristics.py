#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Scripted heuristic agents operating at the RL action-vector level.

These reproduce the legacy hand-coded policies: producers fix a price per
facility class and keep a moderate production rate; consumers reorder the
most under-stocked BOM input from an exporting supplier. They consume the
*raw* observation features (not the normalized vectors).
'''

from __future__ import annotations

import random

import numpy as np

from world_of_supply.rl.agents import consumer_agent_id, producer_agent_id
from world_of_supply.world import World

_CLASS_PRICE_INDEX = {
    'SteelFactoryCell': 0,
    'LumberFactoryCell': 0,
    'ToyFactoryCell': 2,
    'WarehouseCell': 4,
    'RetailerCell': 6,
}
_PRODUCTION_RATE_INDEX = 2   # level 2 -> 4 lots
_ORDER_QUANTITY_INDEX = 4    # level 4 -> 8 units


class ScriptedProducer:
  '''Emits producer actions from raw facility features.'''

  def action(self, facility_type_name: str) -> np.ndarray:
    '''Compute the fixed producer action for a facility class.

    Args:
      facility_type_name: Facility class name.

    Returns:
      np.ndarray: ``[price_index, rate_index]``.
    '''
    price_index = _CLASS_PRICE_INDEX.get(facility_type_name, 0)
    return np.array([price_index, _PRODUCTION_RATE_INDEX])


class ScriptedConsumer:
  '''Emits consumer actions using fulfillment-ratio heuristics.'''

  def __init__(self, n_products: int, max_sources: int, seed: int | None = None) -> None:
    '''Capture feature layout dimensions.

    Args:
      n_products: Global product count.
      max_sources: Maximum suppliers any facility has.
      seed: Optional seed for supplier choice.
    '''
    self.n_products = n_products
    self.max_sources = max_sources
    self._rng = random.Random(seed)

  def action(self, raw: dict) -> np.ndarray:
    '''Pick what to order, from whom, and how much.

    Stops ordering when the facility is broke or its booked inventory
    exceeds storage capacity; otherwise targets the input with the lowest
    fulfillment ratio.

    Args:
      raw: Raw feature dict produced by :class:`ObservationEncoder`.

    Returns:
      np.ndarray: ``[product_index, source_index, quantity_index]``.
    '''
    if raw['is_positive_balance'] <= 0:
      return np.zeros(3, dtype=np.int64)

    inputs = np.asarray(raw['bom_inputs'])
    inventory = np.asarray(raw['storage_levels'])
    in_transit = np.asarray(raw['consumer_in_transit_orders']).reshape(self.max_sources, self.n_products).sum(axis=0)
    booked = inventory + in_transit

    if booked.sum() > raw['storage_capacity']:
      return np.zeros(3, dtype=np.int64)

    most_needed: int | None = None
    min_ratio = float('inf')
    for product_index, required in enumerate(inputs):
      if required > 0:
        ratio = float(booked[product_index]) / float(required)
        if ratio < min_ratio:
          min_ratio = ratio
          most_needed = product_index
    if most_needed is None:
      return np.zeros(3, dtype=np.int64)

    mask = np.asarray(raw['consumer_source_export_mask']).reshape(self.max_sources, self.n_products)
    exporters = [s for s in range(self.max_sources) if mask[s][most_needed] == 1]
    if not exporters:
      return np.zeros(3, dtype=np.int64)

    source = self._rng.choice(exporters)
    return np.array([most_needed, source, _ORDER_QUANTITY_INDEX], dtype=np.int64)


class ScriptedAgentController:
  '''Produces a full multi-agent action dict for the scripted baseline.'''

  def __init__(self, encoder, n_products: int, max_sources: int, seed: int | None = None) -> None:
    '''Wire the controller to the observation encoder's conventions.

    Args:
      encoder: ObservationEncoder providing raw feature dicts.
      n_products: Global product count.
      max_sources: Maximum suppliers per facility.
      seed: Optional seed for reproducibility.
    '''
    self.encoder = encoder
    self.producer = ScriptedProducer()
    self.consumer = ScriptedConsumer(n_products, max_sources, seed)

  def actions(self, world: World) -> dict[str, np.ndarray]:
    '''Build the action dictionary for every agent of the world.

    Args:
      world: Current world snapshot.

    Returns:
      dict[str, np.ndarray]: Actions keyed by agent id.
    '''
    _, raws = self.encoder.encode_world(world)
    actions: dict[str, np.ndarray] = {}
    for facility_id, facility in world.facilities.items():
      class_name = type(facility).__name__
      actions[producer_agent_id(facility_id)] = self.producer.action(class_name)
      actions[consumer_agent_id(facility_id)] = self.consumer.action(raws[consumer_agent_id(facility_id)])
    return actions
