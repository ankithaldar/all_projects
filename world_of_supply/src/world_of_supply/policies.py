#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Hand-coded facility control policies (Strategy pattern).

These operate directly on :class:`World` objects and produce
:class:`Control` dictionaries — no reinforcement-learning stack
is required. They serve as the behavioral baseline the RL
agents are compared against.
'''

from __future__ import annotations

import random
from typing import Protocol

from world_of_supply.facility import (
    FacilityCell,
    FacilityControl,
    RawMaterialsFactoryCell,
    RetailerCell,
    ValueAddFactoryCell,
    WarehouseCell,
)
from world_of_supply.world import Control, World

_DEFAULT_PRICES = {
    RawMaterialsFactoryCell: 400,
    ValueAddFactoryCell: 1000,
    WarehouseCell: 1400,
    RetailerCell: 1800,
}
_PRODUCTION_RATE = 4
_ORDER_QUANTITY = 8


class ControlPolicy(Protocol):
  '''Anything translating world state into per-facility controls.

  Attributes:
    None: Policy holds no per-facility state by contract.
  '''

  def compute_control(self, world: World) -> Control:
    '''Produce controls for all facilities of a world.

    Args:
      world: Current world snapshot.

    Returns:
      Control: Controls keyed by facility id.

    Raises:
      NotImplementedError: Policies must implement this method.
    '''
    raise NotImplementedError


class ScriptedSupplyChainPolicy:
  '''Heuristic baseline mirroring the legacy SimpleControlPolicy.

  Sets class-specific fixed prices, keeps production at a moderate
  rate, and reorders the least-stocked BOM input from a random
  exporting source while solvent and with storage head-room.

  Attributes:
    None: Random state is private (``_rng``).
  '''

  def __init__(self, seed: int | None = None) -> None:
    '''Create the policy.

    Args:
      seed: Optional seed for reproducible supplier choices.

    Returns:
      None: The policy is mutated in place.
    '''
    self._rng = random.Random(seed)

  def compute_control(self, world: World) -> Control:
    '''Build controls for every facility in the world.

    Args:
      world: Current world snapshot.

    Returns:
      Control: One FacilityControl per facility id.
    '''
    controls: dict[str, FacilityControl] = {}
    for facility in world.facilities.values():
      price = self._price_for(facility)
      product_id, source_id = self._select_source(facility)
      quantity = _ORDER_QUANTITY if product_id is not None else 0
      controls[facility.id] = FacilityControl(
          unit_price=price,
          production_rate=_PRODUCTION_RATE,
          consumer_product_id=product_id,
          consumer_source_id=source_id,
          consumer_quantity=quantity,
      )
    return Control(controls)

  def _price_for(self, facility: FacilityCell) -> int:
    '''Look up the fixed offer price for a facility type.

    Args:
      facility: Facility in question.

    Returns:
      int: Price for the most specific matching class.
    '''
    for clazz in reversed(facility.__class__.__mro__):
      if clazz in _DEFAULT_PRICES:
        return _DEFAULT_PRICES[clazz]
    return 0

  def _select_source(
      self, facility: FacilityCell,
  ) -> tuple[str | None, int | None]:
    '''Pick product and supplier to reorder from, if any.

    Stops ordering when out of money or when booked inventory
    exceeds capacity. Otherwise orders the lowest-fulfilled input.

    Args:
      facility: Ordering facility.

    Returns:
      tuple[str | None, int | None]: Product id and supplier index,
        or ``(None, None)`` when no order should be placed.
    '''
    consumer = facility.consumer
    if consumer is None or not facility.economy.total_balance.total() > 0:
      return (None, None)
    inputs = dict(facility.bom.inputs)
    open_orders: dict[str, int] = {}
    for counter in consumer.open_orders.values():
      for product_id, quantity in counter.items():
        prior = open_orders.get(product_id, 0)
        open_orders[product_id] = prior + quantity
    assert facility.storage is not None
    storage = facility.storage
    used = storage.used_capacity()
    booked_total = used + sum(open_orders.values())
    if booked_total > storage.max_capacity:
      return (None, None)
    most_needed_product: str | None = None
    min_ratio = float('inf')
    for product_id, required in inputs.items():
      booked = open_orders.get(product_id, 0)
      booked += storage.stock_levels.get(product_id, 0)
      ratio = booked / required
      if ratio < min_ratio:
        min_ratio = ratio
        most_needed_product = product_id
    exporters = self.find_exporting_sources(facility, most_needed_product)
    if not exporters:
      return (None, None)
    return (most_needed_product, self._rng.choice(exporters))

  @staticmethod
  def find_exporting_sources(
      facility: FacilityCell,
      product_id: str | None,
  ) -> list[int]:
    '''List indices of sources that output the requested product.

    Args:
      facility: Ordering facility.
      product_id: Desired product, or ``None``.

    Returns:
      list[int]: Indices into the consumer sources list.
    '''
    if product_id is None or facility.consumer is None:
      return []
    exporters = []
    for index, source in enumerate(facility.consumer.sources):
      if source.bom.output_product_id == product_id:
        exporters.append(index)
    return exporters
