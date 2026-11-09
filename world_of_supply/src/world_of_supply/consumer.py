#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Consumer units: facility-level buyers ordering upstream.'''

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

from world_of_supply.base import Agent
from world_of_supply.distribution import Order
from world_of_supply.economy import BalanceSheet

if TYPE_CHECKING:
  from world_of_supply.facility import FacilityCell, FacilityControl


class ConsumerEconomy:
  '''Lifetime purchasing statistics of a consumer unit.

  Attributes:
    total_units_purchased: Units ordered since creation.
    total_units_received: Units actually delivered so far.
  '''

  def __init__(self) -> None:
    '''Zero-initialize both counters.

    Returns:
      None: The economy is mutated in place.
    '''
    self.total_units_purchased = 0
    self.total_units_received = 0


class ConsumerUnit(Agent):
  '''Places orders with upstream facilities and tracks open orders.

  Attributes:
    facility: Owning facility that receives the goods.
    sources: Upstream facilities this consumer may order from.
    open_orders: Mapping of source id to outstanding quantities.
    economy: Lifetime statistics.
  '''

  def __init__(
      self,
      facility: 'FacilityCell',
      sources: list['FacilityCell'],
  ) -> None:
    '''Attach the consumer to a facility.

    Args:
      facility: Owning facility.
      sources: Allowed suppliers.

    Returns:
      None: The unit is mutated in place.
    '''
    self.facility = facility
    self.sources = list(sources)
    self.open_orders: dict[str, Counter] = {}
    self.economy = ConsumerEconomy()

  def _shift_open_order(
      self,
      source_id: str,
      product_id: str,
      delta: int,
  ) -> None:
    '''Apply a signed delta to the open-order book, prune zeros.

    Args:
      source_id: Supplier whose book is updated.
      product_id: Ordered product.
      delta: Positive when ordering, negative when receiving.

    Returns:
      None: The open-order book is mutated in place.
    '''
    counter = self.open_orders.setdefault(source_id, Counter())
    counter[product_id] += delta
    pruned = counter + Counter()
    if len(pruned) == 0:
      del self.open_orders[source_id]
    else:
      self.open_orders[source_id] = pruned

  def on_order_reception(
      self,
      source_id: str,
      product_id: str,
      quantity: int,
  ) -> None:
    '''Record a delivery and clear the matching open-order quantity.

    Args:
      source_id: Id of the shipping facility.
      product_id: Delivered product.
      quantity: Number of units received.

    Returns:
      None: The open-order book is mutated in place.
    '''
    self.economy.total_units_received += quantity
    self._shift_open_order(source_id, product_id, -quantity)

  def act(
      self,
      control: 'FacilityControl | None' = None,
  ) -> BalanceSheet:
    '''Place an order at the selected upstream facility.

    Args:
      control: Carries product id, source id, and quantity;
        missing or non-positive entries are no-ops.

    Returns:
      BalanceSheet: Loss equal to the supplier prepayment.

    Raises:
      IndexError: If the requested source index is out of range.
    '''
    if control is None or control.consumer_product_id is None:
      return BalanceSheet()
    if not control.consumer_quantity:
      return BalanceSheet()
    source = self.sources[control.consumer_source_id]
    order = Order(
        self.facility,
        control.consumer_product_id,
        control.consumer_quantity,
    )
    assert source.distribution is not None
    payment = source.distribution.place_order(order)
    self.economy.total_units_purchased += control.consumer_quantity
    source_id = source.id
    product_id = control.consumer_product_id
    quantity = control.consumer_quantity
    self._shift_open_order(source_id, product_id, quantity)
    return BalanceSheet(0, -payment)
