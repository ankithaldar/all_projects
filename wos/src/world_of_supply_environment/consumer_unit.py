#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Consumer Unit'''


# imports
from collections import Counter
from dataclasses import dataclass
#   script imports
from .agent import Agent
from .balance_sheet import BalanceSheet
from .distribution_unit import DistributionUnit

# imports


class ConsumerUnit(Agent):
  '''Consumer Unit'''

  @dataclass
  class Economy:
    total_units_purchased: int = 0
    total_units_received: int = 0

  @dataclass
  class Control:
    consumer_product_id: int  # what to purchase
    consumer_source_id: int   # where to purchase
    consumer_quantity: int    # how many to purchase

  @dataclass
  class Config:
    sources: list

  def __init__(self, facility, sources):
    self.facility = facility
    self.sources = sources
    self.open_orders = {}
    self.economy = ConsumerUnit.Economy()

  def on_order_reception(self, source_id, product_id, quantity):
    self.economy.total_units_received += quantity
    self._update_open_orders(source_id, product_id, -quantity)

  def act(self, control):
    if (control is None or
      control.consumer_product_id is None or
      control.consumer_quantity <= 0
    ):
      return BalanceSheet()

    source_obj = self.sources[control.consumer_source_id]
    self._update_open_orders(
      source_obj.id,
      control.consumer_product_id,
      control.consumer_quantity
    )
    order = DistributionUnit.Order(
      self.facility,
      control.consumer_product_id,
      control.consumer_quantity
    )
    order_cost = source_obj.distribution.place_order( order )
    self.economy.total_units_purchased += control.consumer_quantity

    return BalanceSheet(0, order_cost)

  def _update_open_orders(self, source_id, product_id, qty_delta):
    if qty_delta > 0:
      if source_id not in self.open_orders:
        self.open_orders[source_id] = Counter()
      self.open_orders[source_id][product_id] += qty_delta
    else:
      self.open_orders[source_id][product_id] += qty_delta
      self.open_orders[source_id] += Counter() # remove zeros
      if len(self.open_orders[source_id]) == 0:
        del self.open_orders[source_id]
