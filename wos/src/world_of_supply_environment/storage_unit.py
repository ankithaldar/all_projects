#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Storage Unit'''


# imports
from collections import Counter
from dataclasses import dataclass

from .agent import Agent
from .balance_sheet import BalanceSheet
# imports


class StorageUnit(Agent):

  '''Storage Unit'''

  @dataclass
  class Economy:
    unit_storage_cost: int    # cost per unit per time step

    def step_balance_sheet(self, storage):
      return BalanceSheet(
        0,
        -storage.used_capacity() * self.unit_storage_cost
      )

  @dataclass
  class Config:
    max_storage_capacity: int
    unit_storage_cost: int

  def __init__(self, max_capacity, economy):
    self.max_capacity = max_capacity
    self.stock_levels = Counter()
    self.economy = economy

  def used_capacity(self):
    return sum(self.stock_levels.values())

  def available_capacity(self):
    return self.max_capacity - self.used_capacity()

  def try_add_units(self, product_quantities, all_or_nothing = True) -> dict:

    # validation
    if (
      all_or_nothing and
      self.available_capacity() < sum(product_quantities.values())
    ):
      return {}

    # depositing
    unloaded_quantities = {}
    for p_id, q in product_quantities.items():
      unloading_qty = min(self.available_capacity(), q)
      self.stock_levels[p_id] += unloading_qty
      unloaded_quantities[p_id] = unloading_qty

    return unloaded_quantities

  def try_take_units(self, product_quantities):
    # validation
    for p_id, q in product_quantities.items():
      if self.stock_levels[p_id] < q:
        return False
    # withdrawal
    for p_id, q in product_quantities.items():
      self.stock_levels[p_id] -= q
    return True

  def take_available(self, product_id, quantity):
    available = self.stock_levels[product_id]
    actual = min(available, quantity)
    self.stock_levels[product_id] -= actual
    return actual

  def act(self, control = None):
    return self.economy.step_balance_sheet(self)
