#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Storage units: bounded inventories that hold product stock levels.'''

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from world_of_supply.base import Agent
from world_of_supply.economy import BalanceSheet

if TYPE_CHECKING:
  from world_of_supply.facility import FacilityControl


@dataclass
class StorageEconomy:
  '''Cost parameters of operating a storage unit.

  Attributes:
    unit_storage_cost: Cost per stored unit per time step.
  '''

  unit_storage_cost: int = 1

  def step_balance_sheet(self, used_capacity: int) -> BalanceSheet:
    '''Compute the holding cost for one time step.

    Args:
      used_capacity: Number of units currently stored.

    Returns:
      BalanceSheet: Loss proportional to the occupied capacity.
    '''
    return BalanceSheet(0, -used_capacity * self.unit_storage_cost)


class StorageUnit(Agent):
  '''A capacity-bounded inventory mapping product ids to stock counts.

  Attributes:
    max_capacity: Maximum number of units the storage can hold.
    stock_levels: Current stock per product id.
    economy: Cost parameters.
  '''

  def __init__(self, max_capacity: int, economy: StorageEconomy) -> None:
    '''Initialize an empty storage.

    Args:
      max_capacity: Maximum total number of units.
      economy: Holding-cost parameters.
    '''
    self.max_capacity = max_capacity
    self.stock_levels: dict[str, int] = {}
    self.economy = economy

  def used_capacity(self) -> int:
    '''Return the number of units currently stored.

    Returns:
      int: Sum over all product stock levels.
    '''
    return sum(self.stock_levels.values())

  def available_capacity(self) -> int:
    '''Return the remaining free capacity.

    Returns:
      int: ``max_capacity - used_capacity``.
    '''
    return self.max_capacity - self.used_capacity()

  def try_add_units(self, product_quantities: dict[str, int], all_or_nothing: bool = True) -> dict[str, int]:
    '''Deposit units into storage, respecting the capacity limit.

    Args:
      product_quantities: Mapping of product id to requested quantity.
      all_or_nothing: When True, deposit only if everything fits at once;
        otherwise deposit as much as fits, product by product.

    Returns:
      dict[str, int]: Quantities actually deposited per product id.
    '''
    if all_or_nothing and self.available_capacity() < sum(product_quantities.values()):
      return {}
    deposited: dict[str, int] = {}
    for product_id, quantity in product_quantities.items():
      quantity_deposited = min(self.available_capacity(), quantity)
      self.stock_levels[product_id] = self.stock_levels.get(product_id, 0) + quantity_deposited
      deposited[product_id] = quantity_deposited
    return deposited

  def try_take_units(self, product_quantities: dict[str, int]) -> bool:
    '''Withdraw units atomically; nothing is taken unless all is available.

    Args:
      product_quantities: Mapping of product id to requested quantity.

    Returns:
      bool: True when the full withdrawal succeeded.
    '''
    for product_id, quantity in product_quantities.items():
      if self.stock_levels.get(product_id, 0) < quantity:
        return False
    for product_id, quantity in product_quantities.items():
      self.stock_levels[product_id] -= quantity
    return True

  def take_available(self, product_id: str, quantity: int) -> int:
    '''Withdraw up to ``quantity`` units of one product.

    Args:
      product_id: Product to withdraw.
      quantity: Desired quantity.

    Returns:
      int: Quantity actually withdrawn.
    '''
    available = self.stock_levels.get(product_id, 0)
    actual = min(available, quantity)
    self.stock_levels[product_id] = available - actual
    return actual

  def act(self, control: 'FacilityControl | None' = None) -> BalanceSheet:
    '''Book holding costs for one time step.

    Args:
      control: Unused; accepted to satisfy the :class:`Agent` protocol.

    Returns:
      BalanceSheet: Storage cost of the current stock level.
    '''
    return self.economy.step_balance_sheet(self.used_capacity())
