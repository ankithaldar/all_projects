#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Seller units: expose inventory to market demand through a price curve.'''

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from world_of_supply.base import Agent
from world_of_supply.economy import BalanceSheet

if TYPE_CHECKING:
  from world_of_supply.facility import FacilityCell, FacilityControl


@dataclass
class SellerEconomy:
  '''Demand-curve parameters and sales statistics of a seller unit.

  Attributes:
    price_demand_intercept: Demand at price zero.
    price_demand_slope: Demand reduction per currency unit of price.
    unit_price: Currently offered price.
    total_units_sold: Lifetime sales counter.
  '''

  price_demand_intercept: float = 50.0
  price_demand_slope: float = 0.005
  unit_price: int = 0
  total_units_sold: int = 0

  def market_demand(self, unit_price: int) -> int:
    '''Evaluate the linear demand curve at a price.

    Args:
      unit_price: Offered price.

    Returns:
      int: Non-negative demand ``intercept - slope * price``.
    '''
    return max(0, int(self.price_demand_intercept - self.price_demand_slope * unit_price))


class SellerUnit(Agent):
  '''Sells the facility output product to an abstract market each tick.

  Attributes:
    facility: Owning facility providing storage and output product.
    economy: Demand-curve parameters.
  '''

  def __init__(self, facility: 'FacilityCell', economy: SellerEconomy) -> None:
    '''Attach the seller to a facility.

    Args:
      facility: Owning facility.
      economy: Demand-curve parameters.
    '''
    self.facility = facility
    self.economy = economy

  def act(self, control: 'FacilityControl | None') -> BalanceSheet:
    '''Sell as much stock as the market demands at the current price.

    Args:
      control: Carries ``unit_price`` updates; ``None`` keeps the price.

    Returns:
      BalanceSheet: Revenue proportional to the units actually sold.
    '''
    if control is not None and control.unit_price is not None:
      self.economy.unit_price = control.unit_price
    product_id = self.facility.bom.output_product_id
    demand = self.economy.market_demand(self.economy.unit_price)
    sold_quantity = self.facility.storage.take_available(product_id, demand)
    self.economy.total_units_sold += sold_quantity
    return BalanceSheet(sold_quantity * self.economy.unit_price, 0)
