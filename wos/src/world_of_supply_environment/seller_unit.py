#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Seller Unit
'''

# imports
from dataclasses import dataclass

from .agent import Agent
from .balance_sheet import BalanceSheet

# imports


class SellerUnit(Agent):
  '''
  Seller Unit
  '''

  @dataclass
  class Economy:
    '''
    Economy of seller Unit
    '''
    price_demand_intercept: int
    price_demand_slope: int
    unit_price: int = 0
    total_units_sold: int = 0

    def market_demand(self, unit_price):
      return max(
        0,
        self.price_demand_intercept - self.price_demand_slope * unit_price
      )

    def profit(self, units_sold, unit_price):
      return units_sold * unit_price

    def step_balance_sheet(self, units_sold, unit_price):
      return BalanceSheet(self.profit(units_sold, unit_price), 0)

  @dataclass
  class Config:
    price_demand_intercept: float
    price_demand_slope: float

  @dataclass
  class Control:
    unit_price: int

  def __init__(self, facility, economy):
    self.facility = facility
    self.economy = economy

  def act(self, control):
    if control is not None:
      # update the current unit price
      self.economy.unit_price = control.unit_price

    product_id = self.facility.bom.output_product_id
    demand = self.economy.market_demand(self.economy.unit_price)
    sold_qty = self.facility.storage.take_available(product_id, demand)
    self.economy.total_units_sold += sold_qty
    return self.economy.step_balance_sheet(sold_qty, self.economy.unit_price)
