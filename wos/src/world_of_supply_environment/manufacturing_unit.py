#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Manufacturing Unit
'''

# imports
from dataclasses import dataclass

from .agent import Agent
from .balance_sheet import BalanceSheet

# imports


class ManufacturingUnit(Agent):
  '''
  Manufacturing Unit
  '''

  @dataclass
  class Economy:
    unit_cost: int                   # production cost per unit

    def cost(self, units_produced):
      return -self.unit_cost * units_produced

    def step_balance_sheet(self, units_produced):
      return BalanceSheet(0, self.cost(units_produced))

  @dataclass
  class Config:
    unit_manufacturing_cost: int

  @dataclass
  class Control:
    production_rate: int                  # lots per time step

  def __init__(self, facility, economy):
    self.facility = facility
    self.economy = economy

  def act(self, control):
    units_produced = 0

    if control is not None:
      for _ in range(control.production_rate):
        # check we have enough storage space for the output lot
        if self.facility.storage.available_capacity() >= self.facility.bom.output_lot_size - self.facility.bom.input_units_per_lot():
          # check we have enough input materials
          if self.facility.storage.try_take_units(self.facility.bom.inputs):
            self.facility.storage.stock_levels[self.facility.bom.output_product_id] += self.facility.bom.output_lot_size
            units_produced += self.facility.bom.output_lot_size

    return self.economy.step_balance_sheet(units_produced)
