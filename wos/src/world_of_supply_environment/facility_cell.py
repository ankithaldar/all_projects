#!/usr/bin/env python
# -*- coding: utf-8 -*-


'''Base class for facility cell'''


# imports
from dataclasses import dataclass

from .agent import Agent
from .balance_sheet import BalanceSheet
from .bill_of_materials import BillOfMaterials
from .cell import Cell
from .consumer_unit import ConsumerUnit
from .distribution_unit import DistributionUnit
from .manufacturing_unit import ManufacturingUnit
from .seller_unit import SellerUnit
from .storage_unit import StorageUnit

# imports


class FacilityCell(Cell, Agent):
  '''Base class for facility cell'''

  @dataclass
  class Config(StorageUnit.Config,
         ConsumerUnit.Config,
         DistributionUnit.Config,
         ManufacturingUnit.Config,
         SellerUnit.Config):
    bill_of_materials: BillOfMaterials

  @dataclass
  class EconomyConfig:
    initial_balance: int

  @dataclass
  class Economy:
    total_balance: BalanceSheet

    def deposit(self, balance_sheets):
      total_balance_sheet = sum(balance_sheets)
      self.total_balance += total_balance_sheet
      return total_balance_sheet

  @dataclass
  class Control(ConsumerUnit.Control,
          DistributionUnit.Control,
          ManufacturingUnit.Control,
          SellerUnit.Control):
    pass

  def __init__(self, x, y, world, config, economy_config):
    super(FacilityCell, self).__init__(x, y)
    self.id_num = world.generate_id()
    self.id = f"{self.__class__.__name__}_{self.id_num}"
    self.world = world
    self.economy = FacilityCell.Economy(
      BalanceSheet(economy_config.initial_balance, 0)
    )
    self.bom = config.bill_of_materials
    self.storage = None
    self.consumer = None
    self.manufacturing = None
    self.distribution = None
    self.seller = None

  def act(self, control):
    units = filter(
      None,
      [
        self.storage,
        self.consumer,
        self.manufacturing,
        self.distribution,
        self.seller
      ]
    )
    balance_sheets = [ u.act(control) for u in  units ]
    return self.economy.deposit(balance_sheets)
