#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Value Add Facility Cell
'''

from .consumer_unit import ConsumerUnit
from .create_distribution_unit import create_distribution_unit
# imports
from .facility_cell import FacilityCell
from .manufacturing_unit import ManufacturingUnit
from .storage_unit import StorageUnit

# imports


class ValueAddFactoryCell(FacilityCell):
  '''
  Value Add Facility Cell
  '''

  def __init__(self, x, y, world, config, economy_config):
    super(ValueAddFactoryCell, self).__init__(
      x,
      y,
      world,
      config,
      economy_config
    )
    self.storage = StorageUnit(
      config.max_storage_capacity,
      StorageUnit.Economy(config.unit_storage_cost)
    )
    self.consumer = ConsumerUnit(self, config.sources)
    self.manufacturing = ManufacturingUnit(
      self,
      ManufacturingUnit.Economy(config.unit_manufacturing_cost)
    )
    self.distribution = create_distribution_unit(self, config)
