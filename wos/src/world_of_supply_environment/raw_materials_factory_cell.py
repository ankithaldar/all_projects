#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Raw material Facility Cell'''


# imports
from .create_distribution_unit import create_distribution_unit
from .facility_cell import FacilityCell
from .manufacturing_unit import ManufacturingUnit
from .storage_unit import StorageUnit

# imports


class RawMaterialsFactoryCell(FacilityCell):
  '''Raw material Facility Cell'''

  def __init__(self, x, y, world, config, economy_config):
    super(RawMaterialsFactoryCell, self).__init__(
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
    self.manufacturing = ManufacturingUnit(
      self,
      ManufacturingUnit.Economy(config.unit_manufacturing_cost)
    )
    self.distribution = create_distribution_unit(self, config)
