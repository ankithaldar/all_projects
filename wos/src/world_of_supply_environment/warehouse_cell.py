#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
WareHouse Cell
'''

from .consumer_unit import ConsumerUnit
from .create_distribution_unit import create_distribution_unit
# imports
from .facility_cell import FacilityCell
from .storage_unit import StorageUnit

# imports


class WarehouseCell(FacilityCell):
  '''
  WareHouse Cell
  '''

  def __init__(self, x, y, world, config, economy_config):
    super(WarehouseCell, self).__init__(x, y, world, config, economy_config)
    self.storage = StorageUnit(
      config.max_storage_capacity,
      StorageUnit.Economy(config.unit_storage_cost)
    )
    self.consumer = ConsumerUnit(self, config.sources)
    self.distribution = create_distribution_unit(self, config)
