#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Retailer Cell
'''

from .consumer_unit import ConsumerUnit
from .facility_cell import FacilityCell
from .seller_unit import SellerUnit
# imports
from .storage_unit import StorageUnit

# imports


class RetailerCell(FacilityCell):
  '''
  Retailer Cell
  '''

  def __init__(self, x, y, world, config, economy_config):
    super(RetailerCell, self).__init__(x, y, world, config, economy_config)
    self.storage = StorageUnit(
      config.max_storage_capacity,
      StorageUnit.Economy(config.unit_storage_cost)
    )
    self.consumer = ConsumerUnit(self, config.sources)
    self.seller = SellerUnit(
      self,
      SellerUnit.Economy(
        config.price_demand_intercept,
        config.price_demand_slope
      )
    )
