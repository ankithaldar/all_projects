#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Lumber Factory Cell
'''

# imports
from .raw_materials_factory_cell import RawMaterialsFactoryCell

# imports


class LumberFactoryCell(RawMaterialsFactoryCell):
  '''
  Lumber Factory Cell
  '''

  def __init__(self, x, y, world, config, economy_config):
    super(LumberFactoryCell, self).__init__(x, y, world, config, economy_config)
