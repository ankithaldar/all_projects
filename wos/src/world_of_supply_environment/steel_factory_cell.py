#!/usr/bin/env python
# -*- coding: utf-8 -*-


'''Steel Factory Cell'''


# imports
#   script imports
from .raw_materials_factory_cell import RawMaterialsFactoryCell
# imports


class SteelFactoryCell(RawMaterialsFactoryCell):
  '''Steel Factory Cell'''

  def __init__(self, x, y, world, config, economy_config):
    super(SteelFactoryCell, self).__init__(x, y, world, config, economy_config)
