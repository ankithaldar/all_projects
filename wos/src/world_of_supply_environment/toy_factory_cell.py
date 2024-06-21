#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Toy Factory Cell
'''

# imports
from .value_add_factory_cell import ValueAddFactoryCell

# imports


class ToyFactoryCell(ValueAddFactoryCell):
  '''
  Toy Factory Cell
  '''

  def __init__(self, x, y, world, config, economy_config):
    super(ToyFactoryCell, self).__init__(x, y, world, config, economy_config)
