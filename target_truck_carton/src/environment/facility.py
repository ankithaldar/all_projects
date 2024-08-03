#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Doc String for the module'''


# imports
from dataclasses import dataclass
from abc import ABC
#    script imports
# imports


# constants
# constants


# classes
@dataclass
class Facility(ABC):
  '''docstring for Facility'''

  def __init__(self, x, y, world):
    self.x = x
    self.y = y
    self.id_num = world.generate_id()
    self.id = f'{self.__class__.__name__}_{self.id_num}'
    self.world = world


class Store(Facility):
  '''docstring for Store'''

  def __init__(self, x, y, world):
    super(Store, self).__init__(x, y, world)


class Warehouse(Facility):
  '''docstring for Warehouse'''

  def __init__(self, x, y, world):
    super(Warehouse, self).__init__(x, y, world)


# classes


# functions
def function_name():
  pass
# functions


# main
def main():
  pass


# if main script
if __name__ == '__main__':
  main()
