#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Doc String for the module'''


# imports
from __future__ import annotations
from typing import Tuple
from dataclasses import dataclass
#    script imports
import truck as t
# imports


# constants
# constants


# classes
@dataclass
class Carton:
  '''docstring for Carton'''

  def __init__(self, dimensions: Tuple[int], carton_id: int|str, weight: int):
    self.length = dimensions[0]
    self.width = dimensions[1]
    self.height = dimensions[2]
    self.carton_id = carton_id # can take in registration number
    self.weight = weight

    self.truck = None  # can take in truck object if loaded in truck
    self.dropped_off = False

  @property
  def volume(self):
    return self.length * self.width * self.height

  # def add_source_destination(self, destination: Store, source: Warehouse):
  #   self.warehouse = source
  #   self.store = destination

  def is_in_truck(self, truck_id: int|str):
    pass

  def load_in_truck(self, truck: t.Truck):
    self.truck = truck

  def unload_from_truck(self):
    self.truck = None


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
