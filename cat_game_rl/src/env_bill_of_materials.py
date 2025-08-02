#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Stores and calculates crafting recipes, costs, and times for each item'''


# imports
from collections import Counter
from dataclasses import dataclass
#    script imports
# imports


# constants
# constants


# classes
@dataclass
class BillOfMaterials:
  '''Stores and calculates crafting recipes, costs, and times for each item'''

  inputs: Counter
  init_cost: int
  req_time: int

  def calculate_batch_input(self, batch_size):
    return Counter({k: v * batch_size for k, v in self.inputs.items()})

  def calculate_batch_cost(self, batch_size):
    return self.init_cost * batch_size * (1 + 0.25 * (batch_size - 1))

  def calculate_actual_batch_cost(self, batch_size):
    return self.init_cost * batch_size

  def calculate_value_of_next_batch_count(self, n):
    return self.init_cost * (1 + 0.5 * (n - 1))
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
