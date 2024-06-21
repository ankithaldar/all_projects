#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Bill Of Materials'''


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
  '''handle bill of materials'''

  inputs: Counter  # (product_id -> quantity per lot)
  init_cost: int
  req_time: int
  # output_lot_size: int = 1

  def get_batch_input(self, batch_size):
    return Counter({k: v * batch_size for k, v in self.inputs.items()})

  def get_batch_cost(self, batch_size):
    return self.init_cost * batch_size * (1 + 0.25 * (batch_size - 1))

  def get_actual_batch_cost(self, batch_size):
    return self.init_cost * batch_size
# classes
