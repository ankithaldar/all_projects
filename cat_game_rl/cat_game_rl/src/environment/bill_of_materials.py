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
  '''Data class representing a Bill of Materials (BOM) for a product.

    This class captures the materials required, initial cost, and lead time
    for producing a lot of the product. It also provides methods to calculate
    material requirements and cost for a specific batch size.

    Attributes:
      inputs: A `collections.Counter` object mapping product IDs to the
      quantity of each required per lot.
      init_cost: The initial cost per unit to produce the product (excluding
      volume discounts).
      req_time: The lead time required to produce a lot of the product.
      # output_lot_size: (Optional) The default output lot size (defaults to 1).

    Methods:
      get_batch_input(self, batch_size): Calculates the material requirements
      (quantities) for a given batch size.
      get_batch_cost(self, batch_size): Calculates the estimated cost to produce
      a batch of the product, considering a 25% volume discount for larger
      batches (excluding initial setup costs).
      get_actual_batch_cost(self, batch_size): Calculates the total cost to
      produce a batch of the product, considering only the initial cost per
      unit and batch size (without volume discounts).
  '''

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
