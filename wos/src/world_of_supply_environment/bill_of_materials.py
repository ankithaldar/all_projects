#!/usr/bin/env python
# -*- coding: utf-8 -*-


'''Bill of materials'''


# imports
from collections import Counter
from dataclasses import dataclass
# imports


@dataclass
class BillOfMaterials:
  '''Bill of materials'''

  # One manufacturing cycle consumes inputs
  # and produces output_lot_size units of output_product_id
  inputs: Counter  # (product_id -> quantity per lot)
  output_product_id: str
  output_lot_size: int = 1

  def input_units_per_lot(self):
    return sum(self.inputs.values())
