#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Manufacturing units and bills of materials.

A manufacturing unit converts one lot of input products into
``output_lot_size`` units of its output product, consuming storage space and
booking production cost.
'''

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from world_of_supply.base import Agent
from world_of_supply.economy import BalanceSheet

if TYPE_CHECKING:
  from world_of_supply.facility import FacilityCell, FacilityControl


@dataclass
class BillOfMaterials:
  '''Recipe of a single manufacturing cycle.

  Attributes:
    inputs: Product id to quantity consumed per lot.
    output_product_id: Product produced by the recipe.
    output_lot_size: Number of output units produced per lot.
  '''

  inputs: Counter = field(default_factory=Counter)
  output_product_id: str = ''
  output_lot_size: int = 1

  def input_units_per_lot(self) -> int:
    '''Return the total number of input units consumed by one lot.

    Returns:
      int: Sum of all per-lot input quantities.
    '''
    return sum(self.inputs.values())


@dataclass
class ManufacturingEconomy:
  '''Cost parameters of a manufacturing unit.

  Attributes:
    unit_manufacturing_cost: Production cost per produced unit.
  '''

  unit_manufacturing_cost: int = 100

  def step_balance_sheet(self, units_produced: int) -> BalanceSheet:
    '''Compute the production cost booked for one time step.

    Args:
      units_produced: Number of units produced during the step.

    Returns:
      BalanceSheet: Loss proportional to the produced volume.
    '''
    return BalanceSheet(0, -self.unit_manufacturing_cost * units_produced)


class ManufacturingUnit(Agent):
  '''Converts BOM inputs into outputs inside the owning facility.

  Attributes:
    facility: Owning facility providing storage and BOM.
    economy: Cost parameters.
  '''

  def __init__(self, facility: 'FacilityCell', economy: ManufacturingEconomy) -> None:
    '''Attach the unit to a facility.

    Args:
      facility: Owning facility.
      economy: Cost parameters.
    '''
    self.facility = facility
    self.economy = economy

  def act(self, control: 'FacilityControl | None') -> BalanceSheet:
    '''Run up to ``control.production_rate`` production lots.

    A lot only executes when the storage can absorb the net output volume
    (output lot minus consumed inputs) and all BOM inputs are in stock.

    Args:
      control: Carries ``production_rate``; ``None`` means zero lots.

    Returns:
      BalanceSheet: Production cost of everything actually produced.
    '''
    production_rate = control.production_rate if control is not None else 0
    bom = self.facility.bom
    storage = self.facility.storage
    units_produced = 0
    for _ in range(production_rate or 0):
      space_needed = bom.output_lot_size - bom.input_units_per_lot()
      if storage.available_capacity() < space_needed:
        continue
      if not storage.try_take_units(dict(bom.inputs)):
        continue
      storage.stock_levels[bom.output_product_id] = (
          storage.stock_levels.get(bom.output_product_id, 0) + bom.output_lot_size
      )
      units_produced += bom.output_lot_size
    return self.economy.step_balance_sheet(units_produced)
