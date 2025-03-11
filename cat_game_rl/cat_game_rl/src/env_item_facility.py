#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Item Class to maintain transaction and manufacturing of items'''


# imports
from dataclasses import dataclass
#    script imports
from env_bill_of_materials import BillOfMaterials
from env_game_clock import GameClock
from env_game_economy import GameEconomy
from env_manufacturing_unit import ManufacturingUnit
# imports


# constants
# constants


# classes
@dataclass
class ItemFacility:
  '''Item Class to maintain transaction and manufacturing of items'''

  name: str
  bom: BillOfMaterials
  target_count: int # final number that is needed for decor crafting
  total_target_count: int # total number of pieces needed to fulfil
  total_crafted_count: int
  sources: list
  game_economy: GameEconomy
  clock: GameClock

  def __post_init__(self):
    self.is_crafting = False
    self.current_stash = self.get_current_count_in_stash()
    self.manufacturing = ManufacturingUnit(self)

    # self.define_item_production_level()

  # def define_item_production_level(self):
  #   if len(self.sources) == 0:
  #     self.crafting_level = 1
  #   else:
  #     self.crafting_level = max([i.crafting_level for i in self.sources]) + 1

  def get_current_count_in_stash(self) -> int:
    return self.game_economy.items_in_stash[self.name]


    # RL Actions -----------------------------------------------------------------
  def step(self, batch_size:int=0):
    if batch_size != 0:
      self.manufacturing.act(batch_size)

    self.current_stash = self.get_current_count_in_stash()

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
