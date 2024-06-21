#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''item Facilities'''


# imports
from dataclasses import dataclass
#    script imports
from bill_of_materials import BillOfMaterials
from game_clock import GameClock
from game_economy import GameEconomy
from manufacturing_unit import ManufacturingUnit
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
  total_target_count: int # total number of pirces needed to fulfil
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

  # change manufacturing time base on
  def set_one_min_manufacturing_time(self):
    if self.is_crafting and self.manufacturing.end_time > self.clock.time + 1:
      self.manufacturing.end_time = self.clock.time + 1

  # RL Actions -----------------------------------------------------------------
  def act(self, batch_size:int=1):
    if self.total_target_count > self.total_crafted_count:
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
