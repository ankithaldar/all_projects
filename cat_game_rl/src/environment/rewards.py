#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Define Reward Mechanisms'''


# imports
from dataclasses import dataclass
#    script imports
from item_facilities import ItemFacility
# imports


# constants
# constants


# classes
@dataclass
class RewardBalanceSheet:
  '''Balance Sheet For Rewards'''

  time: int = 0 # positive
  cost: int = 0 # negative

  def total(self) -> int:
    return self.time - self.cost

  def __add__(self, other):
    return RewardBalanceSheet(self.time + other.time, self.cost + other.cost)

  def __sub__(self, other):
    return RewardBalanceSheet(self.time - other.time, self.cost - other.cost)

  def __repr__(self):
    return f'{self.time-self.cost} ({self.time} {self.cost})'

  def __radd__(self, other):
    return self if other == 0 else self.__add__(other)


@dataclass
class RewardCalculation:
  '''Calculate Rewards'''
  item: ItemFacility
  time: int = 0 # positive
  cost: int = 0 # negative

  def calculate_reward(self) -> RewardBalanceSheet:
    pass

  def item_idle_reward(self) -> RewardBalanceSheet:
    if not self.item.is_crafting:
      # target counts finished
      # waiting for previous items
      # waiting for coins
      pass
    pass

  def start_crafting_reward(self) -> RewardBalanceSheet:
    return RewardBalanceSheet(
      # time advantage for time saved
      time = ((self.item.manufacturing.batch_size - 1) * self.item.bom.req_time) / self.item.clock.run_time,
      # coin penalty for extra coins used
      cost = (self.item.bom.get_batch_cost(self.item.manufacturing.batch_size) - self.item.bom.init_cost) / self.item.bom.get_actual_batch_cost(self.item.manufacturing.batch_size - 1)
    )



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
