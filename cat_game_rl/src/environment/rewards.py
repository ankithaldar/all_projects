#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Define Reward Mechanisms'''


# imports
from dataclasses import dataclass
from collections import Counter
#    script imports
# imports


# constants
COST_PENALTY = 1
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
  item: object
  time: int = 0 # positive
  cost: int = 0 # negative

  def calculate_reward(self):
    self.item_idle_reward()
    # self.start_crafting_reward()


  def item_idle_reward(self):
    def get_source_stash():
      return Counter({i.name: i.get_current_count_in_stash()
        for i in self.item.sources
      })

    if not self.item.is_crafting:

      # target counts finished
      if self.item.total_target_count <= self.item.total_crafted_count:
        self.item.reward_memory.append(
          RewardBalanceSheet(
            time = 1,
            cost = 0
          )
        )

      # waiting for previous items
      if self.item.bom.get_batch_input(self.item.manufacturing.batch_size) > get_source_stash():
        for item_source in self.item.sources:
          item_source.reward_memory.append(
            RewardBalanceSheet(
              time = -1, # negative to minimize the wating time
              cost = 0
            )
          )

  def start_crafting_reward(self):
    return RewardBalanceSheet(
      # time advantage for time saved
      time = ((self.item.manufacturing.batch_size) * self.item.bom.req_time) / self.item.clock.run_time,
      # coin penalty for extra coins used
      cost = self._safe_divide(
        (self.item.bom.get_batch_cost(self.item.manufacturing.batch_size) - self.item.bom.init_cost),
        self.item.bom.get_actual_batch_cost(self.item.manufacturing.batch_size - 1)
      ) * COST_PENALTY
    )

  @staticmethod
  def _safe_divide(x, y):
    if y != 0:
      return x/y
    return 0



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
