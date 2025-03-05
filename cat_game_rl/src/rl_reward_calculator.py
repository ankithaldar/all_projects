#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Calculate Rewards'''


# imports
#    script imports
# imports


# constants
TIME_ALPHA = 0.8
# constants


# classes
class RewardCalculator:
  '''Calculate Rewards'''

  def __init__(self, world):
    self.world = world

  def calculate_reward(self):
    self.reward = 0


  def level_completion_reward(self):
    if self.world.check_terminate_condition():
      self.reward += 10000

  def calculate_time_reward(self):
    for item, facility in self.world.item_facilities.items():
      # crafting time reward
      self.reward += facility.crafted_time / self.world.clock.time

      # idle time penalty
      self.reward -= facility.idle_time / self.world.clock.run_time

      # time saved with large batch size crafting
      if facility.is_crafting and self.world.clock.time == facility.manufacturing.start_time:
        self.reward +=

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
