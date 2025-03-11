#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Doc String for the module'''


# imports
#    script imports
# imports


# constants
LEVEL_END_REWARD = 100
# constants


# classes
class RewardCalculator:
  '''Calclate rewards'''

  def __init__(self, world):
    self.world = world

  def calculate_total_rewards(self, prev_state, current_state):
    self.prev_state = prev_state
    self.current_state = current_state

    step_reward = 0
    step_reward += self.calculate_level_end_reward()
    step_reward += self.calculate_negative_coin_change()
    step_reward += self.calculate_time_over_target_not_complete_reward()
    step_reward += self.calculate_inventory_holding_reward()
    step_reward += self.calculate_crafting_start_reward()
    step_reward += self.calculate_idle_time_reward()
    step_reward += self.calculate_game_time_reward()


    return step_reward

  def calculate_level_end_reward(self):
    return LEVEL_END_REWARD * self.current_state['remaining_time'] if self.current_state['level_complete'] else 0

  def calculate_negative_coin_change(self):
    return -0.1 * (self.prev_state['coins'] - self.current_state['coins'])

  def calculate_time_over_target_not_complete_reward(self):
    return -LEVEL_END_REWARD if self.current_state['remaining_time'] == 0 and not self.current_state['level_complete'] else 0

  def calculate_inventory_holding_reward(self):
    return -0.1 * sum([
      self.current_state[f'current_stash_{item}'] for item in self.world.item_facilities.keys()
    ])

  def calculate_crafting_start_reward(self):
    return 10 * sum([
      1 if not self.prev_state[f'is_crafting_{item}'] and self.current_state[f'is_crafting_{item}'] else 0
      for item in self.world.item_facilities.keys()
    ])

  def calculate_idle_time_reward(self):
    return -10 * sum([
      1 if not self.prev_state[f'is_crafting_{item}'] and not self.current_state[f'is_crafting_{item}'] else 0
      for item in self.world.item_facilities.keys()
    ])

  def calculate_game_time_reward(self):
    return -1 * self.current_state['current_time']


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
