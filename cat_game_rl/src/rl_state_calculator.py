#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Calculate States from given world environemnt'''


# imports
import torch
#    script imports
# imports


# constants
# constants


# classes
class StateCalculator:
  '''Calculate States from given world environemnt'''

  def __init__(self, world):
    self.world = world

  def world_to_state(self):
    state = {}
    state['current_time'] = self.world.clock.current_time
    state['remaining_time'] = self.world.clock.time_limit - self.world.clock.current_time
    state['coins'] = self.world.economy.coins
    for item, facility in self.world.item_facilities.items():
      state[f'is_crafting_{item}'] = int(facility.is_crafting)
      state[f'total_target_{item}'] = facility.total_target_count
      state[f'total_crafted_{item}'] = facility.total_crafted_count
      state[f'current_stash_{item}'] = facility.get_current_count_in_stash()
      state[f'completed_{item}'] = int(True if facility.total_target_count <= facility.total_crafted_count else False)
      state[f'batch_size_{item}'] = facility.manufacturing.batch_size

    state['level_complete'] = self.world.check_terminate_condition()

    return state

  def state_to_torch_tensor(self):
    return torch.tensor(list(self.world_to_state().values()), dtype=torch.float32)

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
