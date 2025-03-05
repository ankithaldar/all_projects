#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Doc String for the module'''


# imports
import numpy as np
import torch
#    script imports
# imports


# constants
# constants


# classes
class StateCalculator:
  '''State Calculator'''

  def __init__(self, world):
    self.world = world


  def world_to_state(self):
    coins = self.world.economy.coins/10^8
    game_time = self.world.clock.time/self.world.clock.run_time
    crafting_slots = [1 if facility.is_crafting else 0 for _, facility in self.world.item_facilities.items()]
    stock_status = [facility.current_stash/facility.total_target_count for _, facility in self.world.item_facilities.items()]
    pending_status = [(facility.total_target_count - facility.total_crafted_count)/facility.total_target_count for _, facility in self.world.item_facilities.items()]

    return [coins], [game_time], crafting_slots, stock_status, pending_status


  def world_to_state_numpy_ndarray(self):
    return np.array(
      object=self.world_to_state(),
      dtype=np.float32
    )


  def world_to_state_torch_tensor(self):
    return torch.tensor(
      data=self.world_to_state()
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
