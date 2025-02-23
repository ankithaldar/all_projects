#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''State Represenatation for RL environment'''


# imports
from dataclasses import dataclass
import numpy as np
#    script imports
from env_utils import RUN_TIME
# imports


# constants
# constants


# classes
@dataclass
class StateCalculator:
  '''State Represenatation for RL environment'''

  def __init__(self, env) -> None:
    self.env = env

  def __post__init__(self) -> None:
    self.coins = self.env.economy.coins/10^10
    self.game_time = self.env.clock.time/RUN_TIME
    self.one_min_crafting = self.env.clock.flag_one_min_crafting
    self.is_crafting = [1 if facility.is_crafting else 0 for _, facility in self.env.item_facilities.items()]
    self.stock_status = [facility.current_stash/facility.total_target_count for _, facility in self.env.item_facilities.items()]
    self.pending_status = [(facility.total_target_count - facility.total_crafted_count)/facility.total_target_count for _, facility in self.env.item_facilities.items()]

  def _to_numpy(self):
    return np.array(
      [self.coins] +
      [self.game_time] +
      [self.one_min_crafting] +
      self.is_crafting +
      self.stock_status +
      self.pending_status,
      dtype=np.float32
    )

  def _to_torch_tensor(self):
    '''Convert state to a PyTorch tensor.'''
    return torch.tensor(self._to_numpy(), dtype=torch.float32)

  def _to_tf_tensor(self):
    pass


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
