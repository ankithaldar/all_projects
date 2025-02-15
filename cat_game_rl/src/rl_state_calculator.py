#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''State Represenatation for RL environment'''


# imports
from dataclasses import dataclass
#    script imports
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
    self.state = {
      'game_time': self.env.clock.time,
      'coins': self.env.economy.coins,
      'one_min_crafting': self.env.clock.flag_one_min_crafting,
      'item_states':{
        item_name: {
          'crafted_count': facility.total_crafted_count,
          'target_count': facility.total_target_count,
          'crafting_status': facility.is_crafting,
          'stock_status': facility.current_stash,
          'source_stock': {
            source_name: source.get_current_count_in_stash()
            for source_name, source in facility.sources.items()
          },
          'target_stock': facility.total_target_count
        }
        for item_name, facility in self.env.item_facilities.items()
      }
    }

  def _to_numpy(self):
    pass

  def _to_tensor(self):
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
