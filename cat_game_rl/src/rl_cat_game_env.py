#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Doc String for the module'''


# imports
#    script imports
from env_game_world import worldbuilder_create
from rl_state_calculator import StateCalculator
from rl_action_calculator import ActionCalculator
from rl_reward_calculator import RewardCalculator
# imports


# constants
# constants


# classes
class CatGameEnv:
  '''RL environment for Cat Game'''

  def __init__(self):
    self.world = worldbuilder_create()
    self.state_calculator = StateCalculator(self.world)
    self.reward_calculator = RewardCalculator(self.world)
    self.action_calculator = ActionCalculator(self.world)

  def reset(self):
    self.world = worldbuilder_create()


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
