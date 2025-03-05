#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Cat Game Environment'''


# imports
from dataclasses import dataclass
import numpy as np
#    script imports
from env_game_world import worldbuilder_create
from env_state_calculator import StateCalculator
from env_action_calculator import ActionCalculator
from env_reward_calculator import RewardCalculator
from env_agent import Agent
# imports


# constants
EPSILON = 1
EPSILON_DECAY = 0.995
# constants


# classes
@dataclass
class GameMemory:
  '''Game Memory'''

  time_step: int
  states: np.array
  actions: np.array
  rewards: np.array
  terminate: bool


class CatGameEnv:
  '''Cat Game Environment'''

  def __init__(self):
    self.world = worldbuilder_create()
    self.state_calculator = StateCalculator(self.world)
    self.action_calculator = ActionCalculator(self.world)
    self.reward_calculator = RewardCalculator(self.world)

    self.memory = []

    self.append_to_memory(
      state=self.state_calculator.world_to_state_numpy_ndarray(),
      action=None,
      reward=None,
      terminate=None,
    )

  def reset(self):
    self.world = worldbuilder_create()
    self.memory = []

    self.append_to_memory(
      state=self.state_calculator.world_to_state_numpy_ndarray(),
      action=None,
      reward=None,
      terminate=None,
    )

  def step(self, batch_size:dict):
    self.time_step = self.world.clock.time

    for item, _ in self.world.item_facilities.items():
      self.world.item_facilities[item].act(batch_size[item])

    self.world.world_actions()

    terminate = self.world.check_terminate_condition()

    state = self.state_calculator.world_to_state_numpy_ndarray()
    reward = self.reward_calculator.calculate_reward(batch_size)

    self.append_to_memory(
      state=state,
      action=batch_size,
      reward=reward,
      terminate=terminate,
    )

    return state, reward, terminate

  def append_to_memory(self, state, action, reward, terminate):
    self.memory.append(
      GameMemory(
        time_step=self.world.clock.time,
        states=state,
        actions=action,
        rewards=reward,
        terminate=terminate,
      )
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
