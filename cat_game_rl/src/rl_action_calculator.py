#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Action Represenatation for RL environment'''


# imports
import torch
#    script imports
# imports


# constants
MAX_BATCH_SIM = 20
# constants


# classes
class ActionCalculator:
  '''Action Represenatation for RL environment'''

  def __init__(self, env):
    self.env = env

  def get_action(self):
     item_list_one_hot = torch.nn.functional.one_hot(
      [i for i, _ in enumerate(self.env.facilities.keys())],
      num_classes=-1
    )
    batch_size = torch.nn.functional.one_hot(
      [i for i in range(MAX_BATCH_SIM + 1)],
      num_classes=-1
    )

  def constraint_check(self):
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
