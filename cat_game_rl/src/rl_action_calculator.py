#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Action Calculator'''


# imports
import numpy as np
#    script imports
# imports


# constants
# constants


# classes
class ActionCalculator:
  '''Calculate actions'''

  def __init__(self, world, max_batch_size):
    self.world = world
    self.max_batch_size = max_batch_size + 1 # added 1 for no action
    self.action_size = self.max_batch_size * len(self.world.item_facilities.keys())

  def action_to_dict(self, action):
    batch_size = {}
    for i in range(len(self.world.item_facilities.keys())):
      array_start = self.max_batch_size * i
      array_end = (self.max_batch_size - 1) + self.max_batch_size * i
      batch_size[list(self.world.item_facilities.keys())[i]] = np.argmax(action[array_start:array_end]).item()

    return batch_size
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
