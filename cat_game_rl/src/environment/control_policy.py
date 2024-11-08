#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Control Policy for the game'''


# imports
import random
#    script imports
# imports


# constants
# constants


# classes
class SimpleControlPolicy:
  '''Policy control for the game. Choosing one batch bize per flow'''

  def compute_batch_size(self, world):
    '''computing batch size for each flow'''

    batch_size = self.random_batch_size(world=world)

    return batch_size

  def fixed_batch_size(self, world):
    return {
      'string': 10,
      'wood': 10,
      'metal': 12,
      'ribbon': 5,
      'needles': 3,
      'sparkles': 2,
      'bronze': 3,
      'silver': 2,
      'gold': 1,
      'amethyst': 3,
      'pendant': 1,
      'necklace': 1,
      'orb': 2,
      'water': 2,
      'fire': 1,
      'waterstone': 1,
      'firestone': 1,
      'elementstone': 1,
      'artifact': 1
    }

  def random_batch_size(self, world):
    '''Randomly choosing batch size for each flow'''

    batch_size = {}
    for k, v in world.item_facilities.items():
      size = min(20, v.total_target_count - v.total_crafted_count)
      size =  0 if size <= 0 else random.randint(0, size)
      batch_size[k] = size
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
