#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Manages coins and item stash.'''


# imports
from collections import Counter
#    script imports
# imports


# constants
# constants


# classes
class GameEconomy:
  '''Manages coins and item stash.'''

  def __init__(self):
    self.coins = 0
    self.stash = Counter()
    self.used_coins = 0
    self.gained_coins = 0

  def update_coins(self, used_coins=0, gained_coins=0) -> None:
    self.coins += gained_coins - used_coins
    self.used_coins += used_coins
    self.gained_coins += gained_coins

  def update_stash(self, used_stash=None, gained_stash=None) -> None:
    if used_stash is not None:
      self.items_in_stash.subtract(used_stash)

    if gained_stash is not None:
      self.items_in_stash.update(gained_stash)



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
