#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Game Economy - Coins & item stash'''


# imports
from collections import Counter
#    script imports
# imports


# constants
# constants


# classes
class GameEconomy:
  '''Game Economy - Coins & item stash'''

  def __init__(self) -> None:
    self.coins = 0
    self.items_in_stash = Counter()

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
