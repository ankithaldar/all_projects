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
  """Class representing the in-game economy system for a game.

    This class tracks the player's coins and item stash, including methods to
    update their balances and provides separate counters for tracking total
    coins used and gained throughout the game.

    Attributes:
      coins: The current number of coins the player possesses.
      items_in_stash: A `collections.Counter` object that tracks the quantities
      of each item type the player has stored.
      used_coins: Internal counter for tracking the total number of coins spent
      throughout the game (for informational purposes).
      gained_coins: Internal counter for tracking the total number of coins gained
      throughout the game (for informational purposes).

    Methods:
      update_coins(self, used_coins=0, gained_coins=0) -> None: Updates the
      player's coin balance, considering both used and gained coins in a
      single transaction.
      update_stash(self, used_stash=None, gained_stash=None) -> None: Updates
      the items in the player's stash. Takes separate arguments for items used
      (removed from stash) and items gained (added to stash), allowing for
      transactions involving both.
  """

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
