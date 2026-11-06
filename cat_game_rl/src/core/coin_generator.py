#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Coin generation and spending for the crafting game.'''

from __future__ import annotations


class CoinGenerator:
  '''Generates and tracks coins over time.

  Attributes:
    COINS_PER_TICK: Coins added on each tick.
  '''

  COINS_PER_TICK = 210

  def __init__(self, initial_balance: int = 0):
    '''Initializes the generator.

    Args:
      initial_balance: Starting coin balance.

    Raises:
      ValueError: If initial_balance is negative.
    '''
    if initial_balance < 0:
      raise ValueError(
        f'Cannot set negative initial_balance: {initial_balance}'
      )
    self._balance = initial_balance

  @property
  def balance(self) -> int:
    '''Returns the current coin balance.'''
    return self._balance

  def tick(self) -> None:
    '''Adds coins for one tick.'''
    self._balance += self.COINS_PER_TICK

  def spend(self, amount: int) -> bool:
    '''Attempts to spend coins.

    Args:
      amount: Coins to spend.

    Returns:
      True if the balance covered the amount, else False.

    Raises:
      ValueError: If amount is negative.
    '''
    if amount < 0:
      raise ValueError(f'Cannot spend negative amount: {amount}')
    if self._balance < amount:
      return False
    self._balance -= amount
    return True

  def add(self, amount: int) -> None:
    '''Adds coins to the balance.

    Args:
      amount: Coins to add.

    Raises:
      ValueError: If amount is negative.
    '''
    if amount < 0:
      raise ValueError(f'Cannot add negative amount: {amount}')
    self._balance += amount

  def reset(self, initial: int = 0) -> None:
    '''Resets the balance.

    Args:
      initial: New starting balance.

    Raises:
      ValueError: If initial is negative.
    '''
    if initial < 0:
      raise ValueError(f'Cannot reset to negative initial_balance: {initial}')
    self._balance = initial
