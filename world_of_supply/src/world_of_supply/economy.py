#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Financial primitives of the World of Supply economy.

The whole simulation does its accounting with :class:`BalanceSheet` objects.
Convention: ``profit`` is non-negative, ``loss`` is non-positive, and
``total() == profit + loss`` is the net effect on a facility balance.
'''

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BalanceSheet:
  '''A single profit/loss entry produced by one agent during one time step.

  Attributes:
    profit: Money earned during the step (non-negative).
    loss: Money spent during the step (non-positive).
  '''

  profit: int = 0
  loss: int = 0

  def total(self) -> int:
    '''Compute the net monetary effect of this sheet.

    Returns:
      int: ``profit + loss``.
    '''
    return self.profit + self.loss

  def __add__(self, other: 'BalanceSheet') -> 'BalanceSheet':
    '''Add two sheets component-wise.'''
    return BalanceSheet(self.profit + other.profit, self.loss + other.loss)

  def __sub__(self, other: 'BalanceSheet') -> 'BalanceSheet':
    '''Subtract another sheet from this one component-wise.'''
    return BalanceSheet(self.profit - other.profit, self.loss - other.loss)

  def __radd__(self, other):
    '''Support :func:`sum` over collections of sheets.

    Args:
      other: Either an int (``0`` from :func:`sum`) or another sheet.
    '''
    if other == 0:
      return self
    return self.__add__(other)

  def __repr__(self) -> str:
    '''Render as ``net (profit loss)`` for compact logs.'''
    return f'{self.total()} ({self.profit} {self.loss})'
