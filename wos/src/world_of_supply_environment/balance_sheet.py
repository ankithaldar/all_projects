#!/usr/bin/env python
# -*- coding: utf-8 -*-


'''Balance Sheet'''


# imports
from dataclasses import dataclass
# imports


@dataclass
class BalanceSheet:
  '''Balance Sheet'''

  profit: int = 0
  loss: int = 0

  def total(self) -> int:
    return self.profit + self.loss

  def __add__(self, other):
    return BalanceSheet(self.profit + other.profit, self.loss + other.loss)

  def __sub__(self, other):
    return BalanceSheet(self.profit - other.profit, self.loss - other.loss)

  def __repr__(self):
    return f'{self.profit+self.loss} ({self.profit} {self.loss})'

  def __radd__(self, other):
    if other == 0:
      return self
    else:
      return self.__add__(other)
