#!/usr/bin/env python
# -*- coding: utf-8 -*-


'''Basic Cell class.'''


# imports
from abc import ABC
# imports


class Cell(ABC):
  '''Basic Cell class.'''

  def __init__(self, x, y):
    self.x = x
    self.y = y

  def __repr__(self):
    return f'{self.__class__.__name__} ({self.x}, {self.y})'
