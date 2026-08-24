#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Grid geography: terrain, railroads, and the cell base class.'''

from __future__ import annotations

from abc import ABC


class Cell(ABC):
  '''Base class for anything occupying a grid coordinate.

  Attributes:
    x: Horizontal grid position.
    y: Vertical grid position.
  '''

  def __init__(self, x: int, y: int) -> None:
    '''Place the cell on the grid.

    Args:
      x: Horizontal position.
      y: Vertical position.
    '''
    self.x = x
    self.y = y

  def __repr__(self) -> str:
    '''Render as ``ClassName (x, y)``.'''
    return f'{self.__class__.__name__} ({self.x}, {self.y})'


class TerrainCell(Cell):
  '''Impassable filler making up the default background of the map.'''


class RailroadCell(Cell):
  '''Traversable road segment used by transport vehicles.'''
