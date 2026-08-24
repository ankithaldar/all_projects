#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Railroad sprite selection for text renderers.

Picks a box-drawing character for each railroad cell based on which
neighbors are also railroads (e.g. ═ for horizontal runs, ╬ for crossings).
'''

from __future__ import annotations

from collections.abc import Callable

_SPRITE_TABLE = {
    frozenset({'top', 'bottom'}): '║',
    frozenset({'left', 'right'}): '═',
    frozenset({'top', 'right'}): '╚',
    frozenset({'top', 'left'}): '╝',
    frozenset({'bottom', 'right'}): '╔',
    frozenset({'bottom', 'left'}): '╗',
    frozenset({'top', 'bottom', 'left'}): '╣',
    frozenset({'top', 'bottom', 'right'}): '╠',
    frozenset({'top', 'left', 'right'}): '╩',
    frozenset({'bottom', 'left', 'right'}): '╦',
    frozenset({'top', 'bottom', 'left', 'right'}): '╬',
}


def railroad_glyph(x: int, y: int, is_railroad: Callable[[int, int], bool]) -> str:
  '''Select the box-drawing glyph for one railroad cell.

  Args:
    x: Cell x position.
    y: Cell y position.
    is_railroad: Predicate telling whether a neighbor cell is railroad.

  Returns:
    str: Box-drawing character; empty string for isolated cells.
  '''
  connections = {
      name
      for name, (dx, dy) in {
          'top': (0, -1),
          'bottom': (0, 1),
          'left': (-1, 0),
          'right': (1, 0),
      }.items()
      if is_railroad(x + dx, y + dy)
  }
  if not connections:
    return '═'
  return _SPRITE_TABLE.get(frozenset(connections), '═')
