#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Bag inventory counting via item icons and adjacent count OCR."""

from cat_game_bot.config import CANONICAL_IDS
from cat_game_bot.models import Match, Rect, Screen
from cat_game_bot.ocr import OcrReader, parse_count
from cat_game_bot.targets import item_template

__all__ = ('InventoryReader', 'CANONICAL_IDS')

_OFFSET_PREFIX = 'item_count.'


class InventoryReader:
  """Reads all 23 canonical Bag counts from icon matches and OCR.

  Attributes:
    ocr: Reader used for the count text next to each icon.
    matcher: Template matcher used to locate item icons.
    offsets: Mapping of 'item_count.<id>' to (dx, dy, w, h).
    library: Template library exposing available template names.
  """

  def __init__(self, ocr: OcrReader, matcher: object,
               offsets: dict[str, tuple[int, int, int, int]],
               library: object):
    self.ocr = ocr
    self.matcher = matcher
    self.offsets = offsets
    self.library = library
    self._names = frozenset(library.names())

  def _icon_name(self, item_id: str) -> str | None:
    """Return the canonical template name for an item icon, if present.

    Args:
      item_id: Canonical item id.

    Returns:
      The single canonical name when the library provides it, else None.
    """
    name = item_template(item_id)
    return name if name in self._names else None

  @staticmethod
  def _fits(rect: Rect, screen: Screen) -> bool:
    """Return True when a count rectangle lies within the screen."""
    return (rect.x >= 0 and rect.y >= 0
            and rect.x + rect.width <= screen.width
            and rect.y + rect.height <= screen.height)

  def read(self, screen: Screen) -> dict[str, int]:
    """Return the observed count of every canonical item.

    Args:
      screen: The current Bag screen.

    Returns:
      A mapping of all 23 canonical ids to their observed counts.
    """
    counts: dict[str, int] = {}
    for item_id in CANONICAL_IDS:
      name = self._icon_name(item_id)
      if name is None:
        counts[item_id] = 0
        continue
      matches: list[Match] = self.matcher.match(screen, name)
      if not matches:
        counts[item_id] = 0
        continue
      best = max(matches, key=lambda match: match.score)
      dx, dy, width, height = self.offsets[_OFFSET_PREFIX + item_id]
      rect = Rect(best.rect.x + dx, best.rect.y + dy, width, height)
      if not self._fits(rect, screen):
        counts[item_id] = 0
        continue
      count = parse_count(self.ocr.read(screen, rect))
      counts[item_id] = count if count is not None else 0
    return counts
