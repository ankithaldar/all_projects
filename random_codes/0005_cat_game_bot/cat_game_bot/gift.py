#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Timer-ready gift collection and coin-delta verification."""

from cat_game_bot.models import (
    Action,
    ActionKind,
    GiftResult,
    Rect,
    Screen,
)
from cat_game_bot.ocr import OcrReader, parse_count, parse_duration

__all__ = ('GiftCollector',)

MAX_COIN_DELTA = 100000


class GiftCollector:
  """Reads the gift timer and coins, and plans a single icon tap.

  Attributes:
    ocr: Reader used for the timer and coin regions.
    regions: Mapping with 'coins' and 'gift_timer' rectangles.
  """

  def __init__(self, ocr: OcrReader, regions: dict[str, Rect]):
    self.ocr = ocr
    self.regions = regions

  def is_ready(self, screen: Screen) -> bool:
    """Return True only when the timer is absent or already at 00:00.

    Unparseable text (a popup or banner covering the timer) must not be
    mistaken for an absent timer, because that would open a gift cycle on a
    screen whose coin counter is also covered.

    Args:
      screen: The current screen.

    Returns:
      True when the timer region is blank or shows an elapsed countdown.
    """
    text = self.ocr.read(screen, self.regions['gift_timer']).strip()
    if not text:
      return True
    duration = parse_duration(text)
    if duration is None:
      return False
    return duration == 0.0

  def plan(self, screen: Screen) -> Action | None:
    """Plan the single gift-icon tap when a gift is ready.

    Args:
      screen: The current screen.

    Returns:
      A TAP action aimed at the timer/icon centre, or None.
    """
    if not self.is_ready(screen):
      return None
    return Action(ActionKind.TAP, self.regions['gift_timer'].center(),
                  None, 'gift_icon')

  def read_coins(self, screen: Screen) -> int | None:
    """Read the coin balance from the configured region.

    Args:
      screen: The current screen.

    Returns:
      The parsed coin count, or None when unreadable.
    """
    return parse_count(self.ocr.read(screen, self.regions['coins']))

  def verify(self, before: int | None, after: int | None) -> GiftResult:
    """Verify that a coin delta is a plausible gift reward.

    Args:
      before: Coin balance before collection, if readable.
      after: Coin balance after collection, if readable.

    Returns:
      A GiftResult describing acceptance and the observed delta.
    """
    if before is None or after is None:
      return GiftResult(False, 0, 'coins unreadable')
    delta = after - before
    if delta <= 0:
      return GiftResult(False, delta, 'no positive coin delta')
    if delta > MAX_COIN_DELTA:
      return GiftResult(False, delta, 'implausible coin delta')
    return GiftResult(True, delta, 'ok')
