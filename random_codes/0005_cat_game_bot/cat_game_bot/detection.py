#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Ordered screen-state detection from template matches and brightness."""

import time

from cat_game_bot.models import Detection, GameState, Match, Screen

__all__ = (
  'ScreenDetector', 'is_loading', 'BANNER_MARKERS', 'GIFT_DIALOG_MARKERS',
  'POPUP_MARKERS', 'DIALOG_MARKERS', 'CLOSE_DIALOG', 'LOADING_TEXT',
)

CLOSE_DIALOG = 'close_dialog'
GIFT_COLLECT = 'gift_collect'
# Popup banners are distinctive modal content. Generic close buttons are NOT
# state markers: close_dialog_2..5 match the craft-screen close X at 0.978 and
# would make DIALOG outrank CRAFTING. They stay navigator close-action targets.
BANNER_FLASH_SALE = 'banner_flash_sale'
BANNER_COINS = 'banner_coins'
BANNER_VIP = 'banner_vip'
BANNER_CLUB_TOKEN = 'banner_club_token'
BANNER_HARVEST = 'banner_harvest'
# The FRESH FRUIT SPECIAL OFFER popup carries this distinctive ribbon, so it
# joins the banner set (close_dialog_2..5 stay navigator-only close targets).
BANNER_FRESH_FRUIT = 'banner_fresh_fruit'
BANNER_MARKERS = (BANNER_FLASH_SALE, BANNER_COINS, BANNER_VIP,
                  BANNER_CLUB_TOKEN, BANNER_HARVEST, BANNER_FRESH_FRUIT)
# The real COINS! gift popup carries banner_coins alongside its Collect
# control. GIFT_DIALOG requires BOTH on the same screen so a bare COINS
# promo without a Collect button stays a normal POPUP.
GIFT_DIALOG_MARKERS = (BANNER_COINS, GIFT_COLLECT)
POPUP_MARKERS = BANNER_MARKERS + (CLOSE_DIALOG,)
DIALOG_MARKERS = BANNER_MARKERS
TUTORIAL_SKIP = 'tutorial_skip'
INVENTORY_MARKER = 'inventory_marker'
RECIPE_MARKER = 'recipe_marker'
TAP_TO_START = 'tap_to_start'
CRAFTING_MARKER = 'crafting_marker'
CATALOG_MARKER = 'catalog_marker'
HOME_MARKER = 'home_marker'
LOADING_TEXT = 'loading_text'

LOADING_BRIGHTNESS = 40.0

STATE_RULES = (
    (GameState.GIFT_DIALOG, GIFT_DIALOG_MARKERS),
    (GameState.POPUP, POPUP_MARKERS),
    (GameState.DIALOG, DIALOG_MARKERS),
    (GameState.TUTORIAL, (TUTORIAL_SKIP,)),
    (GameState.BAG_DIALOG, (INVENTORY_MARKER,)),
    (GameState.RECIPE_DIALOG, (RECIPE_MARKER,)),
    (GameState.TAP_TO_START, (TAP_TO_START,)),
    (GameState.CRAFTING, (CRAFTING_MARKER,)),
    (GameState.CATALOG, (CATALOG_MARKER,)),
    (GameState.HOME, (HOME_MARKER,)),
)


def is_loading(screen: Screen) -> bool:
  """Detect the near-black LOADING transition screen by brightness.

  Args:
    screen: The current screen.

  Returns:
    True when the mean pixel brightness is below the loading threshold.
  """
  pixels = screen.pixels
  if pixels.size == 0:
    return False
  return float(pixels.mean()) < LOADING_BRIGHTNESS


class ScreenDetector:
  """Classifies screens by the first matching ordered state rule.

  Attributes:
    matcher: Template matcher used to probe for state markers.
    now: Clock callable supplying detection timestamps.
  """

  STATE_RULES = STATE_RULES

  def __init__(self, matcher: object, now: object = time.time):
    self.matcher = matcher
    self.now = now

  def interactive_matches(self, screen: Screen) -> list[Match]:
    """Produce the interactive matches carried by a Detection.

    Args:
      screen: The current screen.

    Returns:
      All matched templates except the LOADING transition marker.
    """
    return [match for match in self.matcher.match_all(screen)
            if match.name != LOADING_TEXT]

  def detect(self, screen: Screen) -> Detection:
    """Return the first-match-wins state for the screen.

    Args:
      screen: The current screen.

    Returns:
      The recognised state with its interactive matches and timestamp.
    """
    matches = tuple(self.interactive_matches(screen))
    for state, names in self.STATE_RULES:
      if state is GameState.GIFT_DIALOG:
        # Conjunction: every marker must be present on the same screen.
        if all(self.matcher.match(screen, name) for name in names):
          return Detection(state, matches, self.now())
      elif any(self.matcher.match(screen, name) for name in names):
        return Detection(state, matches, self.now())
    if self.matcher.match(screen, LOADING_TEXT) or is_loading(screen):
      return Detection(GameState.LOADING, matches, self.now())
    return Detection(GameState.UNKNOWN, matches, self.now())
