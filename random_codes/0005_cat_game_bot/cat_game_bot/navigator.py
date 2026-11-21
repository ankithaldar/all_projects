#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Navigation memory and action selection (plan v14 sections 7 and 9).

``Memory`` tracks 16-pixel-quantified interactive matches so the session can
detect repeated ticks.  ``Navigator`` turns a detection (plus an optional
planner action) into the single action taken on a tick.  A gift dialog always
yields exactly one Collect tap.  Popups and dialogs are dismissed by tapping
the highest-scored close-dialog X inside the overlay-X band, so the
higher-scored craft-screen X beneath the modal is never tapped; BACK is the
fallback when no in-band close X is matched.  A Bag dialog is exited with
BACK so a paused craft batch can resume.
State priorities then favour the craft tab on HOME, an item tap or shelf
scroll in CRAFTING, the gold craft button in RECIPE_DIALOG, and dismissal in
CATALOG, before the generic unvisited-match fallback.
"""

from cat_game_bot.models import (
  Action, ActionKind, Detection, GameState, Match, Point,
)
from cat_game_bot.targets import TARGET_IDS, item_template

_TARGET_NAMES = frozenset(item_template(item_id) for item_id in TARGET_IDS)
_GIFT_COLLECT_POINT = Point(540, 1625)
_CLOSE_TARGETS = (
  'close_dialog', 'close_dialog_2', 'close_dialog_3',
  'close_dialog_4', 'close_dialog_5',
)
_DISMISS = 'dismiss'
# Observed overlay X centers sit at y ~1865-2082; the craft-screen X beneath a
# modal sits at y ~2310-2320.  This band separates them for every popup seen.
_OVERLAY_X_MAX_Y = 2150
_CLOSE_BAG = 'close_bag'
_CRAFT_SCROLL = 'craft_scroll'
_CRAFT_SCROLL_START = Point(540, 1900)
_CRAFT_SCROLL_END = Point(540, 900)
_ITEM_PREFIX = 'item_'


class Memory:
  """Tracks visited matches and a no-progress counter.

  Attributes:
      visited: Set of ``(name, x // 16, y // 16)`` keys already seen.
      no_progress: Number of consecutive ticks without a fresh match.
      last_state: State observed on the previous tick.
  """

  def __init__(self):
    """Initialise empty memory."""
    self.visited = set()
    self.no_progress = 0
    self.last_state = None

  def mark_visited(self, name, rect):
    """Record a quantified match and report whether it was new.

    Args:
        name: Template name.
        rect: Quantified ``(x, y)`` cell coordinates.

    Returns:
        bool: True when this key had not been visited before.
    """
    key = (name, rect[0], rect[1])
    if key in self.visited:
      return False
    self.visited.add(key)
    return True

  def reset_progress(self):
    """Clear only the no-progress counter, preserving visited keys."""
    self.no_progress = 0


class Navigator:
  """Chooses the single action for a tick from a detection."""

  def __init__(self, detector):
    """Store the detector used to produce the supplied detections.

    Args:
        detector: ``ScreenDetector`` instance (held for completeness).
    """
    self.detector = detector

  def action_for_template(self, detection: Detection, name: str):
    """Return a tap at the center of the named match, if present.

    Args:
        detection: Current screen detection.
        name: Template name to locate.

    Returns:
        Action | None: A tap action, or None when the name is absent.
    """
    for match in detection.matches:
      if match.name == name:
        return Action(ActionKind.TAP, match.rect.center(), None, name)
    return None

  def choose(self, detection: Detection, memory: Memory,
             planner_action: Action | None):
    """Select the action for this tick.

    A gift dialog always produces the lone Collect tap.  POPUP/DIALOG states
    tap the highest-scored in-band close X (BACK only when none is in band),
    and BAG_DIALOG is closed with BACK.  A supplied planner action then wins.
    Remaining states follow section-7 priorities: HOME taps the
    craft tab, CRAFTING taps an item or scrolls the shelf, RECIPE_DIALOG taps
    the craft button, CATALOG dismisses, TAP_TO_START taps its marker, and
    LOADING/UNKNOWN wait.  Finally the generic first-unvisited-match fallback
    (target items first) applies, ending in a wait.

    Args:
        detection: Current screen detection.
        memory: Session memory of visited matches.
        planner_action: Optional action chosen by the craft planner.

    Returns:
        Action: The action to execute this tick.
    """
    if detection.state is GameState.GIFT_DIALOG:
      collect = self.action_for_template(detection, 'gift_collect')
      if collect is not None:
        return collect
      return Action(ActionKind.TAP, _GIFT_COLLECT_POINT, None,
                    'gift_collect')
    if detection.state in (GameState.POPUP, GameState.DIALOG):
      return self._dismiss(detection)
    if detection.state is GameState.BAG_DIALOG:
      return Action(ActionKind.BACK, None, None, _CLOSE_BAG)
    if planner_action is not None:
      return planner_action
    state = detection.state
    if state is GameState.HOME:
      craft_tab = self.action_for_template(detection, 'craft_tab')
      if craft_tab is not None:
        return craft_tab
    elif state is GameState.CRAFTING:
      item = self._first_unvisited_item(detection, memory)
      if item is not None:
        return item
      return Action(ActionKind.SWIPE, _CRAFT_SCROLL_START,
                    _CRAFT_SCROLL_END, _CRAFT_SCROLL)
    elif state is GameState.RECIPE_DIALOG:
      craft = self.action_for_template(detection, 'craft_button')
      if craft is not None:
        return craft
      return self._dismiss(detection)
    elif state is GameState.CATALOG:
      return self._dismiss(detection)
    elif state is GameState.TAP_TO_START:
      start = self.action_for_template(detection, 'tap_to_start')
      if start is not None:
        return start
    elif state in (GameState.LOADING, GameState.UNKNOWN):
      return Action(ActionKind.WAIT, None, None, 'idle')
    candidates = [
        match for match in detection.matches
        if self._is_unvisited(match, memory)
    ]
    if candidates:
      candidates.sort(key=lambda match: match.name not in _TARGET_NAMES)
      chosen = candidates[0]
      return Action(ActionKind.TAP, chosen.rect.center(), None,
                    chosen.name)
    return Action(ActionKind.WAIT, None, None, 'idle')

  def _dismiss(self, detection: Detection):
    """Return a dismiss action that closes the focused overlay only.

    Close-dialog names are action targets only; the detector has already
    classified the state, so they never drive state selection.  All variants
    are the same red-X crop, so score cannot identify the overlay's X: on a
    real popup the craft-screen X beneath the modal scores higher.  Only
    close targets inside the overlay-X band are candidates, and the
    highest-scored of those is tapped.  With no in-band close X, BACK closes
    the focused overlay so the craft X is never tapped from a popup state.

    Args:
        detection: Current POPUP, DIALOG, RECIPE_DIALOG or CATALOG detection.

    Returns:
        Action: A tap on the highest-scored in-band close target, or BACK
            when no close X is in band.
    """
    in_band = [
        match for match in detection.matches
        if match.name in _CLOSE_TARGETS
        and match.rect.center().y <= _OVERLAY_X_MAX_Y
    ]
    if in_band:
      chosen = max(in_band, key=lambda match: match.score)
      return Action(ActionKind.TAP, chosen.rect.center(), None, _DISMISS)
    return Action(ActionKind.BACK, None, None, _DISMISS)

  def _first_unvisited_item(self, detection: Detection, memory: Memory):
    """Return a tap on the first unvisited item_* match, if any.

    Args:
        detection: Current CRAFTING detection.
        memory: Session memory of visited matches.

    Returns:
        Action | None: A tap on an item match, or None when absent.
    """
    for match in detection.matches:
      if (match.name.startswith(_ITEM_PREFIX)
              and self._is_unvisited(match, memory)):
        return Action(ActionKind.TAP, match.rect.center(), None,
                      match.name)
    return None

  @staticmethod
  def _is_unvisited(match: Match, memory: Memory):
    """Return whether a match has not yet been visited.

    Args:
        match: Detected interactive match.
        memory: Session memory.

    Returns:
        bool: True when the match key is new.
    """
    key = (match.name, match.rect.x // 16, match.rect.y // 16)
    return key not in memory.visited
