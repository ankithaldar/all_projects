#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Navigation memory and action selection (plan v14 sections 7 and 9).

``Memory`` tracks 16-pixel-quantified interactive matches so the session can
detect repeated ticks.  ``Navigator`` turns a detection (plus an optional
planner action) into the single action taken on a tick.  A gift dialog always
yields exactly one Collect tap.  Popups and dialogs are dismissed by tapping a
close-dialog target (never by treating that target as a state marker), and a
Bag dialog is exited with a BACK action so a paused craft batch can resume.
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
_CLOSE_BAG = 'close_bag'


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
    are dismissed by tapping a close-dialog target (or BACK when none is
    visible), and BAG_DIALOG is closed with BACK.  Otherwise a supplied planner
    action wins; failing that, the first unvisited match is tapped (target
    items first), and finally a wait is emitted.

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
      return self._dismiss(detection, memory)
    if detection.state is GameState.BAG_DIALOG:
      return Action(ActionKind.BACK, None, None, _CLOSE_BAG)
    if planner_action is not None:
      return planner_action
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

  def _dismiss(self, detection: Detection, memory: Memory):
    """Return a real dismiss action for a POPUP or DIALOG detection.

    Close-dialog names are action targets only; the state has already been
    classified by the detector, so these names never drive state selection.

    Args:
        detection: Current POPUP or DIALOG detection.
        memory: Session memory of visited matches.

    Returns:
        Action: A close-target tap, a tap on a visible match, or BACK.
    """
    for name in _CLOSE_TARGETS:
      close = self.action_for_template(detection, name)
      if close is not None:
        return Action(ActionKind.TAP, close.point, None, _DISMISS)
    for match in detection.matches:
      if self._is_unvisited(match, memory):
        return Action(ActionKind.TAP, match.rect.center(), None, _DISMISS)
    return Action(ActionKind.BACK, None, None, _DISMISS)

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
