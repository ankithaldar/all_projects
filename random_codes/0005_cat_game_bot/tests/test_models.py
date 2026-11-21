#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for the cat_game_bot.models data containers and enums."""

import dataclasses
import numpy as np
import pytest

from cat_game_bot.models import (
  Action,
  ActionKind,
  ActionResult,
  CraftStep,
  Detection,
  GameState,
  GiftResult,
  InventoryItem,
  Match,
  Point,
  Rect,
  Screen,
  SessionStats,
)


def test_point_is_frozen_and_stores_coordinates():
  """Point stores its coordinates and rejects mutation."""
  point = Point(3, 7)
  assert (point.x, point.y) == (3, 7)
  with pytest.raises(dataclasses.FrozenInstanceError):
    point.x = 9


def test_rect_center_uses_floor_division():
  """Rect.center returns the midpoint using integer floor division."""
  assert Rect(0, 0, 10, 20).center() == Point(5, 10)
  assert Rect(1, 1, 4, 4).center() == Point(3, 3)


def test_rect_contains_is_half_open():
  """Rect.contains treats the rectangle as half-open [x, x+width)."""
  rect = Rect(10, 20, 5, 5)
  assert rect.contains(Point(10, 20))
  assert rect.contains(Point(14, 24))
  assert not rect.contains(Point(15, 24))
  assert not rect.contains(Point(14, 25))
  assert not rect.contains(Point(9, 20))


def test_rect_area_is_width_times_height():
  """Rect.area multiplies width by height."""
  assert Rect(0, 0, 3, 7).area() == 21


def test_screen_reports_pixel_dimensions():
  """Screen.width/height come from the pixel array shape."""
  screen = Screen(np.zeros((2400, 1080, 3), dtype=np.uint8))
  assert (screen.width, screen.height) == (1080, 2400)
  assert (screen.pixels.shape[1], screen.pixels.shape[0]) == (1080, 2400)


def test_screen_uses_identity_equality():
  """Screen is declared with eq=False so distinct screens compare unequal."""
  first = Screen(np.zeros((2, 2, 3), dtype=np.uint8))
  second = Screen(np.zeros((2, 2, 3), dtype=np.uint8))
  assert first != second
  assert first == first


def test_match_is_frozen():
  """Match is a frozen dataclass holding name, rect and score."""
  match = Match('icon', Rect(0, 0, 4, 4), 0.9)
  assert (match.name, match.rect, match.score) == (
    'icon', Rect(0, 0, 4, 4), 0.9)
  with pytest.raises(dataclasses.FrozenInstanceError):
    match.score = 0.1


def test_detection_is_frozen_and_holds_tuple_of_matches():
  """Detection is frozen and keeps its matches as a tuple."""
  matches = (Match('a', Rect(0, 0, 1, 1), 1.0),)
  detection = Detection(GameState.HOME, matches, 12.5)
  assert detection.state is GameState.HOME
  assert detection.matches == matches
  assert isinstance(detection.matches, tuple)
  assert detection.timestamp == 12.5
  with pytest.raises(dataclasses.FrozenInstanceError):
    detection.state = GameState.LOADING


def test_game_state_has_all_members_with_string_values():
  """GameState exposes all twelve states with stable string values."""
  expected = ('UNKNOWN', 'LOADING', 'TAP_TO_START', 'TUTORIAL', 'HOME',
              'CRAFTING', 'CATALOG', 'POPUP', 'RECIPE_DIALOG',
              'BAG_DIALOG', 'GIFT_DIALOG', 'DIALOG')
  assert tuple(member.name for member in GameState) == expected
  assert tuple(member.value for member in GameState) == expected


def test_action_kind_has_all_members_with_string_values():
  """ActionKind exposes TAP, SWIPE, WAIT, BACK and STOP."""
  expected = ('TAP', 'SWIPE', 'WAIT', 'BACK', 'STOP')
  assert tuple(member.name for member in ActionKind) == expected
  assert tuple(member.value for member in ActionKind) == expected


def test_action_and_action_result_are_frozen():
  """Action and ActionResult are frozen and store their fields."""
  action = Action(ActionKind.TAP, Point(1, 2), None, 'reason')
  assert action.kind is ActionKind.TAP
  assert action.point == Point(1, 2)
  assert action.end is None
  assert action.reason == 'reason'
  result = ActionResult(action, True, 'ok')
  assert result.action is action
  assert result.ok is True
  assert result.detail == 'ok'
  with pytest.raises(dataclasses.FrozenInstanceError):
    result.ok = False


def test_gift_result_and_inventory_item_are_mutable():
  """GiftResult and InventoryItem are plain mutable dataclasses."""
  gift = GiftResult(True, 11160, 'ok')
  gift.ok = False
  assert gift.ok is False
  assert (gift.delta, gift.reason) == (11160, 'ok')
  item = InventoryItem('silver', 2, Rect(0, 0, 4, 4))
  item.count = 5
  assert (item.name, item.count, item.rect) == (
    'silver', 5, Rect(0, 0, 4, 4))


def test_craft_step_is_frozen():
  """CraftStep is frozen and stores its five fields."""
  step = CraftStep('string', 'basic', 'craft_string', 5, 100)
  assert (step.item, step.section, step.template, step.batch,
          step.cost) == ('string', 'basic', 'craft_string', 5, 100)
  with pytest.raises(dataclasses.FrozenInstanceError):
    step.batch = 1


def test_session_stats_defaults_include_interrupted_result():
  """SessionStats defaults every counter and result to the spec values."""
  stats = SessionStats()
  assert (stats.actions, stats.screenshots, stats.errors,
          stats.gifts_collected) == (0, 0, 0, 0)
  assert stats.states_seen == {}
  assert (stats.started_at, stats.finished_at) == (0.0, 0.0)
  assert stats.result == 'INTERRUPTED'


def test_session_stats_states_seen_uses_default_factory():
  """Each SessionStats instance gets its own states_seen dictionary."""
  first = SessionStats()
  second = SessionStats()
  first.states_seen['HOME'] = 1
  assert second.states_seen == {}
