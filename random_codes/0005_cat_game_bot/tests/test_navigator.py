#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for navigation memory and action selection (plan v11 section 7)."""

from cat_game_bot.models import (
  Action, ActionKind, Detection, GameState, Match, Point, Rect,
)
from cat_game_bot.navigator import Memory, Navigator
from cat_game_bot.targets import TARGET_IDS, item_template


def _detection(state, matches=()):
  return Detection(state, tuple(matches), 0.0)


def test_mark_visited_uses_key_and_does_not_reset_progress():
  memory = Memory()
  memory.no_progress = 5
  assert memory.mark_visited('craft_tab', (3, 4)) is True
  assert memory.mark_visited('craft_tab', (3, 4)) is False
  assert ('craft_tab', 3, 4) in memory.visited
  assert memory.no_progress == 5


def test_reset_progress_clears_only_no_progress():
  memory = Memory()
  memory.no_progress = 7
  memory.last_state = GameState.HOME
  memory.mark_visited('craft_tab', (1, 2))
  memory.reset_progress()
  assert memory.no_progress == 0
  assert ('craft_tab', 1, 2) in memory.visited
  assert memory.last_state is GameState.HOME


def test_action_for_template_returns_center_tap_or_none():
  navigator = Navigator(object())
  match = Match('close_dialog', Rect(10, 20, 16, 16), 0.9)
  detection = _detection(GameState.DIALOG, [match])
  action = navigator.action_for_template(detection, 'close_dialog')
  assert action.kind is ActionKind.TAP
  assert action.point == Point(18, 28)
  assert action.reason == 'close_dialog'
  assert navigator.action_for_template(detection, 'missing') is None


def test_gift_dialog_choose_returns_single_collect_tap():
  navigator = Navigator(object())
  detection = _detection(GameState.GIFT_DIALOG)
  action = navigator.choose(detection, Memory(), None)
  assert action.kind is ActionKind.TAP
  assert action.reason == 'gift_collect'
  assert action.point is not None

  with_template = _detection(GameState.GIFT_DIALOG, [
      Match('gift_collect', Rect(532, 1617, 16, 16), 0.9)])
  action = navigator.choose(with_template, Memory(), None)
  assert action.kind is ActionKind.TAP
  assert action.reason == 'gift_collect'


def test_choose_prefers_supplied_planner_action():
  navigator = Navigator(object())
  planned = Action(ActionKind.TAP, Point(1, 2), None, 'planned')
  chosen = navigator.choose(_detection(GameState.CRAFTING), Memory(),
                            planned)
  assert chosen is planned


def test_choose_fallback_prefers_unvisited_target_match():
  navigator = Navigator(object())
  target = item_template(TARGET_IDS[0])
  matches = [
    Match('generic_icon', Rect(0, 0, 16, 16), 0.8),
    Match(target, Rect(32, 48, 16, 16), 0.9),
  ]
  chosen = navigator.choose(_detection(GameState.HOME, matches), Memory(),
                            None)
  assert chosen.reason == target
  assert chosen.kind is ActionKind.TAP


def test_choose_does_not_prioritise_bare_target_ids():
  """A bare item id is not a canonical target match name."""
  navigator = Navigator(object())
  matches = [
    Match('generic_icon', Rect(0, 0, 16, 16), 0.8),
    Match(TARGET_IDS[0], Rect(32, 48, 16, 16), 0.9),
  ]
  chosen = navigator.choose(_detection(GameState.HOME, matches), Memory(),
                            None)
  assert chosen.reason == 'generic_icon'


def test_action_for_template_with_real_manifest_item_name(manifest_names):
  """A canonical manifest item name yields a tap via action_for_template."""
  name = item_template(TARGET_IDS[0])
  assert name in manifest_names
  navigator = Navigator(object())
  match = Match(name, Rect(100, 200, 16, 16), 0.9)
  detection = _detection(GameState.CRAFTING, [match])
  action = navigator.action_for_template(detection, name)
  assert action is not None
  assert action.kind is ActionKind.TAP
  assert action.reason == name
  assert action.point == Point(108, 208)


def test_choose_returns_wait_when_nothing_available():
  navigator = Navigator(object())
  chosen = navigator.choose(_detection(GameState.LOADING), Memory(), None)
  assert chosen.kind is ActionKind.WAIT


# --- F21 / F22: close targets, bag exit, collect tap ------------------------

def test_choose_popup_taps_any_close_target():
  """POPUP taps the center of whichever close X variant is matched."""
  navigator = Navigator(object())
  for name in ('close_dialog', 'close_dialog_2', 'close_dialog_3',
               'close_dialog_4', 'close_dialog_5'):
    close = Match(name, Rect(1000, 100, 16, 16), 0.9)
    action = navigator.choose(_detection(GameState.POPUP, [close]),
                              Memory(), None)
    assert action.kind is ActionKind.TAP
    assert action.reason == 'dismiss'
    assert action.point == Point(1008, 108)
    assert action.end is None


def test_choose_popup_prefers_in_band_x_over_higher_scored_craft_x():
  """POPUP taps the in-band X even when the craft X scores higher."""
  navigator = Navigator(object())
  craft_x = Match('close_dialog_4', Rect(490, 2270, 100, 100), 0.9999)
  popup_x = Match('close_dialog_2', Rect(490, 1910, 100, 100), 0.9993)
  action = navigator.choose(
      _detection(GameState.POPUP, [craft_x, popup_x]), Memory(), None)
  assert action.kind is ActionKind.TAP
  assert action.reason == 'dismiss'
  assert action.point == Point(540, 1960)


def test_choose_dialog_prefers_in_band_x_over_higher_scored_craft_x():
  """DIALOG taps the in-band X even when the craft X scores higher."""
  navigator = Navigator(object())
  craft_x = Match('close_dialog_4', Rect(490, 2270, 100, 100), 0.9999)
  popup_x = Match('close_dialog_2', Rect(490, 1910, 100, 100), 0.9993)
  action = navigator.choose(
      _detection(GameState.DIALOG, [craft_x, popup_x]), Memory(), None)
  assert action.kind is ActionKind.TAP
  assert action.reason == 'dismiss'
  assert action.point == Point(540, 1960)


def test_choose_popup_out_of_band_close_falls_back_to_back():
  """POPUP with only the craft-screen X falls back to BACK, not that X."""
  navigator = Navigator(object())
  craft_x = Match('close_dialog_4', Rect(490, 2270, 100, 100), 0.9999)
  action = navigator.choose(
      _detection(GameState.POPUP, [craft_x]), Memory(), None)
  assert action.kind is ActionKind.BACK
  assert action.reason == 'dismiss'
  assert action.point is None


def test_choose_popup_without_close_match_still_dismisses_with_back():
  """POPUP without a close X still dismisses with BACK, not a banner tap."""
  navigator = Navigator(object())
  banner = Match('banner_flash_sale', Rect(0, 0, 16, 16), 0.9)
  action = navigator.choose(_detection(GameState.POPUP, [banner]),
                            Memory(), None)
  assert action.kind is ActionKind.BACK
  assert action.reason == 'dismiss'


def test_choose_popup_repeated_ticks_tap_close_each_time():
  """Repeated POPUP ticks keep producing one dismiss tap each."""
  navigator = Navigator(object())
  close = Match('close_dialog_2', Rect(490, 1910, 100, 100), 0.9993)
  detection = _detection(GameState.POPUP, [close])
  memory = Memory()
  for _ in range(3):
    action = navigator.choose(detection, memory, None)
    assert action.kind is ActionKind.TAP
    assert action.reason == 'dismiss'
    assert action.point == Point(540, 1960)


def test_choose_popup_with_no_matches_backs_out():
  navigator = Navigator(object())
  action = navigator.choose(_detection(GameState.POPUP), Memory(), None)
  assert action.kind is ActionKind.BACK
  assert action.reason == 'dismiss'


def test_choose_bag_dialog_returns_back():
  navigator = Navigator(object())
  action = navigator.choose(_detection(GameState.BAG_DIALOG), Memory(), None)
  assert action.kind is ActionKind.BACK
  assert action.reason == 'close_bag'


def test_action_for_template_exposes_craft_ready_collect_tap():
  navigator = Navigator(object())
  ready = Match('craft_ready_check', Rect(300, 400, 16, 16), 0.9)
  detection = _detection(GameState.CRAFTING, [ready])
  action = navigator.action_for_template(detection, 'craft_ready_check')
  assert action.kind is ActionKind.TAP
  assert action.reason == 'craft_ready_check'
  assert action.point == Point(308, 408)


# --- Plan section 7: state-driven action priorities -------------------------

def test_choose_home_prefers_craft_tab_over_other_matches():
  """HOME taps the craft tab even when a close X is matched first."""
  navigator = Navigator(object())
  matches = [
    Match('close_dialog_2', Rect(544, 1983, 16, 16), 0.9),
    Match('craft_tab', Rect(600, 1700, 40, 40), 0.9),
    Match('gift_icon', Rect(1020, 300, 32, 32), 0.9),
    Match('home_marker', Rect(20, 20, 16, 16), 0.9),
  ]
  action = navigator.choose(_detection(GameState.HOME, matches), Memory(),
                            None)
  assert action.kind is ActionKind.TAP
  assert action.reason == 'craft_tab'
  assert action.point == Point(620, 1720)


def test_choose_home_without_craft_tab_falls_back_to_generic():
  """HOME without a craft tab uses the generic unvisited-match fallback."""
  navigator = Navigator(object())
  match = Match('generic_icon', Rect(0, 0, 16, 16), 0.8)
  action = navigator.choose(_detection(GameState.HOME, [match]), Memory(),
                            None)
  assert action.kind is ActionKind.TAP
  assert action.reason == 'generic_icon'


def test_choose_crafting_taps_first_unvisited_item_match():
  """CRAFTING taps the first unvisited item_* match, including materials."""
  navigator = Navigator(object())
  matches = [
    Match('close_dialog_2', Rect(1000, 100, 16, 16), 0.9),
    Match('item_string', Rect(200, 300, 32, 32), 0.9),
    Match('item_silver', Rect(400, 500, 32, 32), 0.9),
  ]
  action = navigator.choose(_detection(GameState.CRAFTING, matches), Memory(),
                            None)
  assert action.kind is ActionKind.TAP
  assert action.reason == 'item_string'
  assert action.point == Point(216, 316)


def test_choose_crafting_skips_visited_item_match():
  """CRAFTING skips a visited item and taps the next unvisited item."""
  navigator = Navigator(object())
  first = Match('item_string', Rect(200, 300, 32, 32), 0.9)
  second = Match('item_silver', Rect(400, 500, 32, 32), 0.9)
  memory = Memory()
  memory.mark_visited('item_string', (200 // 16, 300 // 16))
  action = navigator.choose(
    _detection(GameState.CRAFTING, [first, second]), memory, None)
  assert action.reason == 'item_silver'


def test_choose_crafting_scrolls_when_no_item_match():
  """CRAFTING with no item match scrolls the shelf instead of tapping."""
  navigator = Navigator(object())
  matches = [Match('close_dialog_2', Rect(1000, 100, 16, 16), 0.9)]
  action = navigator.choose(_detection(GameState.CRAFTING, matches), Memory(),
                            None)
  assert action.kind is ActionKind.SWIPE
  assert action.point == Point(540, 1900)
  assert action.end == Point(540, 900)
  assert action.reason == 'craft_scroll'


def test_choose_recipe_dialog_taps_craft_button():
  """RECIPE_DIALOG taps the gold craft button when it is visible."""
  navigator = Navigator(object())
  matches = [
    Match('close_dialog_2', Rect(1000, 100, 16, 16), 0.9),
    Match('recipe_marker', Rect(100, 100, 16, 16), 0.9),
    Match('craft_button', Rect(388, 1480, 302, 132), 0.9),
  ]
  action = navigator.choose(_detection(GameState.RECIPE_DIALOG, matches),
                            Memory(), None)
  assert action.kind is ActionKind.TAP
  assert action.reason == 'craft_button'
  assert action.point == Point(539, 1546)


def test_choose_recipe_dialog_without_button_dismisses():
  """RECIPE_DIALOG without a craft button is dismissed via a close target."""
  navigator = Navigator(object())
  close = Match('close_dialog_5', Rect(100, 200, 16, 16), 0.9)
  action = navigator.choose(
    _detection(GameState.RECIPE_DIALOG, [close]), Memory(), None)
  assert action.kind is ActionKind.TAP
  assert action.reason == 'dismiss'
  assert action.point == Point(108, 208)


def test_choose_recipe_dialog_out_of_band_close_falls_back_to_back():
  """RECIPE_DIALOG with only the craft-screen X falls back to BACK."""
  navigator = Navigator(object())
  craft_x = Match('close_dialog_4', Rect(490, 2270, 100, 100), 0.9999)
  action = navigator.choose(
      _detection(GameState.RECIPE_DIALOG, [craft_x]), Memory(), None)
  assert action.kind is ActionKind.BACK
  assert action.reason == 'dismiss'


def test_choose_recipe_dialog_with_no_matches_backs_out():
  """RECIPE_DIALOG with no matches backs out like a dialog."""
  navigator = Navigator(object())
  action = navigator.choose(_detection(GameState.RECIPE_DIALOG), Memory(),
                            None)
  assert action.kind is ActionKind.BACK
  assert action.reason == 'dismiss'


def test_choose_catalog_dismisses_via_close_target():
  """CATALOG is dismissed by tapping a visible close-dialog target."""
  navigator = Navigator(object())
  close = Match('close_dialog_5', Rect(1000, 100, 16, 16), 0.9)
  action = navigator.choose(
    _detection(GameState.CATALOG, [close]), Memory(), None)
  assert action.kind is ActionKind.TAP
  assert action.reason == 'dismiss'
  assert action.point == Point(1008, 108)


def test_choose_catalog_out_of_band_close_falls_back_to_back():
  """CATALOG with only the craft-screen X falls back to BACK."""
  navigator = Navigator(object())
  craft_x = Match('close_dialog_4', Rect(490, 2270, 100, 100), 0.9999)
  action = navigator.choose(
      _detection(GameState.CATALOG, [craft_x]), Memory(), None)
  assert action.kind is ActionKind.BACK
  assert action.reason == 'dismiss'


def test_choose_catalog_without_close_target_backs_out():
  """CATALOG without a close target backs out."""
  navigator = Navigator(object())
  action = navigator.choose(_detection(GameState.CATALOG), Memory(), None)
  assert action.kind is ActionKind.BACK
  assert action.reason == 'dismiss'


def test_choose_tap_to_start_taps_its_marker_explicitly():
  """TAP_TO_START taps tap_to_start even when another match comes first."""
  navigator = Navigator(object())
  matches = [
    Match('generic_icon', Rect(0, 0, 16, 16), 0.9),
    Match('tap_to_start', Rect(400, 1200, 40, 40), 0.9),
  ]
  action = navigator.choose(
    _detection(GameState.TAP_TO_START, matches), Memory(), None)
  assert action.kind is ActionKind.TAP
  assert action.reason == 'tap_to_start'
  assert action.point == Point(420, 1220)


def test_choose_loading_and_unknown_wait():
  """LOADING and UNKNOWN states wait instead of tapping any match."""
  navigator = Navigator(object())
  for state in (GameState.LOADING, GameState.UNKNOWN):
    action = navigator.choose(
      _detection(state, [Match('generic_icon', Rect(0, 0, 16, 16), 0.9)]),
      Memory(), None)
    assert action.kind is ActionKind.WAIT
    assert action.reason == 'idle'


# --- Committed craft-button template wiring ---------------------------------

def test_craft_button_template_is_registered_in_manifest(manifest_names):
  """The recipe-dialog craft button has a committed manifest entry."""
  assert 'craft_button' in manifest_names


def test_craft_button_template_file_exists(project_root, manifest):
  """The manifest craft button points at a real committed PNG file."""
  filename = manifest['templates']['craft_button']
  assert (project_root / 'templates' / filename).is_file()
