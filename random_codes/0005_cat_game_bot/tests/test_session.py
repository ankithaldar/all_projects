#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for the locked session control flow (plan v11 section 6)."""

# Tests deliberately inspect Session internals and import CLI helpers lazily.
# pylint: disable=protected-access,import-outside-toplevel

import json
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from cat_game_bot.config import Config
from cat_game_bot.db import InteractionLog
from cat_game_bot.errors import AdbError, DbError, VisionError
from cat_game_bot.gift import GiftCollector
from cat_game_bot.models import (
  Action, ActionKind, ActionResult, CraftStep, Detection, GameState,
  GiftResult, Match, Point, Rect, Screen, SessionStats,
)
from cat_game_bot.navigator import Navigator
from cat_game_bot.session import Session
from cat_game_bot.targets import TARGET_IDS, TargetSpec


class Pixels:
  """Minimal pixel buffer providing ``tobytes`` for hashing."""

  def tobytes(self):
    """Return deterministic bytes."""
    return b'01234567'


class DetectSequence:
  """Return each supplied detection for two consecutive detect calls."""

  def __init__(self, steps):
    """Store steps as ``(state, matches)`` pairs."""
    self.steps = steps
    self.index = 0
    self.calls = 0

  def __call__(self, screen):
    """Return the next detection, repeating a step for stability."""
    step = self.steps[min(self.index // 2, len(self.steps) - 1)]
    self.index += 1
    self.calls += 1
    return Detection(step[0], tuple(step[1]), float(self.calls))


class DetectionSequence:
  """Return one supplied detection per call, then repeat the last one."""

  def __init__(self, detections):
    """Store the ordered detections and a call cursor."""
    self.detections = list(detections)
    self.index = 0

  def __call__(self, unused_screen):
    """Return the next detection for a settle poll."""
    detection = self.detections[min(self.index, len(self.detections) - 1)]
    self.index += 1
    return detection


def _det(state, matches=()):
  """Build a detection with a stable timestamp for settle tests."""
  return Detection(state, tuple(matches), 1.0)


def _settle_env(detections, **config_kw):
  """Build a session whose detector yields one detection per settle poll."""
  env = _make([(GameState.HOME, [])], **config_kw)
  env.detector.detect.side_effect = DetectionSequence(detections)
  return env


def _make(states, **config_kw):
  """Build a fully mocked session plus handles to its dependencies."""
  overrides = {
    'dry_run': False, 'max_actions': 50, 'no_progress_limit': 100,
    'error_limit': 100, 'duration_seconds': 1000.0,
    'bag_visit_interval': 1000000, 'poll_delay': 0.0,
  }
  overrides.update(config_kw)
  cfg = replace(Config(), **overrides)

  adb = MagicMock()
  adb.is_connected.return_value = True
  adb.screenshot.side_effect = lambda: Screen(Pixels())

  detector = MagicMock()
  detector.detect.side_effect = DetectSequence(states)

  navigator = MagicMock()
  navigator.choose.return_value = Action(ActionKind.WAIT, None, None,
                                         'general')
  navigator.action_for_template.return_value = None

  ocr = MagicMock()
  inventory = MagicMock()
  inventory.read.return_value = {}
  planner = MagicMock()
  planner.next_step.return_value = None

  gift = MagicMock()
  gift.is_ready.return_value = False
  gift.plan.return_value = Action(ActionKind.TAP, Point(540, 615), None,
                                  'gift_icon')
  gift.read_coins.return_value = None
  gift.verify.return_value = GiftResult(True, 100, 'ok')

  db = MagicMock()
  db.open_session.return_value = 1

  targets = TargetSpec(items={k: 5 for k in TARGET_IDS},
                       initial_coins=0, initial_stash={})
  times = iter(range(0, 100000))
  session = Session(adb, detector, navigator, ocr, inventory, planner,
                    gift, db, cfg, targets, now=lambda: next(times))
  return SimpleNamespace(session=session, adb=adb, detector=detector,
                         navigator=navigator, ocr=ocr, inventory=inventory,
                         planner=planner, gift=gift, db=db, config=cfg)


def _gift_actions(db):
  """Return action reasons logged to the mock database."""
  return [call.args[2].reason for call in db.log_action.call_args_list]


def test_dry_run_first_screen_gift_dialog_has_zero_device_taps():
  env = _make([(GameState.GIFT_DIALOG, [])], dry_run=True, max_actions=1)
  stats = env.session.run()
  assert env.adb.tap.call_count == 0
  assert env.adb.launch.call_count == 0
  assert env.gift.read_coins.call_count == 0
  assert env.gift.verify.call_count == 0
  assert env.db.log_gift.call_count == 0
  assert stats.errors == 0
  assert stats.result == 'MAX_ACTIONS'
  assert stats.states_seen[GameState.GIFT_DIALOG.value] == 1


def test_live_first_screen_gift_dialog_no_before_logs_one_failure():
  env = _make([(GameState.GIFT_DIALOG, []), (GameState.HOME, [])],
              max_actions=2)
  env.gift.read_coins.return_value = None
  stats = env.session.run()
  reasons = _gift_actions(env.db)
  assert 'gift_no_before' in reasons
  failure_call = next(call for call in env.db.log_action.call_args_list
                      if call.args[2].reason == 'gift_no_before')
  assert failure_call.args[3].ok is False
  assert stats.errors == 1
  assert env.adb.tap.call_count == 0
  assert 'general' in reasons
  assert env.session._gift_collect_tapped is False
  assert stats.result == 'MAX_ACTIONS'


def test_phase1_unreadable_coins_then_later_gift_completes():
  env = _make([(GameState.HOME, []), (GameState.HOME, []),
               (GameState.GIFT_DIALOG, []), (GameState.HOME, [])],
              max_actions=4)
  env.gift.is_ready.side_effect = [True, True, False]
  env.gift.read_coins.side_effect = [None, 100, 200]
  env.gift.verify.return_value = GiftResult(True, 100, 'ok')
  collect = Action(ActionKind.TAP, Point(540, 1625), None, 'gift_collect')
  env.navigator.choose.side_effect = lambda det, mem, act: (
      collect if det.state is GameState.GIFT_DIALOG
      else Action(ActionKind.WAIT, None, None, 'general'))
  stats = env.session.run()
  reasons = _gift_actions(env.db)
  assert 'gift_coins_unreadable' in reasons
  assert reasons.count('gift_collect') == 1
  assert env.db.log_gift.call_count == 1
  before, after = env.db.log_gift.call_args.args[2:4]
  assert before == 100
  assert after == 200
  assert before is not None
  assert stats.gifts_collected == 1
  assert stats.errors == 1
  assert env.adb.tap.call_count == 2
  assert env.session._pending_gift_before is None
  assert env.session._gift_collect_tapped is False


def test_session_calls_gift_plan_with_a_single_screen_argument():
  """Session uses the locked one-argument gift.plan(screen) signature."""
  env = _make([(GameState.HOME, [])], max_actions=3)
  env.gift.is_ready.return_value = True
  env.gift.read_coins.return_value = 100
  env.session.run()
  assert env.gift.plan.call_count >= 1
  assert all(len(call.args) == 1 and call.kwargs == {}
             for call in env.gift.plan.call_args_list)


def test_verify_failure_logs_one_failure_row_and_no_gift():
  env = _make([(GameState.HOME, []), (GameState.GIFT_DIALOG, []),
               (GameState.HOME, [])], max_actions=3)
  env.gift.is_ready.return_value = True
  env.gift.read_coins.side_effect = [100, 100]
  env.gift.verify.return_value = GiftResult(False, 0, 'no coin delta')
  env.navigator.choose.side_effect = lambda det, mem, act: (
      Action(ActionKind.TAP, Point(540, 1625), None, 'gift_collect')
      if det.state is GameState.GIFT_DIALOG
      else Action(ActionKind.WAIT, None, None, 'general'))
  stats = env.session.run()
  reasons = _gift_actions(env.db)
  assert 'gift_verify_failed' in reasons
  failed = next(call for call in env.db.log_action.call_args_list
                if call.args[2].reason == 'gift_verify_failed')
  assert failed.args[3].ok is False
  assert failed.args[3].detail == 'no coin delta'
  assert env.db.log_gift.call_count == 0
  assert stats.gifts_collected == 0
  assert stats.errors == 1


def test_two_consecutive_gift_dialog_ticks_yield_one_collect_tap():
  env = _make([(GameState.GIFT_DIALOG, []), (GameState.GIFT_DIALOG, []),
               (GameState.HOME, []), (GameState.HOME, [])], max_actions=2)
  env.gift.is_ready.return_value = False
  env.gift.read_coins.side_effect = [100, 200]
  env.navigator.choose.side_effect = lambda det, mem, act: (
      Action(ActionKind.TAP, Point(540, 1625), None, 'gift_collect')
      if det.state is GameState.GIFT_DIALOG
      else Action(ActionKind.WAIT, None, None, 'general'))
  env.session.run()
  reasons = _gift_actions(env.db)
  assert reasons.count('gift_collect') == 1
  assert reasons[0] == 'gift_collect'
  assert 'general' not in reasons[:1]
  assert env.adb.tap.call_count == 1
  assert env.adb.tap.call_args.args[0] == Point(540, 1625)
  assert env.db.log_gift.call_count == 1


def test_max_actions_stop():
  env = _make([(GameState.HOME, [])], max_actions=1)
  stats = env.session.run()
  assert stats.result == 'MAX_ACTIONS'
  assert stats.actions == 1


def test_no_progress_stop():
  env = _make([(GameState.HOME, [])], no_progress_limit=2, max_actions=50)
  stats = env.session.run()
  assert stats.result == 'NO_PROGRESS'
  assert env.session.memory.no_progress == 2


def test_error_limit_stop_via_action_failures():
  env = _make([(GameState.HOME, [])], error_limit=2)
  env.gift.is_ready.return_value = True
  env.gift.read_coins.return_value = None
  stats = env.session.run()
  assert stats.result == 'ERROR_LIMIT'
  assert stats.errors == 2


def test_duration_stop():
  env = _make([(GameState.HOME, [])], duration_seconds=1.0)
  times = iter(range(0, 100000, 100))
  env.session.now = lambda: next(times)
  stats = env.session.run()
  assert stats.result == 'DURATION'


def test_disconnected_stop():
  env = _make([(GameState.HOME, [])])
  env.adb.is_connected.side_effect = [True, False]
  stats = env.session.run()
  assert stats.result == 'DISCONNECTED'


def test_initial_disconnect_returns_stats_and_closes_db():
  env = _make([(GameState.HOME, [])], dry_run=True)
  env.adb.is_connected.return_value = False
  stats = env.session.run()
  assert stats.result == 'DISCONNECTED'
  assert stats.screenshots == 0
  assert env.db.open_session.called
  assert env.db.close_session.call_args.args[1] == 'DISCONNECTED'
  assert env.db.close.called


def test_goal_met_stop():
  env = _make([(GameState.BAG_DIALOG, [])])
  env.inventory.read.return_value = {k: 5 for k in TARGET_IDS}
  stats = env.session.run()
  assert stats.result == 'GOAL_MET'
  assert env.db.log_goal_progress.call_count == len(TARGET_IDS)


def test_keyboard_interrupt_maps_to_interrupted_and_closes_db():
  env = _make([(GameState.HOME, [])])
  env.detector.detect.side_effect = KeyboardInterrupt
  stats = env.session.run()
  assert stats.result == 'INTERRUPTED'
  assert env.db.close_session.call_args.args[1] == 'INTERRUPTED'
  assert env.db.close.called


def test_system_exit_maps_to_interrupted_and_closes_db():
  """SystemExit is mapped to INTERRUPTED with deterministic cleanup."""
  env = _make([(GameState.HOME, [])])
  env.detector.detect.side_effect = SystemExit
  stats = env.session.run()
  assert stats.result == 'INTERRUPTED'
  assert env.db.close_session.call_args.args[1] == 'INTERRUPTED'
  assert env.db.close.called


def test_vision_fault_maps_to_error_limit_with_run_fault():
  env = _make([(GameState.HOME, [])])
  env.gift.is_ready.return_value = True
  env.gift.read_coins.side_effect = VisionError('ocr exploded')
  stats = env.session.run()
  assert stats.result == 'ERROR_LIMIT'
  assert stats.errors == 1
  fault = env.db.log_event.call_args
  assert fault.args[2] == 'RUN_FAULT'
  assert fault.args[4] == {'detail': 'ocr exploded'}
  assert env.db.close_session.call_args.args[1] == 'ERROR_LIMIT'
  assert env.db.close.called


def test_db_error_in_first_handler_is_swallowed():
  env = _make([(GameState.HOME, [])])
  env.adb.is_connected.side_effect = AdbError('adb down')
  env.db.log_event.side_effect = DbError('db down')
  stats = env.session.run()
  assert stats.result == 'ERROR_LIMIT'
  assert stats.errors == 1
  assert env.db.close_session.call_args.args[1] == 'ERROR_LIMIT'
  assert env.db.close.called


def test_db_error_from_log_call_maps_to_error_limit():
  env = _make([(GameState.HOME, [])])
  env.db.log_observation.side_effect = DbError('write failed')
  stats = env.session.run()
  assert stats.result == 'ERROR_LIMIT'
  assert stats.errors == 1
  assert env.db.log_event.call_args.args[2] == 'RUN_FAULT'
  assert stats.finished_at >= stats.started_at


def test_db_error_from_close_session_still_closes_db():
  env = _make([(GameState.HOME, [])], max_actions=1)
  env.db.close_session.side_effect = DbError('close failed')
  with pytest.raises(DbError):
    env.session.run()
  assert env.db.close.called


def test_execute_dry_run_and_live_failure():
  env = _make([(GameState.HOME, [])], dry_run=True)
  action = Action(ActionKind.TAP, Point(1, 2), None, 'tap')
  result = env.session._execute(action)
  assert result.ok is True
  assert result.detail == 'dry_run simulated'
  assert env.adb.tap.call_count == 0

  env.session.config = replace(env.config, dry_run=False)
  env.adb.tap.side_effect = AdbError('tap failed')
  result = env.session._execute(action)
  assert result.ok is False
  assert result.detail == 'tap failed'


class _EqualTapKind:
  """A foreign kind equal to TAP but not the ActionKind enum singleton."""

  def __eq__(self, other):
    """Report equality only with the canonical TAP kind."""
    return other is ActionKind.TAP


def test_execute_requires_identity_for_tap_kind():
  """_execute dispatches taps only for the canonical TAP singleton."""
  env = _make([(GameState.HOME, [])])
  action = Action(_EqualTapKind(), Point(1, 2), None, 'foreign')
  result = env.session._execute(action)
  assert result.ok is True
  assert env.adb.tap.call_count == 0


def test_record_outcome_and_count_failure():
  env = _make([(GameState.HOME, [])])
  env.session.session_id = 7
  env.session._stats = SessionStats()
  action = Action(ActionKind.TAP, None, None, 'x')
  ok = env.session._record_outcome(ActionResult(action, True, 'fine'))
  assert ok.ok is True
  assert env.session._stats.actions == 1
  assert env.session._stats.errors == 0

  bad = env.session._record_outcome(ActionResult(action, False, 'broken'))
  assert bad.ok is False
  assert env.session._stats.actions == 2
  assert env.session._stats.errors == 1
  event = env.db.log_event.call_args.args
  assert event[0] == 7
  assert event[2] == 'ACTION_FAILURE'
  assert event[4] == {'detail': 'broken'}


def test_config_json_is_serialised_to_open_session():
  env = _make([(GameState.HOME, [])], max_actions=1)
  env.session.run()
  payload = env.db.open_session.call_args.args[0]
  assert '"serial"' in payload
  assert env.db.open_session.call_args.args[1] == 'INTERRUPTED'


def test_main_exit_mapping(monkeypatch):
  from cat_game_bot import __main__ as cli

  fake = SessionStats(result='GOAL_MET')
  monkeypatch.setattr(cli, '_run', lambda argv: fake)
  assert cli.main([]) == 0

  fake.result = 'MAX_ACTIONS'
  assert cli.main([]) == 1

  def boom(argv):
    raise DbError('setup failed')

  monkeypatch.setattr(cli, '_run', boom)
  assert cli.main([]) == 2


# --- F7: settle() -----------------------------------------------------------

def test_settle_waits_through_loading_until_stable():
  """settle polls past LOADING and returns once two HOME polls agree."""
  detections = [
    _det(GameState.LOADING), _det(GameState.LOADING),
    _det(GameState.HOME), _det(GameState.HOME),
  ]
  env = _settle_env(detections, settle_timeout=10.0)
  screen, detection = env.session.settle()
  assert isinstance(screen, Screen)
  assert detection.state is GameState.HOME
  assert env.detector.detect.call_count == 4
  assert env.db.log_event.call_count == 0


def test_settle_requires_two_consecutive_identical_signatures():
  """One stable state is not enough; the next poll must match it."""
  first = (Match('item_silver', Rect(0, 0, 16, 16), 0.9),)
  second = (Match('item_silver', Rect(160, 0, 16, 16), 0.9),)
  detections = [
    _det(GameState.HOME, first),
    _det(GameState.HOME, second),
    _det(GameState.HOME, second),
  ]
  env = _settle_env(detections, settle_timeout=10.0)
  unused_screen, detection = env.session.settle()
  assert detection.matches == second
  assert env.detector.detect.call_count == 3


def test_settle_does_not_mask_coin_and_timer_named_matches():
  """No template is named 'coins'/'gift_timer', so masking was dead code."""
  detections = [
    _det(GameState.HOME, (Match('coins', Rect(0, 0, 16, 16), 0.9),)),
    _det(GameState.HOME, (Match('coins', Rect(160, 0, 16, 16), 0.9),)),
    _det(GameState.HOME, (Match('gift_timer', Rect(0, 0, 16, 16), 0.9),)),
  ]
  env = _settle_env(detections, settle_timeout=2.0)
  env.session.settle()
  assert env.db.log_event.call_count == 1
  assert env.db.log_event.call_args.args[2] == 'SETTLE_TIMEOUT'


def test_settle_logs_settle_timeout_once_and_returns_last_pair():
  """Timeout logs one SETTLE_TIMEOUT event and returns the final pair."""
  detections = [
    _det(GameState.HOME, (Match('item_silver', Rect(0, 0, 16, 16), 0.9),)),
    _det(GameState.HOME, (Match('item_silver', Rect(160, 0, 16, 16), 0.9),)),
  ]
  env = _settle_env(detections, settle_timeout=2.0)
  screen, detection = env.session.settle()
  event = env.db.log_event.call_args.args
  assert env.db.log_event.call_count == 1
  assert event[0] == env.session.session_id
  assert event[2] == 'SETTLE_TIMEOUT'
  assert event[3] == GameState.HOME.value
  assert event[4] == {}
  assert detection is detections[-1]
  assert isinstance(screen, Screen)


def test_settle_returns_only_one_screen_to_caller():
  """settle may poll many frames but returns exactly one screen."""
  env = _make([(GameState.HOME, [])], settle_timeout=10.0)
  only_screen = Screen(Pixels())
  env.adb.screenshot.side_effect = None
  env.adb.screenshot.return_value = only_screen
  result = env.session.settle()
  assert len(result) == 2
  assert result[0] is only_screen
  assert env.adb.screenshot.call_count == 2


# --- F12: periodic Bag visit -----------------------------------------------

def test_periodic_bag_visit_taps_bag_marker_at_configured_interval():
  """At the configured action count the bag marker is tapped."""
  env = _make([(GameState.HOME, [])] * 4, bag_visit_interval=1,
              max_actions=3)
  bag = Action(ActionKind.TAP, Point(100, 100), None, 'bag_marker')
  env.navigator.action_for_template.side_effect = (
      lambda det, name: bag if name == 'bag_marker' else None)
  env.session.run()
  reasons = _gift_actions(env.db)
  assert 'bag_marker' in reasons
  assert env.adb.tap.called


def test_goal_detected_after_periodic_bag_revisit():
  """A Bag revisit that reads goal counts ends the run with GOAL_MET."""
  env = _make([(GameState.HOME, []), (GameState.HOME, []),
               (GameState.BAG_DIALOG, [])], bag_visit_interval=1,
              max_actions=10)
  bag = Action(ActionKind.TAP, Point(100, 100), None, 'bag_marker')
  env.navigator.action_for_template.side_effect = (
      lambda det, name: bag if name == 'bag_marker' else None)
  env.inventory.read.return_value = {k: 5 for k in TARGET_IDS}
  stats = env.session.run()
  assert stats.result == 'GOAL_MET'
  assert env.db.log_goal_progress.call_count == len(TARGET_IDS)


def test_gift_handled_tick_suppresses_periodic_bag_visit():
  """A tick that performs a gift action never also taps the bag marker."""
  env = _make([(GameState.HOME, [])] * 3, bag_visit_interval=1,
              max_actions=2)
  bag = Action(ActionKind.TAP, Point(100, 100), None, 'bag_marker')
  env.navigator.action_for_template.side_effect = (
      lambda det, name: bag if name == 'bag_marker' else None)
  env.gift.is_ready.side_effect = [False, True, False]
  env.gift.read_coins.side_effect = [100, 200]
  env.session.run()
  reasons = _gift_actions(env.db)
  assert 'gift_icon' in reasons
  assert 'bag_marker' not in reasons


# --- F10 / F16: gift logging strength --------------------------------------

def test_phase3_logs_real_before_after_and_delta():
  """The verified gift row carries the real before and after coin counts."""
  env = _make([(GameState.HOME, []), (GameState.GIFT_DIALOG, []),
               (GameState.HOME, [])], max_actions=3)
  env.gift.is_ready.return_value = True
  env.gift.read_coins.side_effect = [1000, 11160, 999]
  env.gift.verify.return_value = GiftResult(True, 10160, 'ok')
  env.navigator.choose.side_effect = lambda det, mem, act: (
      Action(ActionKind.TAP, Point(540, 1625), None, 'gift_collect')
      if det.state is GameState.GIFT_DIALOG
      else Action(ActionKind.WAIT, None, None, 'general'))
  stats = env.session.run()
  assert env.db.log_gift.call_count == 1
  before, after = env.db.log_gift.call_args.args[2:4]
  assert before == 1000
  assert after == 11160
  assert after - before == 10160
  assert stats.gifts_collected == 1


@pytest.mark.parametrize('reason', [
    'gift_coins_unreadable', 'gift_no_before', 'gift_verify_failed',
])
def test_failure_action_rows_are_ok_false_and_log_no_gift(reason):
  """Every gift failure emits one ok=False action and no gift row."""
  if reason == 'gift_coins_unreadable':
    env = _make([(GameState.HOME, [])], max_actions=1)
    env.gift.is_ready.return_value = True
    env.gift.read_coins.return_value = None
  elif reason == 'gift_no_before':
    env = _make([(GameState.GIFT_DIALOG, [])], max_actions=1)
    env.gift.read_coins.return_value = None
  else:
    env = _make([(GameState.HOME, []), (GameState.GIFT_DIALOG, []),
                 (GameState.HOME, [])], max_actions=3)
    env.gift.is_ready.return_value = True
    env.gift.read_coins.side_effect = [100, 100]
    env.gift.verify.return_value = GiftResult(False, 0, 'no delta')
    env.navigator.choose.side_effect = lambda det, mem, act: (
        Action(ActionKind.TAP, Point(540, 1625), None, 'gift_collect')
        if det.state is GameState.GIFT_DIALOG
        else Action(ActionKind.WAIT, None, None, 'general'))
  env.session.run()
  assert reason in _gift_actions(env.db)
  failure = next(call for call in env.db.log_action.call_args_list
                 if call.args[2].reason == reason)
  assert failure.args[3].ok is False
  assert env.db.log_gift.call_count == 0


def test_unreadable_coins_tick_logs_only_failure_and_no_tap():
  """The unreadable-coins tick emits only its failure and taps nothing."""
  env = _make([(GameState.HOME, [])], max_actions=1)
  env.gift.is_ready.return_value = True
  env.gift.read_coins.return_value = None
  env.session.run()
  assert _gift_actions(env.db) == ['gift_coins_unreadable']
  assert env.adb.tap.call_count == 0
  assert env.db.log_action.call_args.args[3].ok is False


def test_phase1_does_not_overwrite_pending_before_while_collect_tapped():
  """A second dialog tick cannot replace the pending before value."""
  env = _make([(GameState.GIFT_DIALOG, []), (GameState.GIFT_DIALOG, []),
               (GameState.HOME, [])], max_actions=2)
  env.gift.is_ready.return_value = True
  env.gift.read_coins.side_effect = [100, 200, 999]
  env.gift.verify.return_value = GiftResult(True, 100, 'ok')
  env.navigator.choose.side_effect = lambda det, mem, act: (
      Action(ActionKind.TAP, Point(540, 1625), None, 'gift_collect')
      if det.state is GameState.GIFT_DIALOG
      else Action(ActionKind.WAIT, None, None, 'general'))
  env.session.run()
  assert env.db.log_gift.call_count == 1
  before, after = env.db.log_gift.call_args.args[2:4]
  assert before == 100
  assert after == 200


def test_action_failure_row_persists_session_id_and_detail_in_real_db():
  """_count_failure writes a durable row with its session id and detail."""
  env = _make([(GameState.HOME, [])])
  real_db = InteractionLog(':memory:')
  session_id = real_db.open_session('{}')
  env.session.db = real_db
  env.session.session_id = session_id
  env.session._stats = SessionStats()
  action = Action(ActionKind.TAP, None, None, 'x')
  env.session._record_outcome(ActionResult(action, False, 'broken detail'))
  row = real_db.connection.execute(
      'SELECT session_id, event, detail_json FROM events').fetchone()
  assert row is not None
  assert row[0] == session_id
  assert row[1] == 'ACTION_FAILURE'
  assert json.loads(row[2]) == {'detail': 'broken detail'}
  real_db.close()


# --- F20 / F22: batch state machine and universal fallback ------------------

def _craft_step(item='silver', batch=2):
  """Build a CraftStep fixture for batch-state tests."""
  return CraftStep(item, 'JEWEL CRAFTING', f'item_{item}', batch, 100)


def _use_real_navigator(env):
  """Wire a real Navigator into a mocked session and return it."""
  navigator = Navigator(object())
  env.navigator = navigator
  env.session.navigator = navigator
  return navigator


def test_batch2_craft_collect_craft_collect_then_mark_placed_once():
  """A batch of 2 emits craft/collect/craft/collect then marks placed once."""
  craft = Match('item_silver', Rect(100, 200, 16, 16), 0.9)
  ready = Match('craft_ready_check', Rect(300, 400, 16, 16), 0.9)
  states = [
    (GameState.CRAFTING, [craft]),
    (GameState.CRAFTING, [ready]),
    (GameState.CRAFTING, [craft]),
    (GameState.CRAFTING, [ready]),
    (GameState.CRAFTING, []),
  ]
  env = _make(states, max_actions=5)
  _use_real_navigator(env)
  env.planner.next_step.return_value = _craft_step()
  stats = env.session.run()
  assert _gift_actions(env.db) == [
      'item_silver', 'craft_ready_check', 'item_silver',
      'craft_ready_check', 'craft_scroll']
  env.planner.mark_placed.assert_called_once_with('silver')
  assert env.planner.next_step.call_count == 2
  assert stats.actions == 5
  assert stats.errors == 0
  assert env.adb.tap.call_count == 4
  assert env.adb.swipe.call_count == 1
  assert stats.result == 'MAX_ACTIONS'


def test_failed_x1_tap_keeps_remaining_and_never_marks_placed():
  """A failed x1 tap must not consume a batch slot or advance the planner."""
  craft = Match('item_silver', Rect(100, 200, 16, 16), 0.9)
  env = _make([(GameState.CRAFTING, [craft])], max_actions=1)
  _use_real_navigator(env)
  env.planner.next_step.return_value = _craft_step(batch=2)
  env.adb.tap.side_effect = AdbError('tap failed')
  stats = env.session.run()
  assert env.session._step_remaining == 2
  env.planner.mark_placed.assert_not_called()
  assert env.db.log_action.call_count == 1
  assert env.db.log_action.call_args.args[3].ok is False
  assert stats.errors == 1
  assert stats.actions == 1


def test_no_collect_tap_when_no_completed_tile():
  """A normal craft shelf produces no craft_ready_check collect tap."""
  craft = Match('item_silver', Rect(100, 200, 16, 16), 0.9)
  env = _make([(GameState.CRAFTING, [craft])], max_actions=1)
  _use_real_navigator(env)
  env.planner.next_step.return_value = _craft_step()
  env.session.run()
  assert _gift_actions(env.db) == ['item_silver']


def test_popup_during_batch_executes_one_dismiss_fallback_action():
  """A popup with the step template absent yields one BACK dismiss."""
  craft = Match('item_silver', Rect(100, 200, 16, 16), 0.9)
  close = Match('close_dialog_2', Rect(1000, 100, 16, 16), 0.9)
  ready = Match('craft_ready_check', Rect(300, 400, 16, 16), 0.9)
  states = [
    (GameState.CRAFTING, [craft]),
    (GameState.POPUP, [close]),
    (GameState.CRAFTING, [craft]),
    (GameState.CRAFTING, [ready]),
  ]
  env = _make(states, max_actions=4)
  _use_real_navigator(env)
  env.planner.next_step.return_value = _craft_step()
  stats = env.session.run()
  assert _gift_actions(env.db) == [
      'item_silver', 'dismiss', 'item_silver', 'craft_ready_check']
  env.planner.mark_placed.assert_called_once_with('silver')
  assert env.planner.next_step.call_count == 1
  assert stats.actions == 4
  assert stats.errors == 0
  assert env.adb.tap.call_count == 3
  assert env.adb.back.call_count == 1
  dismiss = next(call for call in env.db.log_action.call_args_list
                 if call.args[2].reason == 'dismiss')
  assert dismiss.args[2].kind is ActionKind.BACK
  assert dismiss.args[3].ok is True


def test_ok_fallback_logs_one_action_increment_and_zero_errors():
  """A BACK dismiss contributes one ACTION row and one action."""
  close = Match('close_dialog_3', Rect(1000, 100, 16, 16), 0.9)
  env = _make([(GameState.POPUP, [close])], max_actions=1)
  _use_real_navigator(env)
  env.planner.next_step.return_value = _craft_step()
  stats = env.session.run()
  assert env.db.log_action.call_count == 1
  assert stats.actions == 1
  assert stats.errors == 0
  assert env.adb.tap.call_count == 0
  assert env.adb.back.call_count == 1
  assert env.db.log_action.call_args.args[2].reason == 'dismiss'
  assert env.db.log_action.call_args.args[3].ok is True


def test_ok_fallback_persists_one_action_row_and_zero_errors():
  """A successful BACK dismiss writes exactly one ACTION row and no error."""
  close = Match('close_dialog_3', Rect(1000, 100, 16, 16), 0.9)
  env = _make([(GameState.POPUP, [close])], max_actions=1)
  _use_real_navigator(env)
  real_db = InteractionLog(':memory:')
  real_close = real_db.close
  real_db.close = lambda: None
  env.session.db = real_db
  stats = env.session.run()
  rows = real_db.connection.execute(
      'SELECT kind, ok FROM actions').fetchall()
  assert len(rows) == 1
  assert rows[0][0] == ActionKind.BACK.value
  assert rows[0][1] == 1
  assert stats.actions == 1
  assert stats.errors == 0
  real_close()


def test_batch_resumes_after_periodic_bag_visit_and_back():
  """Popup and bag BACK dismisses let the batch still reach mark_placed."""
  craft = Match('item_silver', Rect(100, 200, 16, 16), 0.9)
  close = Match('close_dialog_2', Rect(1000, 100, 16, 16), 0.9)
  bag = Match('bag_marker', Rect(200, 2200, 16, 16), 0.9)
  stock = Match('inventory_marker', Rect(100, 100, 16, 16), 0.9)
  states = [
    (GameState.CRAFTING, [craft]),
    (GameState.POPUP, [close]),
    (GameState.POPUP, [close]),
    (GameState.CRAFTING, [bag]),
    (GameState.BAG_DIALOG, [stock]),
    (GameState.CRAFTING, [craft]),
  ]
  env = _make(states, max_actions=6, bag_visit_interval=3)
  _use_real_navigator(env)
  env.planner.next_step.return_value = _craft_step()
  stats = env.session.run()
  assert _gift_actions(env.db) == [
      'item_silver', 'dismiss', 'dismiss', 'bag_marker',
      'close_bag', 'item_silver']
  env.planner.mark_placed.assert_called_once_with('silver')
  assert env.adb.back.call_count == 3
  assert stats.errors == 0
  assert stats.result == 'MAX_ACTIONS'


def test_home_detection_taps_craft_tab_with_real_navigator():
  """A real HOME detection yields a craft-tab tap, not the close X."""
  craft = Match('craft_tab', Rect(600, 1700, 40, 40), 0.9)
  close = Match('close_dialog_2', Rect(544, 1983, 16, 16), 0.9)
  env = _make([(GameState.HOME, [close, craft])], max_actions=1)
  _use_real_navigator(env)
  stats = env.session.run()
  assert _gift_actions(env.db) == ['craft_tab']
  assert env.adb.tap.call_count == 1
  assert env.adb.tap.call_args.args[0] == Point(620, 1720)
  assert stats.errors == 0


def test_splash_blank_timer_no_coins_yields_no_gift_action():
  """A splash screen must navigate instead of opening a gift cycle."""

  class SplashOcr:
    """OCR stub returning blank text for every region on a splash."""

    def read(self, unused_screen, unused_rect):
      """Return no text, as on the TAP TO START splash screen."""
      return ''

  collector = GiftCollector(
      SplashOcr(), {'coins': Rect(0, 0, 200, 50),
                    'gift_timer': Rect(100, 100, 50, 50)})
  start = Match('tap_to_start', Rect(400, 1200, 40, 40), 0.9)
  env = _make([(GameState.TAP_TO_START, [start])], max_actions=1)
  env.session.gift = collector
  _use_real_navigator(env)
  stats = env.session.run()
  reasons = _gift_actions(env.db)
  assert reasons == ['tap_to_start']
  assert 'gift_icon' not in reasons
  assert stats.errors == 0
  assert env.adb.tap.call_count == 1
