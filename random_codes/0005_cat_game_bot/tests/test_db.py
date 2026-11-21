#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for the SQLite interaction log (plan v11 section 8)."""

# Pytest injects fixtures through parameters with the same names.
# pylint: disable=redefined-outer-name

import json
import os

import pytest

from cat_game_bot import db as db_module
from cat_game_bot.db import InteractionLog
from cat_game_bot.errors import DbError
from cat_game_bot.models import (
  Action, ActionKind, ActionResult, Match, Point, Rect,
)

TABLES = {
  'sessions', 'events', 'actions', 'observations', 'goal_progress', 'gifts',
}
INDEXES = {
  'idx_events_session_ts', 'idx_actions_session_ts',
  'idx_observations_session_ts', 'idx_goal_session_ts', 'idx_gifts_session_ts',
}


@pytest.fixture
def log():
  """Return an in-memory interaction log, closed after the test."""
  handle = InteractionLog(':memory:')
  yield handle
  handle.close()


def _object_names(handle):
  rows = handle.connection.execute(
      'SELECT type, name FROM sqlite_master').fetchall()
  by_type = {'table': set(), 'index': set()}
  for kind, name in rows:
    if kind in by_type and not name.startswith('sqlite_'):
      by_type[kind].add(name)
  return by_type


def test_schema_creates_all_tables_and_indexes(log):
  names = _object_names(log)
  assert TABLES <= names['table']
  assert INDEXES <= names['index']


def test_open_session_writes_placeholder_then_close_overwrites(log):
  sid = log.open_session('{"serial":"x"}', 'INTERRUPTED')
  assert sid == 1
  row = log.connection.execute(
      'SELECT result, finished_at, config_json FROM sessions WHERE id=?',
      (sid,)).fetchone()
  assert row[0] == 'INTERRUPTED'
  assert row[1] is None
  assert row[2] == '{"serial":"x"}'
  log.close_session(sid, 'GOAL_MET')
  done = log.connection.execute(
      'SELECT result, finished_at FROM sessions WHERE id=?',
      (sid,)).fetchone()
  assert done[0] == 'GOAL_MET'
  assert done[1] is not None


def test_log_event_stores_hostile_detail_verbatim(log):
  sid = log.open_session('{}')
  detail = {'text': 'x\'; DROP TABLE sessions; --\u2603'}
  log.log_event(sid, 1.0, 'RUN_FAULT', 'UNKNOWN', detail)
  row = log.connection.execute(
      'SELECT detail_json FROM events WHERE session_id=?', (sid,)).fetchone()
  assert json.loads(row[0]) == detail
  assert TABLES <= _object_names(log)['table']


def test_log_action_tap_encodes_point(log):
  sid = log.open_session('{}')
  action = Action(ActionKind.TAP, Point(10, 20), None, 'gift_collect')
  result = ActionResult(action, True, 'ok')
  log.log_action(sid, 2.0, action, result)
  row = log.connection.execute(
      'SELECT kind, target, x, y, ok, detail FROM actions '
      'WHERE session_id=?', (sid,)).fetchone()
  assert row == ('TAP', None, 10, 20, 1, 'ok')


def test_log_action_swipe_encodes_end_in_target(log):
  sid = log.open_session('{}')
  action = Action(ActionKind.SWIPE, Point(1, 2), Point(30, 40), 'scroll')
  log.log_action(sid, 3.0, action, ActionResult(action, True, 'ok'))
  row = log.connection.execute(
      'SELECT kind, target, x, y FROM actions WHERE session_id=?',
      (sid,)).fetchone()
  assert row == ('SWIPE', '30,40', 1, 2)


def test_log_observation_serialises_matches_and_counts(log):
  sid = log.open_session('{}')
  match = Match('bag_marker', Rect(32, 48, 16, 16), 0.9)
  log.log_observation(sid, 4.0, 'HOME', (match,), {'silver': 2}, 'deadbeef')
  row = log.connection.execute(
      'SELECT state, matches_json, counts_json, screen_hash '
      'FROM observations WHERE session_id=?', (sid,)).fetchone()
  assert row[0] == 'HOME'
  assert json.loads(row[1])[0]['name'] == 'bag_marker'
  assert json.loads(row[2]) == {'silver': 2}
  assert row[3] == 'deadbeef'


def test_log_goal_progress_and_gift_delta(log):
  sid = log.open_session('{}')
  log.log_goal_progress(sid, 5.0, 'silver', 3, 5, False)
  goal = log.connection.execute(
      'SELECT item, count, required, met FROM goal_progress '
      'WHERE session_id=?', (sid,)).fetchone()
  assert goal == ('silver', 3, 5, 0)
  log.log_gift(sid, 6.0, 100, 200)
  gift = log.connection.execute(
      'SELECT coins_before, coins_after, delta FROM gifts '
      'WHERE session_id=?', (sid,)).fetchone()
  assert gift == (100, 200, 100)


def test_close_is_idempotent(log):
  log.close()
  log.close()


def test_connect_failure_raises_db_error(monkeypatch):
  def boom(*_args, **_kwargs):
    raise OSError('no permission')

  monkeypatch.setattr(db_module.sqlite3, 'connect', boom)
  with pytest.raises(DbError):
    InteractionLog(':memory:')


def test_chmod_failure_raises_db_error(monkeypatch, tmp_path):
  target = tmp_path / 'bot.sqlite3'
  monkeypatch.setattr(db_module, '_resolve_path', lambda *_a: target)
  monkeypatch.setattr(db_module.os, 'chmod',
                      lambda *_a: (_ for _ in ()).throw(OSError('chmod')))
  with pytest.raises(DbError):
    InteractionLog(str(target))


def test_directory_and_file_modes(tmp_path, monkeypatch):
  parent = tmp_path / 'logs'
  monkeypatch.setattr(db_module, '_resolve_path',
                      lambda *_a: parent / 'bot.sqlite3')
  handle = InteractionLog('whatever')
  assert os.stat(parent).st_mode & 0o777 == 0o700
  assert os.stat(parent / 'bot.sqlite3').st_mode & 0o777 == 0o600
  handle.close()


def test_existing_log_dir_is_forced_0700(tmp_path, monkeypatch):
  """An already-existing permissive log dir is reset to mode 0700."""
  parent = tmp_path / 'logs'
  parent.mkdir()
  os.chmod(parent, 0o777)
  target = parent / 'bot.sqlite3'
  monkeypatch.setattr(db_module, '_resolve_path', lambda *_a: target)
  handle = InteractionLog('whatever')
  try:
    assert os.stat(parent).st_mode & 0o777 == 0o700
    assert os.stat(target).st_mode & 0o777 == 0o600
  finally:
    handle.close()
