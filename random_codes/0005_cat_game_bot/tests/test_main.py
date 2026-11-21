#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for the canonical CLI wiring in cat_game_bot.__main__."""

# The CLI wiring under test lives in the private session builder.
# pylint: disable=protected-access

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import yaml

import cat_game_bot.__main__ as cli
import cat_game_bot.config as config
from cat_game_bot.errors import ConfigError, DbError
from cat_game_bot.inventory import CANONICAL_IDS
from cat_game_bot.models import Rect, SessionStats
from cat_game_bot.session import Session

_TARGET_IDS = (
  'silver', 'gold', 'pendant', 'necklace', 'orb', 'water', 'fire',
  'waterstone', 'firestone', 'elementstone', 'artifact')


def _valid_regions_doc():
  """Return a schema-valid ocr-regions document for the test harness.

  Returns:
    dict: A complete, valid regions document.
  """
  return {
    'version': 1,
    'regions': {
      'coins': {'x': 505, 'y': 150, 'w': 240, 'h': 70},
      'gift_timer': {'x': 940, 'y': 620, 'w': 135, 'h': 70},
    },
    'offsets': {
      f'item_count.{item}': {'dx': 0, 'dy': 140, 'w': 120, 'h': 64}
      for item in CANONICAL_IDS
    },
  }


def _valid_targets_doc():
  """Return a schema-valid initial-state document.

  Returns:
    dict: A complete, valid targets document.
  """
  return {
    'targets': {item: 1 for item in _TARGET_IDS},
    'initialCoins': 0,
    'initialStash': {},
  }


def _write_project(tmp_path, tree):
  """Create a minimal project tree of valid files except the tree.

  Args:
    tmp_path: Temporary project root provided by pytest.
    tree: Crafting tree document to write as YAML.

  Returns:
    Path: The created data directory.
  """
  templates = tmp_path / 'templates'
  templates.mkdir()
  (templates / 'manifest.json').write_text(
      json.dumps({'version': 1, 'templates': {}}), encoding='utf-8')
  data = tmp_path / 'data'
  data.mkdir()
  (data / 'ocr-regions.json').write_text(
      json.dumps(_valid_regions_doc()), encoding='utf-8')
  (data / 'initial-state.json').write_text(
      json.dumps(_valid_targets_doc()), encoding='utf-8')
  (data / 'crafting_tree.yaml').write_text(
      yaml.safe_dump(tree), encoding='utf-8')
  return data


def _parse(tmp_path, monkeypatch):
  """Point the project root at the temp tree and parse default CLI args.

  Args:
    tmp_path: Temporary project root provided by pytest.
    monkeypatch: Pytest monkeypatch fixture.

  Returns:
    Config: The parsed frozen configuration.
  """
  monkeypatch.setattr(config, 'PROJECT_ROOT', tmp_path)
  return cli.parse_args(['--serial', 'dev-1'])


def test_build_session_rejects_schema_invalid_tree(tmp_path, monkeypatch):
  """A tree without the required items list is rejected via the CLI."""
  _write_project(tmp_path, {'version': 1})
  cfg = _parse(tmp_path, monkeypatch)
  with pytest.raises(ConfigError):
    cli._build_session(cfg, object())


def test_build_session_rejects_schema_invalid_targets(tmp_path, monkeypatch):
  """A targets file missing goal counts is rejected via the CLI."""
  data = _write_project(tmp_path, {'items': []})
  (data / 'initial-state.json').write_text(
      json.dumps({'targets': {}, 'initialCoins': 0, 'initialStash': {}}),
      encoding='utf-8')
  cfg = _parse(tmp_path, monkeypatch)
  with pytest.raises(ConfigError):
    cli._build_session(cfg, object())


def test_build_session_uses_validated_loaders(tmp_path, monkeypatch):
  """The CLI builds a Session from schema-validated data files."""
  _write_project(tmp_path, {'items': []})
  cfg = _parse(tmp_path, monkeypatch)
  session = cli._build_session(cfg, object())
  assert isinstance(session, Session)
  assert session.gift.regions['coins'] == Rect(505, 150, 240, 70)
  assert set(session.inventory.offsets) == set(config.OFFSET_NAMES)


def test_cli_has_no_dead_box_conversion_helpers():
  """Production-dead _to_rect/_to_offset helpers are removed from the CLI."""
  assert not hasattr(cli, '_to_rect')
  assert not hasattr(cli, '_to_offset')


def test_main_maps_terminal_results_to_exit_codes(monkeypatch):
  """main maps GOAL_MET to 0, any other result to 1 and DbError to 2."""
  cfg = SimpleNamespace(db_path=':memory:')
  db = MagicMock()
  session = MagicMock()
  monkeypatch.setattr(cli, 'parse_args', lambda argv: cfg)
  monkeypatch.setattr(cli, 'InteractionLog', lambda path: db)
  monkeypatch.setattr(cli, '_build_session', lambda cfg, log: session)

  session.run.return_value = SessionStats(result='GOAL_MET')
  assert cli.main([]) == 0
  session.run.return_value = SessionStats(result='MAX_ACTIONS')
  assert cli.main([]) == 1
  assert db.close.call_count == 2

  def boom(cfg, log):
    raise DbError('setup failed')

  monkeypatch.setattr(cli, '_build_session', boom)
  assert cli.main([]) == 2


def test_build_session_wires_fixed_schedule_path(tmp_path, monkeypatch):
  """The planner is wired to the fixed SCHEDULE_PATH constant, not a flag."""
  data = _write_project(tmp_path, {'items': []})
  header = ','.join(('tick',) + tuple(config.CRAFTABLE_IDS))
  row = ','.join(['0'] + ['0'] * len(config.CRAFTABLE_IDS))
  custom = data / 'schedule.csv'
  custom.write_text('\n'.join([header, row]) + '\n', encoding='utf-8')
  monkeypatch.setattr(cli, 'SCHEDULE_PATH', str(custom), raising=False)
  cfg = _parse(tmp_path, monkeypatch)
  session = cli._build_session(cfg, object())
  assert session.planner.schedule_path == str(custom)
