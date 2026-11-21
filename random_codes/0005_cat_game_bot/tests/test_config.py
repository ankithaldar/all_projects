#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for cat_game_bot.config parsing, bounds and path confinement."""

# This suite exercises the module's private confinement helper directly.
# pylint: disable=protected-access

import dataclasses
import json
import os
import subprocess
import sys

import pytest

import cat_game_bot.config as config
import cat_game_bot.db as db_module
import cat_game_bot.session as session_module
from cat_game_bot.errors import ConfigError
from cat_game_bot.inventory import CANONICAL_IDS
from cat_game_bot.models import GameState, Rect


def make_config(monkeypatch, tmp_path, extra=None):
  """Parse args with the project root redirected to a temp directory.

  Args:
    monkeypatch: Pytest monkeypatch fixture.
    tmp_path: Temporary root directory.
    extra: Additional CLI flags.

  Returns:
    The parsed frozen Config.
  """
  monkeypatch.setattr(config, 'PROJECT_ROOT', tmp_path)
  return config.parse_args(['--serial', 'dev-1', *(extra or [])])


def test_defaults_and_frozen_config(monkeypatch, tmp_path):
  """Defaults match the config table and the dataclass is frozen."""
  cfg = make_config(monkeypatch, tmp_path)
  assert cfg.serial == 'dev-1'
  assert cfg.package == 'com.minogames.cats.client'
  assert cfg.duration_seconds == 300.0
  assert cfg.max_actions == 500
  assert cfg.bag_visit_interval == 10
  assert cfg.threshold == 0.8
  assert cfg.poll_delay == 1.0
  assert cfg.settle_timeout == 12.0
  assert cfg.no_progress_limit == 15
  assert cfg.error_limit == 5
  assert cfg.ocr_lang == 'eng'
  assert cfg.dry_run is True
  assert cfg.template_dir == tmp_path / 'templates'
  assert cfg.log_dir == tmp_path / 'logs'
  assert cfg.tesseract_cmd == tmp_path / '.tools' / 'bin' / 'tesseract'
  assert cfg.targets_path == tmp_path / 'data' / 'initial-state.json'
  assert cfg.tree_path == tmp_path / 'data' / 'crafting_tree.yaml'
  assert cfg.ocr_regions_path == tmp_path / 'data' / 'ocr-regions.json'
  assert cfg.db_path == tmp_path / 'logs' / 'bot.sqlite3'
  with pytest.raises(dataclasses.FrozenInstanceError):
    cfg.serial = 'other'


def test_live_polarity(monkeypatch, tmp_path):
  """--live sets dry_run False; omitting it keeps dry_run True."""
  assert make_config(monkeypatch, tmp_path, ['--live']).dry_run is False
  assert make_config(monkeypatch, tmp_path).dry_run is True


def test_help_works_and_focus_flag_absent(capsys):
  """--help succeeds and an unknown --focus flag is rejected."""
  with pytest.raises(SystemExit) as exc:
    config.parse_args(['--help'])
  assert exc.value.code == 0
  assert '--live' in capsys.readouterr().out
  with pytest.raises(SystemExit) as exc:
    config.parse_args(['--serial', 'dev-1', '--focus'])
  assert exc.value.code == 2


@pytest.mark.parametrize('flag,value', [
  ('--duration', '0'), ('--duration', '3601'),
  ('--max-actions', '0'), ('--max-actions', '10001'),
  ('--bag-visit-interval', '0'), ('--bag-visit-interval', '101'),
  ('--threshold', '0.4'), ('--threshold', '1.1'),
  ('--poll-delay', '0.1'), ('--poll-delay', '5.1'),
  ('--settle-timeout', '1'), ('--settle-timeout', '61'),
  ('--no-progress-limit', '0'), ('--no-progress-limit', '101'),
  ('--error-limit', '0'), ('--error-limit', '21'),
])
def test_numeric_bounds(monkeypatch, tmp_path, flag, value):
  """Out-of-range numeric values raise ConfigError."""
  with pytest.raises(ConfigError):
    make_config(monkeypatch, tmp_path, [flag, value])


def test_serial_and_package_validation(monkeypatch, tmp_path):
  """Hostile serial and package values raise ConfigError."""
  monkeypatch.setattr(config, 'PROJECT_ROOT', tmp_path)
  with pytest.raises(ConfigError):
    config.parse_args(['--serial', 'a;b'])
  with pytest.raises(ConfigError):
    config.parse_args(['--serial', 'dev-1', '--package', 'com;rm'])


@pytest.mark.parametrize('flag', [
  '--templates', '--log-dir', '--tesseract-cmd', '--targets',
  '--tree', '--ocr-regions', '--db',
])
def test_path_traversal_rejected(monkeypatch, tmp_path, flag):
  """Traversal outside the allowed root raises ConfigError for every path."""
  outside = str(tmp_path.parent / 'escape')
  with pytest.raises(ConfigError):
    make_config(monkeypatch, tmp_path, [flag, outside])


@pytest.mark.parametrize('flag', [
  '--templates', '--log-dir', '--tesseract-cmd', '--targets',
  '--tree', '--ocr-regions', '--db',
])
def test_absolute_path_outside_root_rejected(monkeypatch, tmp_path, flag):
  """An absolute path outside the allowed root raises ConfigError."""
  with pytest.raises(ConfigError):
    make_config(monkeypatch, tmp_path, [flag, '/etc/passwd'])


def test_resolve_path_rejects_symlink(tmp_path):
  """_resolve_path rejects a symlink outright."""
  target = tmp_path / 'real'
  target.mkdir()
  link = tmp_path / 'link'
  link.symlink_to(target)
  with pytest.raises(ConfigError):
    config._resolve_path(str(link), str(tmp_path))


def test_target_symlink_rejected_by_parse_args(monkeypatch, tmp_path):
  """A symlinked --targets/--tree path is rejected by parse_args."""
  data_dir = tmp_path / 'data'
  data_dir.mkdir()
  link = data_dir / 'linked.json'
  link.symlink_to(tmp_path / 'elsewhere.json')
  with pytest.raises(ConfigError):
    make_config(monkeypatch, tmp_path, ['--targets', str(link)])
  with pytest.raises(ConfigError):
    make_config(monkeypatch, tmp_path, ['--tree', str(link)])


def test_log_dir_is_created(monkeypatch, tmp_path):
  """parse_args creates the configured log directory."""
  cfg = make_config(monkeypatch, tmp_path, ['--log-dir',
                                           str(tmp_path / 'logs')])
  assert cfg.log_dir.is_dir()


def test_log_dir_creation_failure_is_config_error(monkeypatch, tmp_path):
  """An uncreatable log directory raises ConfigError."""
  blocker = tmp_path / 'logs'
  blocker.write_text('not a directory', encoding='utf-8')
  with pytest.raises(ConfigError):
    make_config(monkeypatch, tmp_path)


def test_db_path_must_stay_under_logs(monkeypatch, tmp_path):
  """A --db path outside the logs root is rejected."""
  with pytest.raises(ConfigError):
    make_config(monkeypatch, tmp_path,
                ['--db', str(tmp_path / 'data' / 'evil.sqlite3')])


def test_resolve_path_rejects_traversal_under_python_o(tmp_path):
  """Confinement uses explicit raises, so it also holds under python -O."""
  project_root = str(config.PROJECT_ROOT)
  script = (
    'import sys;'
    'sys.path.insert(0, {project!r});'
    'from cat_game_bot.config import _resolve_path;'
    'from cat_game_bot.errors import ConfigError;'
    'p = {root!r} + "/../escape";'
    '_resolve_path(p, {root!r})'
  ).format(project=project_root, root=str(tmp_path))
  result = subprocess.run(
    [sys.executable, '-O', '-c', script], cwd=str(tmp_path),
    capture_output=True, text=True, check=False)
  assert result.returncode != 0
  assert 'ConfigError' in result.stderr


def _valid_regions_doc():
  """Return a fresh schema-valid ocr-regions document."""
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


def _write_regions(tmp_path, data):
  """Write a regions document to disk and return its path.

  Args:
    tmp_path: Temporary directory provided by pytest.
    data: Document to serialise as JSON.

  Returns:
    str: Absolute path of the written file.
  """
  path = tmp_path / 'ocr-regions.json'
  path.write_text(json.dumps(data), encoding='utf-8')
  return str(path)


def _bad_version(data):
  data['version'] = 99


def _bool_version(data):
  data['version'] = True


def _extra_top_key(data):
  data['extra'] = 1


def _extra_region(data):
  data['regions']['extra'] = {'x': 0, 'y': 0, 'w': 1, 'h': 1}


def _missing_region(data):
  del data['regions']['gift_timer']


def _bool_region_coord(data):
  data['regions']['coins']['x'] = True


def _negative_region_coord(data):
  data['regions']['coins']['x'] = -1


def _zero_region_width(data):
  data['regions']['coins']['w'] = 0


def _negative_region_height(data):
  data['regions']['coins']['h'] = -5


def _region_past_width(data):
  data['regions']['coins']['x'] = 1000


def _region_past_height(data):
  data['regions']['gift_timer']['y'] = 2400


def _region_area_too_large(data):
  data['regions']['coins'].update(w=2000, h=2000)


def _extra_region_key(data):
  data['regions']['coins']['extra'] = 1


def _missing_region_key(data):
  del data['regions']['coins']['h']


def _unknown_offset(data):
  data['offsets']['item_count.evil'] = {'dx': 0, 'dy': 0, 'w': 1, 'h': 1}


def _missing_offset(data):
  del data['offsets']['item_count.artifact']


def _bool_offset_value(data):
  data['offsets']['item_count.cotton']['dx'] = True


def _negative_offset(data):
  data['offsets']['item_count.cotton']['dy'] = -1


def _offset_width_too_large(data):
  data['offsets']['item_count.cotton']['w'] = 201


def _offset_height_too_large(data):
  data['offsets']['item_count.cotton']['h'] = 101


def _zero_offset_width(data):
  data['offsets']['item_count.cotton']['w'] = 0


def _zero_offset_height(data):
  data['offsets']['item_count.cotton']['h'] = 0


def _extra_offset_key(data):
  data['offsets']['item_count.cotton']['z'] = 1


def _missing_offset_key(data):
  del data['offsets']['item_count.cotton']['h']


_MUTATORS = (
  _bad_version, _bool_version, _extra_top_key, _extra_region,
  _missing_region, _bool_region_coord, _negative_region_coord,
  _zero_region_width, _negative_region_height, _region_past_width,
  _region_past_height, _region_area_too_large, _extra_region_key,
  _missing_region_key, _unknown_offset, _missing_offset,
  _bool_offset_value, _negative_offset, _offset_width_too_large,
  _offset_height_too_large, _zero_offset_width, _zero_offset_height,
  _extra_offset_key, _missing_offset_key,
)


@pytest.mark.parametrize('mutator', _MUTATORS,
                         ids=[fn.__name__ for fn in _MUTATORS])
def test_load_regions_rejects_invalid_document(tmp_path, mutator):
  """Every schema violation raises ConfigError, never a silent coercion."""
  data = _valid_regions_doc()
  mutator(data)
  with pytest.raises(ConfigError):
    config.load_regions(_write_regions(tmp_path, data))


def test_load_regions_returns_valid_regions_and_offsets(tmp_path):
  """A valid document yields Rect regions and four-integer offsets."""
  regions, offsets = config.load_regions(
      _write_regions(tmp_path, _valid_regions_doc()))
  assert set(regions) == {'coins', 'gift_timer'}
  assert regions['coins'] == Rect(505, 150, 240, 70)
  assert regions['gift_timer'] == Rect(940, 620, 135, 70)
  assert set(offsets) == set(config.OFFSET_NAMES)
  assert offsets['item_count.cotton'] == (0, 140, 120, 64)


def test_offset_names_match_inventory_canonical_ids():
  """Validated offset names match the inventory reader's canonical ids."""
  expected = frozenset(f'item_count.{item}' for item in CANONICAL_IDS)
  assert config.OFFSET_NAMES == expected


def test_load_regions_validation_holds_under_python_o(tmp_path):
  """Region checks are explicit raises, so they survive python -O."""
  bad = _valid_regions_doc()
  bad['version'] = 42
  path = _write_regions(tmp_path, bad)
  project = str(config.PROJECT_ROOT)
  script = (
    'import sys;'
    f'sys.path.insert(0, {project!r});'
    'from cat_game_bot.config import load_regions;'
    f'load_regions({path!r})'
  )
  result = subprocess.run(
    [sys.executable, '-O', '-c', script],
    capture_output=True, text=True, check=False)
  assert result.returncode != 0
  assert 'ConfigError' in result.stderr


def test_log_dir_is_forced_0700_on_existing_dir(monkeypatch, tmp_path):
  """parse_args forces mode 0700 even when the log dir already exists."""
  logs = tmp_path / 'logs'
  logs.mkdir()
  os.chmod(logs, 0o777)
  cfg = make_config(monkeypatch, tmp_path)
  assert cfg.log_dir == logs
  assert os.stat(logs).st_mode & 0o777 == 0o700


_EXPECTED_SCHEDULE_PATH = (
  '/run/media/helder/Code_Dump/all_projects_all_branches/'
  'crazy_ideas_main/random_codes/0004_cat_game_neat/data/'
  'best-schedule.csv')


def test_json_default_is_one_canonical_shared_helper():
  """session and db reuse the canonical _json_default helper from config."""
  assert session_module._json_default is config._json_default
  assert db_module._json_default is config._json_default
  assert config._json_default(GameState.HOME) == 'HOME'
  assert config._json_default(Rect(1, 2, 3, 4)) == {
      'x': 1, 'y': 2, 'width': 3, 'height': 4}
  assert not hasattr(config, 'json_default')


def test_schedule_path_is_exact_fixed_constant():
  """SCHEDULE_PATH is the exact user-mandated absolute CSV location."""
  assert config.SCHEDULE_PATH == _EXPECTED_SCHEDULE_PATH


def test_schedule_has_no_cli_flag(monkeypatch, tmp_path):
  """The trusted fixed schedule is not configurable via the command line."""
  with pytest.raises(SystemExit) as exc:
    make_config(monkeypatch, tmp_path, ['--schedule', '/tmp/x.csv'])
  assert exc.value.code == 2
  with pytest.raises(SystemExit) as exc:
    make_config(monkeypatch, tmp_path, ['--schedule-path', '/tmp/x.csv'])
  assert exc.value.code == 2
