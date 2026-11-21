#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for the target-spec loading and goal-checking logic."""

import json

import pytest

from cat_game_bot.errors import ConfigError
from cat_game_bot.targets import (
    TARGET_IDS,
    TargetSpec,
    goal_met,
    item_template,
    load_targets,
    verify_initial_state,
)


def _write_state(tmp_path, **overrides):
  """Write a synthetic initial-state.json and return its path."""
  data = {
      'targets': {
          'silver': 12, 'gold': 13, 'pendant': 1, 'necklace': 15,
          'orb': 20, 'water': 5, 'fire': 4, 'waterstone': 6,
          'firestone': 5, 'elementstone': 10, 'artifact': 3,
      },
      'initialCoins': 6423826,
      'initialStash': {
          'string': 86, 'wood': 69, 'ribbon': 59, 'metal': 18,
          'needles': 33, 'sparkles': 2, 'bronze': 30, 'silver': 2,
          'gold': 0, 'amethyst': 163, 'pendant': 6, 'necklace': 0,
          'orb': 44, 'water': 0, 'fire': 1, 'waterstone': 1,
          'firestone': 3, 'elementstone': 1, 'artifact': 0,
      },
  }
  data.update(overrides)
  path = tmp_path / 'initial-state.json'
  path.write_text(json.dumps(data), encoding='utf-8')
  return path


def test_target_ids_are_the_eleven_canonical_goals():
  """TARGET_IDS is exactly the 11 allowlisted goal ids in tree order."""
  assert TARGET_IDS == (
      'silver', 'gold', 'pendant', 'necklace', 'orb', 'water', 'fire',
      'waterstone', 'firestone', 'elementstone', 'artifact')


def test_item_template_returns_canonical_manifest_name():
  """item_template maps a bare id to its real manifest template name."""
  assert item_template('silver') == 'item_silver'
  assert item_template('artifact') == 'item_artifact'


def test_every_target_id_has_a_canonical_template(manifest_names):
  """All 11 canonical target ids resolve to real manifest templates."""
  assert frozenset(item_template(item_id) for item_id in TARGET_IDS) \
      <= manifest_names


def test_load_targets_reads_goals_coins_and_stash(tmp_path):
  """load_targets maps the JSON schema into a frozen TargetSpec."""
  spec = load_targets(str(_write_state(tmp_path)))
  assert isinstance(spec, TargetSpec)
  assert spec.items['silver'] == 12
  assert spec.items['artifact'] == 3
  assert len(spec.items) == 11
  assert spec.initial_coins == 6423826
  assert spec.initial_stash['string'] == 86
  assert len(spec.initial_stash) == 19


def test_load_targets_ignores_extra_target_keys(tmp_path):
  """Allowlist behaviour: unknown target ids are not promoted to goals."""
  data = json.loads(_write_state(tmp_path).read_text(encoding='utf-8'))
  data['targets']['bogus'] = 999
  path = tmp_path / 'state.json'
  path.write_text(json.dumps(data), encoding='utf-8')
  assert 'bogus' not in load_targets(str(path)).items


def test_load_targets_maps_open_failure_to_config_error(tmp_path):
  """A missing or unreadable file raises ConfigError, not raw OSError."""
  with pytest.raises(ConfigError):
    load_targets(str(tmp_path / 'missing.json'))


def test_load_targets_rejects_missing_targets_key(tmp_path):
  """A malformed file without 'targets' is rejected."""
  path = tmp_path / 'bad.json'
  path.write_text('{"initialCoins": 1, "initialStash": {}}',
                  encoding='utf-8')
  with pytest.raises(ValueError):
    load_targets(str(path))


def test_load_targets_rejects_bool_counts(tmp_path):
  """Booleans are not valid integer counts."""
  data = json.loads(_write_state(tmp_path).read_text(encoding='utf-8'))
  data['targets']['silver'] = True
  path = tmp_path / 'bool.json'
  path.write_text(json.dumps(data), encoding='utf-8')
  with pytest.raises(ValueError):
    load_targets(str(path))


def test_goal_met_true_when_all_counts_reached():
  """goal_met returns True only when every goal count is met."""
  spec = TargetSpec({'silver': 2, 'gold': 1}, 0, {})
  assert goal_met({'silver': 2, 'gold': 1}, spec)


def test_goal_met_false_when_one_goal_short():
  """goal_met returns False when a single goal is short."""
  spec = TargetSpec({'silver': 2, 'gold': 1}, 0, {})
  assert not goal_met({'silver': 2, 'gold': 0}, spec)


def test_goal_met_false_when_goal_missing_from_counts():
  """A missing count is treated as zero, so the goal is unmet."""
  spec = TargetSpec({'silver': 2}, 0, {})
  assert not goal_met({}, spec)


def test_verify_initial_state_empty_when_all_match():
  """No mismatches are reported when observed equals the initial stash."""
  spec = TargetSpec({}, 0, {'string': 86, 'wood': 69})
  assert not verify_initial_state({'string': 86, 'wood': 69}, spec)


def test_verify_initial_state_reports_mismatches():
  """Each drifted item maps to (expected, observed)."""
  spec = TargetSpec({}, 0, {'string': 86, 'wood': 69})
  result = verify_initial_state({'string': 80, 'wood': 69}, spec)
  assert result == {'string': (86, 80)}
