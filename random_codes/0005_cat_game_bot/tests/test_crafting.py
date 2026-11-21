#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for crafting metadata helpers and recipe-dialog parsing.

Schedule-driven step selection and the peek/mark_placed cursor are covered
in ``tests/test_schedule.py``; this module keeps the tree, cost and dialog
helpers that the planner still uses for metadata.
"""

import pytest
import yaml

from cat_game_bot.config import CRAFTABLE_IDS
from cat_game_bot.crafting import (
    CraftPlanner,
    batch_cost,
    dialog_ready,
    load_tree,
)
from cat_game_bot.errors import ConfigError
from cat_game_bot.targets import TargetSpec

_HEADER = ('tick',) + CRAFTABLE_IDS


def _tree():
  """Return a small synthetic crafting tree with four craft tiers."""
  return {'items': [
      {'id': 'cotton', 'type': 'base', 'tier': 0},
      {'id': 'tree', 'type': 'base', 'tier': 0},
      {'id': 'rock', 'type': 'base', 'tier': 0},
      {'id': 'string', 'type': 'craftable', 'tier': 1, 'recipe': {
          'ingredients': [{'item': 'cotton', 'quantity': 3}],
          'coin_cost': 50, 'craft_time': 5}},
      {'id': 'wood', 'type': 'craftable', 'tier': 1, 'recipe': {
          'ingredients': [{'item': 'tree', 'quantity': 3}],
          'coin_cost': 50, 'craft_time': 5}},
      {'id': 'bronze', 'type': 'craftable', 'tier': 2, 'recipe': {
          'ingredients': [{'item': 'rock', 'quantity': 4}],
          'coin_cost': 200, 'craft_time': 30}},
      {'id': 'silver', 'type': 'craftable', 'tier': 3, 'recipe': {
          'ingredients': [{'item': 'bronze', 'quantity': 3}],
          'coin_cost': 300, 'craft_time': 120}},
      {'id': 'gold', 'type': 'craftable', 'tier': 4, 'recipe': {
          'ingredients': [{'item': 'silver', 'quantity': 2}],
          'coin_cost': 500, 'craft_time': 360}},
  ]}


def _spec(**items):
  return TargetSpec(items, 0, {})


def _planner(tmp_path, tree, inventory):
  """Build a planner whose schedule has a single benign step."""
  values = ['0'] + ['0'] * len(CRAFTABLE_IDS)
  path = tmp_path / 'schedule.csv'
  path.write_text('\n'.join([','.join(_HEADER), ','.join(values)]) + '\n',
                  encoding='utf-8')
  return CraftPlanner(tree, _spec(), inventory, schedule_path=str(path))


def _planner_with_cost(tmp_path, coin_cost):
  """Build a planner whose sole step uses the supplied recipe coin cost."""
  values = ['0'] + [
    '1' if item == 'string' else '0' for item in CRAFTABLE_IDS]
  path = tmp_path / 'schedule.csv'
  path.write_text('\n'.join([','.join(_HEADER), ','.join(values)]) + '\n',
                  encoding='utf-8')
  tree = {'items': [
    {'id': 'string', 'type': 'craftable', 'recipe': {
      'coin_cost': coin_cost}},
  ]}
  return CraftPlanner(tree, _spec(), {}, schedule_path=str(path))


@pytest.mark.parametrize(
  'coin_cost', ['9' * 5000, 10 ** 5000, 'abc', True, -1],
  ids=['long-digits', 'huge-int', 'text', 'bool', 'negative'])
def test_next_step_rejects_invalid_recipe_coin_cost(tmp_path, coin_cost):
  """Bad or oversized recipe costs raise ConfigError, not ValueError."""
  planner = _planner_with_cost(tmp_path, coin_cost)
  with pytest.raises(ConfigError):
    planner.next_step()


def test_next_step_accepts_decimal_string_coin_cost(tmp_path):
  """A plain decimal-string coin cost is parsed into the step cost."""
  step = _planner_with_cost(tmp_path, '50').next_step()
  assert step is not None
  assert step.cost == 50


def test_batch_cost_formula():
  """Batch cost is init*b*(1+0.25*(b-1))."""
  assert batch_cost(300, 1) == 300
  assert batch_cost(50, 5) == 500
  assert batch_cost(300, 5) == 3000


def test_planner_retains_metadata_for_cost_and_section(tmp_path):
  """The planner keeps tree/targets/inventory and reads cost from the tree."""
  planner = _planner(tmp_path, _tree(), {'cotton': 10})
  assert planner.tree is not None
  assert planner.targets.items == {}
  assert planner.inventory == {'cotton': 10}
  assert planner.dialog_ready('Ready to Craft')


def test_dialog_ready_true_for_ready_text():
  """'Ready to Craft' marks the dialog actionable."""
  assert dialog_ready('Ready to Craft')


def test_dialog_ready_false_for_crafting_text():
  """A crafting countdown marks the dialog as busy."""
  assert not dialog_ready('Crafting.. 04:57')


def test_dialog_ready_is_case_insensitive():
  """Dialog status text is matched case-insensitively."""
  assert dialog_ready('ready to craft')


def test_planner_dialog_ready_method(tmp_path):
  """CraftPlanner exposes dialog_ready as an instance method too."""
  planner = _planner(tmp_path, _tree(), {})
  assert planner.dialog_ready('Ready to Craft')
  assert not planner.dialog_ready('Crafting.. 00:30')


def test_load_tree_parses_yaml(tmp_path):
  """load_tree returns the parsed schema from a YAML file."""
  path = tmp_path / 'tree.yaml'
  path.write_text(yaml.safe_dump(_tree()), encoding='utf-8')
  loaded = load_tree(str(path), str(tmp_path))
  assert loaded['items'][0]['id'] == 'cotton'


def test_load_tree_rejects_unsafe_tag(tmp_path):
  """Unsafe YAML tags are refused by safe_load."""
  path = tmp_path / 'evil.yaml'
  path.write_text('!!python/object/apply:os.system [\'echo hi\']',
                  encoding='utf-8')
  try:
    load_tree(str(path), str(tmp_path))
  except (ValueError, yaml.YAMLError):
    return
  raise AssertionError('unsafe YAML tag was accepted')


def test_load_tree_rejects_missing_items(tmp_path):
  """A mapping without 'items' fails schema validation."""
  path = tmp_path / 'bad.yaml'
  path.write_text(yaml.safe_dump({'version': 1}), encoding='utf-8')
  try:
    load_tree(str(path), str(tmp_path))
  except ValueError:
    return
  raise AssertionError('tree without items was accepted')


def test_load_tree_confines_path(tmp_path):
  """A path outside the allowed root is rejected by _resolve_path."""
  try:
    load_tree('/etc/passwd', str(tmp_path))
  except ValueError:
    return
  raise AssertionError('out-of-root path was accepted')
