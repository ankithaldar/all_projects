#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for the fixed best-schedule CSV loader and the planner cursor."""

# The loader and the planner cursor are the schedule-specific surface.
# pylint: disable=protected-access

import pytest

from cat_game_bot import crafting
from cat_game_bot.config import CRAFTABLE_IDS, SCHEDULE_PATH
from cat_game_bot.crafting import (
    MAX_SCHEDULE_BYTES,
    SECTION_BY_ITEM,
    CraftPlanner,
    load_schedule,
)
from cat_game_bot.errors import ConfigError
from cat_game_bot.models import CraftStep
from cat_game_bot.targets import TargetSpec, item_template

_HEADER = ('tick',) + CRAFTABLE_IDS


def _row(tick, cells=None):
  """Build one CSV line for ``tick`` from a mapping of non-zero cells."""
  cells = cells or {}
  return ','.join([str(tick)] +
                  [str(cells.get(item, 0)) for item in CRAFTABLE_IDS])


def _write(tmp_path, text):
  """Write raw schedule text to a file and return its path."""
  path = tmp_path / 'schedule.csv'
  path.write_text(text, encoding='utf-8')
  return str(path)


def _valid_schedule(rows):
  """Build a valid CSV document from ``(tick, cells)`` pairs."""
  return '\n'.join([','.join(_HEADER)] + [_row(t, c) for t, c in rows]) + '\n'


def _spec():
  """Return an empty goal specification; the schedule drives step choice."""
  return TargetSpec({}, 0, {})


def _planner(tmp_path, rows):
  """Build a planner over a synthetic schedule and a minimal tree."""
  path = _write(tmp_path, _valid_schedule(rows))
  tree = {'items': [
      {'id': 'string', 'type': 'craftable', 'recipe': {'coin_cost': 50}},
      {'id': 'wood', 'type': 'craftable', 'recipe': {'coin_cost': 50}},
  ]}
  return CraftPlanner(tree, _spec(), {}, schedule_path=path)


def test_no_dead_max_batch_constant():
  """The unused MAX_BATCH constant is removed from the planner."""
  assert not hasattr(crafting, 'MAX_BATCH')


def test_every_craftable_has_a_plan_section():
  """Each craftable, including necklace, has a section label."""
  assert SECTION_BY_ITEM['necklace'] == 'JEWEL CRAFTING'
  assert frozenset(CRAFTABLE_IDS) <= frozenset(SECTION_BY_ITEM)


def test_load_schedule_groups_nonzero_cells_in_column_order(tmp_path):
  """Each row is an ordered list of (item, batch) steps from non-zero cells."""
  text = _valid_schedule([
      (0, {'string': 2, 'wood': 4}),
      (1, {'string': 3}),
  ])
  rows = load_schedule(_write(tmp_path, text))
  assert rows == [
      [('string', 2), ('wood', 4)],
      [('string', 3)],
  ]


def test_load_schedule_keeps_empty_rows(tmp_path):
  """An all-zero tick is retained as an empty grouped row."""
  rows = load_schedule(_write(tmp_path, _valid_schedule([
      (0, {'string': 1}),
      (1, {}),
      (2, {'wood': 2}),
  ])))
  assert rows == [[('string', 1)], [], [('wood', 2)]]


def test_load_schedule_rejects_wrong_header(tmp_path):
  """A header that is not tick plus the 19 canonical ids is rejected."""
  bad_header = ('tick',) + CRAFTABLE_IDS[:-1] + ('evil',)
  text = '\n'.join([','.join(bad_header), _row(0)]) + '\n'
  with pytest.raises(ConfigError):
    load_schedule(_write(tmp_path, text))


def test_load_schedule_rejects_missing_header_column(tmp_path):
  """A header missing a craftable column is rejected."""
  header = ('tick',) + CRAFTABLE_IDS[1:]
  text = '\n'.join([','.join(header), _row(0)]) + '\n'
  with pytest.raises(ConfigError):
    load_schedule(_write(tmp_path, text))


def test_load_schedule_rejects_duplicate_tick(tmp_path):
  """A repeated tick is rejected."""
  text = _valid_schedule([(0, {}), (0, {})])
  with pytest.raises(ConfigError):
    load_schedule(_write(tmp_path, text))


def test_load_schedule_rejects_out_of_order_tick(tmp_path):
  """A tick greater than the previous tick plus one is rejected."""
  text = _valid_schedule([(0, {}), (2, {})])
  with pytest.raises(ConfigError):
    load_schedule(_write(tmp_path, text))


def test_load_schedule_rejects_tick_not_starting_at_zero(tmp_path):
  """The first tick must be zero."""
  text = _valid_schedule([(1, {})])
  with pytest.raises(ConfigError):
    load_schedule(_write(tmp_path, text))


def test_load_schedule_rejects_negative_cell(tmp_path):
  """A negative batch cell is rejected."""
  text = _valid_schedule([(0, {'string': -1})])
  with pytest.raises(ConfigError):
    load_schedule(_write(tmp_path, text))


@pytest.mark.parametrize('bad_value', ['x', '1.5', ' 2', '2 ', '+2', ''])
def test_load_schedule_rejects_non_integer_cell(tmp_path, bad_value):
  """Any cell that is not a plain non-negative decimal int is rejected."""
  header = ','.join(_HEADER)
  values = ['0'] + ['0'] * len(CRAFTABLE_IDS)
  values[CRAFTABLE_IDS.index('string') + 1] = bad_value
  text = '\n'.join([header, ','.join(values)]) + '\n'
  with pytest.raises(ConfigError):
    load_schedule(_write(tmp_path, text))


def test_load_schedule_rejects_extra_cell(tmp_path):
  """A row carrying more cells than the fixed header is rejected."""
  header = ','.join(_HEADER)
  values = ['0'] + ['0'] * len(CRAFTABLE_IDS) + ['99']
  text = '\n'.join([header, ','.join(values)]) + '\n'
  with pytest.raises(ConfigError):
    load_schedule(_write(tmp_path, text))


def test_load_schedule_rejects_oversized_integer_cell(tmp_path):
  """A cell with too many digits maps to ConfigError, not ValueError."""
  header = ','.join(_HEADER)
  values = ['0'] + ['0'] * len(CRAFTABLE_IDS)
  values[CRAFTABLE_IDS.index('string') + 1] = '9' * 5000
  text = '\n'.join([header, ','.join(values)]) + '\n'
  with pytest.raises(ConfigError):
    load_schedule(_write(tmp_path, text))


def test_load_schedule_rejects_oversized_file(tmp_path):
  """A file larger than the 8 MiB cap is refused before it is materialised."""
  path = tmp_path / 'huge.csv'
  with open(path, 'wb') as handle:
    handle.truncate(MAX_SCHEDULE_BYTES + 1)
  with pytest.raises(ConfigError):
    load_schedule(str(path))


def test_next_step_peeks_without_advancing(tmp_path):
  """Repeated peeks return the same step; only mark_placed moves the cursor."""
  planner = _planner(tmp_path, [
      (0, {'string': 2, 'wood': 4}),
      (1, {'string': 3}),
  ])
  peeks = [planner.next_step() for _ in range(5)]
  assert all(step == peeks[0] for step in peeks)
  assert peeks[0].item == 'string'
  assert peeks[0].batch == 2


def test_step_carries_tuned_batch_and_tree_metadata(tmp_path):
  """A step carries the CSV batch plus tree-derived section/template/cost."""
  planner = _planner(tmp_path, [(0, {'wood': 4})])
  step = planner.next_step()
  assert isinstance(step, CraftStep)
  assert step == CraftStep('wood', 'CRAFTING', 'item_wood', 4, 350)


def test_mark_placed_advances_within_a_row(tmp_path):
  """Placing the first step exposes the next step of the same row."""
  planner = _planner(tmp_path, [(0, {'string': 2, 'wood': 4})])
  assert planner.next_step().item == 'string'
  planner.mark_placed('string')
  step = planner.next_step()
  assert step.item == 'wood'
  assert step.batch == 4


def test_mark_placed_advances_across_rows(tmp_path):
  """Placing every step of a row exposes the first step of the next row."""
  planner = _planner(tmp_path, [
      (0, {'string': 2, 'wood': 4}),
      (1, {'string': 3}),
  ])
  planner.mark_placed('string')
  planner.mark_placed('wood')
  step = planner.next_step()
  assert step.item == 'string'
  assert step.batch == 3


def test_mark_placed_rejects_a_wrong_item(tmp_path):
  """mark_placed only accepts the item currently under the cursor."""
  planner = _planner(tmp_path, [(0, {'string': 2})])
  with pytest.raises(ConfigError):
    planner.mark_placed('wood')


def test_next_step_skips_empty_rows(tmp_path):
  """All-zero ticks are transparently skipped while peeking."""
  planner = _planner(tmp_path, [
      (0, {}),
      (1, {'wood': 2}),
  ])
  assert planner.next_step().item == 'wood'
  planner.mark_placed('wood')
  assert planner.next_step() is None


def test_mark_placed_skips_consecutive_empty_rows(tmp_path):
  """Advancing past a step lands on the next non-empty row."""
  planner = _planner(tmp_path, [
      (0, {}),
      (1, {'string': 2}),
      (2, {}),
      (3, {}),
      (4, {'wood': 3}),
  ])
  assert planner.next_step().item == 'string'
  planner.mark_placed('string')
  assert planner.next_step().item == 'wood'
  planner.mark_placed('wood')
  assert planner.next_step() is None


def test_next_step_returns_none_when_exhausted(tmp_path):
  """After the final step is placed no further step is available."""
  planner = _planner(tmp_path, [(0, {'string': 1})])
  planner.mark_placed('string')
  assert planner.next_step() is None
  assert planner.next_step() is None


def test_real_schedule_loads_with_expected_shape_and_anchor_rows():
  """The fixed CSV at SCHEDULE_PATH loads with the F19 structure.

  Plan section 12 records that the 0004 data may drift, so the structural
  invariants and the first-three-row anchor are asserted against the file
  as it exists on disk: ticks 0..3348 (3349 rows), 19 item columns (checked
  by the loader header), 2338 non-empty rows, and three single-artifact
  crafts (three rows of batch 1, totalling the plan's 'artifact x3').
  """
  rows = load_schedule(SCHEDULE_PATH)
  non_empty = [row for row in rows if row]
  assert len(rows) == 3349
  assert len(non_empty) == 2338
  assert non_empty[0] == [
      ('string', 2), ('wood', 4), ('metal', 4), ('bronze', 2),
      ('amethyst', 4), ('orb', 1)]
  assert non_empty[1] == [('string', 2), ('wood', 4), ('ribbon', 1)]
  assert non_empty[2] == [('string', 2), ('wood', 4)]
  artifacts = [batch for row in rows for item, batch in row
               if item == 'artifact']
  assert artifacts == [1, 1, 1]


def test_real_schedule_step_templates_exist(manifest_names):
  """Every item named by the real schedule maps to a real manifest icon."""
  rows = load_schedule(SCHEDULE_PATH)
  used_items = {item for row in rows for item, _ in row}
  assert used_items
  assert {item_template(item) for item in used_items} <= manifest_names
