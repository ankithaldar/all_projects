#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Schedule-driven crafting planning and recipe-dialog parsing."""

import csv
import os

import yaml

from cat_game_bot.config import (
    CRAFTABLE_IDS,
    SCHEDULE_PATH,
    _resolve_path,
)
from cat_game_bot.errors import ConfigError
from cat_game_bot.models import CraftStep
from cat_game_bot.targets import TargetSpec, item_template

__all__ = (
  'CraftPlanner', 'MAX_SCHEDULE_BYTES', 'batch_cost', 'dialog_ready',
  'load_schedule', 'load_tree')

BATCH_SURCHARGE = 0.25
MAX_SCHEDULE_BYTES = 8 * 1024 * 1024
_SCHEDULE_HEADER = ('tick',) + CRAFTABLE_IDS

SECTION_BY_ITEM = {
    'string': 'CRAFTING', 'wood': 'CRAFTING', 'ribbon': 'CRAFTING',
    'metal': 'SHINY CRAFTING', 'needles': 'SHINY CRAFTING',
    'sparkles': 'SHINY CRAFTING',
    'bronze': 'PRECIOUS CRAFTING', 'silver': 'PRECIOUS CRAFTING',
    'gold': 'PRECIOUS CRAFTING',
    'amethyst': 'JEWEL CRAFTING', 'orb': 'JEWEL CRAFTING',
    'pendant': 'JEWEL CRAFTING', 'necklace': 'JEWEL CRAFTING',
    'water': 'MAGIC CRAFTING', 'fire': 'MAGIC CRAFTING',
    'waterstone': 'ANCIENT CRAFTING', 'firestone': 'ANCIENT CRAFTING',
    'elementstone': 'ANCIENT CRAFTING',
    'artifact': 'RUNE CRAFTING',
}


def batch_cost(init: int, batch: int) -> int:
  """Return the in-game coin cost of a craft batch.

  Args:
    init: Coin cost of a single craft.
    batch: Number of units crafted together.

  Returns:
    The rounded cost init*b*(1+0.25*(b-1)).
  """
  return int(round(init * batch * (1 + BATCH_SURCHARGE * (batch - 1))))


def dialog_ready(text: str) -> bool:
  """Report whether a recipe dialog shows a ready craft.

  Args:
    text: OCR text from the recipe dialog status line.

  Returns:
    True when the status says the item is ready to craft.
  """
  return 'ready to craft' in text.lower()


def load_tree(path: str, allowed_root: str) -> dict[str, object]:
  """Load and schema-check a crafting tree YAML file.

  Args:
    path: Candidate path to the YAML file.
    allowed_root: Root the resolved path must remain under.

  Returns:
    The parsed tree mapping.

  Raises:
    ConfigError: If the path, YAML, or schema is invalid.
  """
  resolved = _resolve_path(path, allowed_root)
  try:
    with open(resolved, 'r', encoding='utf-8') as handle:
      data = yaml.safe_load(handle)
  except (OSError, yaml.YAMLError) as exc:
    raise ConfigError(f'invalid crafting tree: {exc}') from exc
  if not isinstance(data, dict) or not isinstance(data.get('items'), list):
    raise ConfigError('crafting tree must be an object with items list')
  return data


def _schedule_int(value: object, label: str) -> int:
  """Parse a strict, non-negative decimal integer schedule cell.

  Args:
    value: Raw cell text from the CSV reader.
    label: Column name used in the error message.

  Returns:
    The parsed non-negative integer.

  Raises:
    ConfigError: If the cell is not a plain non-negative decimal int.
  """
  if not isinstance(value, str) or not value.isascii() or not value.isdecimal():
    raise ConfigError(f'invalid schedule cell {label}: {value!r}')
  try:
    return int(value)
  except ValueError as exc:
    raise ConfigError(f'invalid schedule cell {label}: {value!r}') from exc


def load_schedule(path: str = SCHEDULE_PATH) -> list[list[tuple[str, int]]]:
  """Load and strictly validate the fixed best-schedule CSV.

  The 8 MiB size cap is enforced with ``getsize`` before the file is read.
  Rows are kept grouped: each row is the ordered list of ``(item, batch)``
  steps for its non-zero cells, in canonical craftable column order. Empty
  rows are retained so the row index stays aligned with the tick index.

  Args:
    path: Filesystem path of the trusted, fixed schedule CSV.

  Returns:
    An ordered list of grouped rows, one per sequential tick from zero.

  Raises:
    ConfigError: If the file is oversized, unreadable, or violates the
      header, sequential-tick or non-negative-integer-cell schema.
  """
  try:
    size = os.path.getsize(path)
  except OSError as exc:
    raise ConfigError(f'invalid schedule file: {exc}') from exc
  if size > MAX_SCHEDULE_BYTES:
    raise ConfigError('schedule file exceeds the 8 MiB cap')
  rows: list[list[tuple[str, int]]] = []
  expected_tick = 0
  try:
    with open(path, newline='', encoding='utf-8') as handle:
      reader = csv.DictReader(handle)
      if tuple(reader.fieldnames or ()) != _SCHEDULE_HEADER:
        raise ConfigError('schedule header mismatch')
      for raw in reader:
        if None in raw:
          raise ConfigError('schedule row has more cells than the header')
        tick = _schedule_int(raw.get('tick'), 'tick')
        if tick != expected_tick:
          raise ConfigError('schedule ticks must be sequential from zero')
        expected_tick += 1
        steps = []
        for item in CRAFTABLE_IDS:
          batch = _schedule_int(raw.get(item), item)
          if batch > 0:
            steps.append((item, batch))
        rows.append(steps)
  except (OSError, csv.Error) as exc:
    raise ConfigError(f'invalid schedule file: {exc}') from exc
  return rows


_MAX_RECIPE_COST = 10 ** 12
_MAX_COST_DIGITS = 12


def _recipe_cost(recipe: object) -> int:
  """Parse a bounded, non-negative recipe coin cost.

  Args:
    recipe: Recipe mapping from the crafting tree.

  Returns:
    The validated coin cost.

  Raises:
    ConfigError: If the cost has the wrong type, is non-decimal, or is too
      large to scale into a batch cost.
  """
  value = recipe.get('coin_cost', 0) if isinstance(recipe, dict) else 0
  if isinstance(value, bool):
    raise ConfigError('invalid recipe coin_cost')
  if isinstance(value, int):
    cost = value
  elif (isinstance(value, str) and value.isascii()
        and value.isdecimal() and len(value) <= _MAX_COST_DIGITS):
    cost = int(value)
  else:
    raise ConfigError('invalid recipe coin_cost')
  if cost < 0 or cost > _MAX_RECIPE_COST:
    raise ConfigError('invalid recipe coin_cost')
  return cost


class CraftPlanner:
  """Peeks schedule-driven craft steps; only ``mark_placed`` advances it.

  The tree, targets and inventory are retained as planning metadata: the
  tree supplies recipe costs and section names, while the fixed CSV supplies
  the exact item order and tuned batch values. Goal checks stay in the Bag
  OCR layer, so neither targets nor inventory influence step selection.

  Attributes:
    tree: Parsed crafting tree mapping (metadata only).
    targets: Goal counts (metadata only).
    inventory: Observed item counts (metadata only).
    schedule_path: Path of the validated schedule that was loaded.
  """

  def __init__(self, tree: dict[str, object], targets: TargetSpec,
               inventory: dict[str, int],
               schedule_path: str = SCHEDULE_PATH):
    self.tree = tree
    self.targets = targets
    self.inventory = inventory
    self.schedule_path = schedule_path
    self._rows = load_schedule(schedule_path)
    self._row_index = 0
    self._step_index = 0
    raw_items = tree.get('items', [])
    self._items = [item for item in raw_items if isinstance(item, dict)]
    self._by_id = {item.get('id'): item for item in self._items}

  def _filled_row_index(self) -> int:
    """Return the first non-empty row at or after the cursor.

    All-zero ticks are navigation-only and are transparently skipped without
    moving the committed cursor.

    Returns:
      Index of the next row with at least one step, or the row count when
      the schedule is exhausted.
    """
    index = self._row_index
    while index < len(self._rows) and not self._rows[index]:
      index += 1
    return index

  def next_step(self) -> CraftStep | None:
    """Peek at the current scheduled step without advancing the cursor.

    Returns:
      The next CraftStep carrying the tuned CSV batch, or None when the
      schedule is exhausted.
    """
    index = self._filled_row_index()
    if index >= len(self._rows):
      return None
    item, batch = self._rows[index][self._step_index]
    item_meta = self._by_id.get(item) or {}
    recipe = item_meta.get('recipe') or {}
    cost = batch_cost(_recipe_cost(recipe), batch)
    section = item_meta.get('section') or SECTION_BY_ITEM.get(
        item, 'CRAFTING')
    return CraftStep(item, section, item_template(item), batch, cost)

  def mark_placed(self, item: str) -> None:
    """Advance the cursor once a step's whole batch has been placed.

    This is the only method that mutates the cursor; a failed x1 tap must
    not call it. Empty rows between placed steps are skipped as part of the
    advance.

    Args:
      item: Item id of the step that was fully placed.

    Raises:
      ConfigError: If ``item`` is not the step at the cursor.
    """
    index = self._filled_row_index()
    if index >= len(self._rows):
      return
    row = self._rows[index]
    if row[self._step_index][0] != item:
      raise ConfigError(f'mark_placed item mismatch: {item}')
    self._step_index += 1
    if self._step_index >= len(row):
      self._step_index = 0
      self._row_index = index + 1
    else:
      self._row_index = index

  def dialog_ready(self, text: str) -> bool:
    """Report whether the recipe dialog is ready to craft.

    Args:
      text: OCR text from the dialog status line.

    Returns:
      True when the status says ready.
    """
    return dialog_ready(text)
