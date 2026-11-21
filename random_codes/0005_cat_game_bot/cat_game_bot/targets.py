#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Goal specification loaded from the initial-state data file."""

import json
from dataclasses import dataclass

from cat_game_bot.errors import ConfigError

__all__ = (
    'TARGET_IDS', 'TargetSpec', 'load_targets', 'goal_met',
    'verify_initial_state', 'item_template')

TARGET_IDS = (
    'silver', 'gold', 'pendant', 'necklace', 'orb', 'water', 'fire',
    'waterstone', 'firestone', 'elementstone', 'artifact')


def item_template(item_id: str) -> str:
  """Return the single canonical manifest template name for an item id.

  Args:
    item_id: Bare canonical item id (for example ``'silver'``).

  Returns:
    The manifest match name (for example ``'item_silver'``).
  """
  return f'item_{item_id}'


@dataclass(frozen=True)
class TargetSpec:
  """The 11 goal counts plus the observed initial economy.

  Attributes:
    items: Mapping of goal item id to required Bag count.
    initial_coins: Coin balance observed at reconnaissance time.
    initial_stash: Mapping of craftable id to expected starting count.
  """

  items: dict[str, int]
  initial_coins: int
  initial_stash: dict[str, int]


def _is_count(value: object) -> bool:
  """Return True for a non-negative int that is not a bool."""
  return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def load_targets(path: str) -> TargetSpec:
  """Load and validate a TargetSpec from a JSON file.

  Args:
    path: Filesystem path to initial-state.json.

  Returns:
    The validated target specification.

  Raises:
    ConfigError: If the file or its schema is invalid.
  """
  try:
    with open(path, 'r', encoding='utf-8') as handle:
      data = json.load(handle)
  except (OSError, ValueError) as exc:
    raise ConfigError(f'invalid targets file: {exc}') from exc
  if not isinstance(data, dict):
    raise ConfigError('targets file must be a JSON object')
  for key in ('targets', 'initialCoins', 'initialStash'):
    if key not in data:
      raise ConfigError(f'targets file missing {key}')
  raw_targets = data['targets']
  raw_stash = data['initialStash']
  if not isinstance(raw_targets, dict) or not isinstance(raw_stash, dict):
    raise ConfigError('targets and initialStash must be objects')
  if not _is_count(data['initialCoins']):
    raise ConfigError('initialCoins must be a non-negative int')
  items: dict[str, int] = {}
  for item_id in TARGET_IDS:
    value = raw_targets.get(item_id)
    if not _is_count(value):
      raise ConfigError(f'invalid target count for {item_id}')
    items[item_id] = value
  stash: dict[str, int] = {}
  for item_id, value in raw_stash.items():
    if not _is_count(value):
      raise ConfigError(f'invalid stash count for {item_id}')
    stash[item_id] = value
  return TargetSpec(items, data['initialCoins'], stash)


def goal_met(counts: dict[str, int], spec: TargetSpec) -> bool:
  """Return True when every goal count is reached.

  Args:
    counts: Observed Bag counts keyed by item id.
    spec: The target specification.

  Returns:
    True only if all goal counts are met.
  """
  return all(counts.get(item_id, 0) >= required
             for item_id, required in spec.items.items())


def verify_initial_state(
    observed: dict[str, int], spec: TargetSpec
) -> dict[str, tuple[int, int]]:
  """Report drift between observed counts and the initial stash.

  Args:
    observed: Observed Bag counts keyed by item id.
    spec: The target specification.

  Returns:
    A mapping of drifted item id to (expected, observed).
  """
  mismatches: dict[str, tuple[int, int]] = {}
  for item_id, expected in spec.initial_stash.items():
    actual = observed.get(item_id, 0)
    if actual != expected:
      mismatches[item_id] = (expected, actual)
  return mismatches
