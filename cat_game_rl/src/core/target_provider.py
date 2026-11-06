#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Delivery targets for the crafting game.'''

from __future__ import annotations

from typing import Dict

import numpy as np
import yaml

from src.core.items import NUM_ITEMS, ItemId, ITEM_NAME_TO_ID


class TargetProvider:
  '''Tracks delivery targets and delivered counts.'''

  def __init__(self, targets_path: str):
    '''Loads targets from a YAML file.

    Args:
      targets_path: Path to the targets YAML file.
    '''
    self._targets_path = targets_path
    self._targets: Dict[ItemId, int] = {}
    self._load(targets_path)
    self._delivered: Dict[ItemId, int] = {k: 0 for k in self._targets}

  def _load(self, path: str) -> None:
    '''Loads targets from a YAML file.

    Args:
      path: Path to the targets YAML file.

    Raises:
      ValueError: If the YAML is invalid.
    '''
    with open(path, 'r', encoding='utf-8') as f:
      data = yaml.safe_load(f)
    if data is None or 'targets' not in data or data['targets'] is None:
      raise ValueError(f'Invalid targets YAML: {path}')
    self._targets = {}
    for name, count in data['targets'].items():
      try:
        item_id = ITEM_NAME_TO_ID[str(name).lower()]
      except KeyError:
        raise ValueError(f'Unknown target item: {name}') from None
      if count < 0:
        raise ValueError(f'Invalid target count for {name}: {count}')
      self._targets[item_id] = count

  def reload(self, path: str | None = None) -> None:
    '''Reloads targets and resets delivered counts.

    Args:
      path: Optional path to reload from.
    '''
    self._load(path or self._targets_path)
    self._delivered = {k: 0 for k in self._targets}

  @property
  def targets(self) -> Dict[ItemId, int]:
    '''Returns a copy of the targets mapping.'''
    return dict(self._targets)

  def deliver(self, item_id: ItemId, qty: int) -> None:
    '''Records delivered items.

    Args:
      item_id: Item delivered.
      qty: Quantity delivered.
    '''
    if item_id in self._targets:
      self._delivered[item_id] = self._delivered.get(item_id, 0) + qty

  def remaining(self, item_id: ItemId) -> int:
    '''Returns the remaining quantity for an item.

    Args:
      item_id: Item to check.

    Returns:
      Remaining quantity to deliver.
    '''
    if item_id not in self._targets:
      return 0
    return max(0, self._targets[item_id] - self._delivered.get(item_id, 0))

  def is_complete(self) -> bool:
    '''Checks whether all targets are met.

    Returns:
      True if every target is met, else False.
    '''
    return all(
      self._delivered.get(k, 0) >= v for k, v in self._targets.items()
    )

  def targets_remaining_array(self) -> np.ndarray:
    '''Returns remaining quantities as an array.

    Returns:
      Array of remaining quantities per item.
    '''
    arr = np.zeros(NUM_ITEMS, dtype=np.int32)
    for item_id, target in self._targets.items():
      delivered = self._delivered.get(item_id, 0)
      arr[int(item_id)] = target - delivered
    return arr

  def targets_total_array(self) -> np.ndarray:
    '''Returns total targets as an array.

    Returns:
      Array of total quantities per item.
    '''
    arr = np.zeros(NUM_ITEMS, dtype=np.int32)
    for item_id, target in self._targets.items():
      arr[int(item_id)] = target
    return arr

  def fraction_complete(self) -> float:
    '''Returns the fraction of targets completed.

    Returns:
      Fraction of required items delivered.
    '''
    total_required = sum(self._targets.values())
    if total_required == 0:
      return 1.0
    total_delivered = sum(
      min(self._delivered.get(k, 0), v) for k, v in self._targets.items()
    )
    return total_delivered / total_required

  def reset(self) -> None:
    '''Resets delivered counts to zero.'''
    self._delivered = {k: 0 for k in self._targets}
