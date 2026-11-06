#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Inventory storage for crafted items.'''

from __future__ import annotations

from typing import Tuple

import numpy as np

from src.core.items import NUM_ITEMS, ItemId, Ingredient


class Stash:
  '''Stores counts of items in a fixed-size array.'''

  def __init__(self):
    '''Initializes an empty inventory.'''
    self._counts = np.zeros(NUM_ITEMS, dtype=np.int32)

  def get(self, item_id: ItemId) -> int:
    '''Returns the count for an item.

    Args:
      item_id: Item to look up.

    Returns:
      Current count of the item.
    '''
    return int(self._counts[int(item_id)])

  def add(self, item_id: ItemId, qty: int) -> None:
    '''Adds items to the inventory.

    Args:
      item_id: Item to add.
      qty: Quantity to add.

    Raises:
      ValueError: If qty is negative.
    '''
    if qty < 0:
      raise ValueError(f'Cannot add negative quantity: {qty}')
    self._counts[int(item_id)] += qty

  def remove(self, item_id: ItemId, qty: int) -> bool:
    '''Removes items from the inventory.

    Args:
      item_id: Item to remove.
      qty: Quantity to remove.

    Returns:
      True if enough items were present, else False.

    Raises:
      ValueError: If qty is negative.
    '''
    if qty < 0:
      raise ValueError(f'Cannot remove negative quantity: {qty}')
    idx = int(item_id)
    if self._counts[idx] < qty:
      return False
    self._counts[idx] -= qty
    return True

  def can_afford_materials(
    self, ingredients: Tuple[Ingredient, ...], batch_size: int
  ) -> bool:
    '''Checks whether materials cover a batch.

    Args:
      ingredients: Required ingredients.
      batch_size: Batch size to check.

    Returns:
      True if all ingredients are sufficient, else False.
    '''
    for ing in ingredients:
      if self._counts[int(ing.item_id)] < ing.quantity * batch_size:
        return False
    return True

  def max_affordable_batch_materials(
    self, ingredients: Tuple[Ingredient, ...]
  ) -> int:
    '''Computes the largest affordable batch from materials.

    Args:
      ingredients: Required ingredients.

    Returns:
      Largest affordable batch size.
    '''
    if not ingredients:
      return 9999
    max_batch = float('inf')
    for ing in ingredients:
      if ing.quantity <= 0:
        continue
      available = self._counts[int(ing.item_id)]
      max_batch = min(max_batch, available // ing.quantity)
    return int(max_batch) if max_batch != float('inf') else 9999

  def as_array(self) -> np.ndarray:
    '''Returns a copy of the counts array.

    Returns:
      Copy of the internal counts array.
    '''
    return self._counts.copy()

  def reset(self) -> None:
    '''Clears all item counts.'''
    self._counts[:] = 0

  def set_counts(self, counts: np.ndarray) -> None:
    '''Replaces all item counts.

    Args:
      counts: New counts array.
    '''
    self._counts[:] = counts
