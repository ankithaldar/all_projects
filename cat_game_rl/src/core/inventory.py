from __future__ import annotations

from typing import Tuple

import numpy as np

from src.core.items import NUM_ITEMS, ItemId, Ingredient


class Stash:
  def __init__(self):
    self._counts = np.zeros(NUM_ITEMS, dtype=np.int32)

  def get(self, item_id: ItemId) -> int:
    return int(self._counts[int(item_id)])

  def add(self, item_id: ItemId, qty: int) -> None:
    if qty < 0:
      raise ValueError(f"Cannot add negative quantity: {qty}")
    self._counts[int(item_id)] += qty

  def remove(self, item_id: ItemId, qty: int) -> bool:
    if qty < 0:
      raise ValueError(f"Cannot remove negative quantity: {qty}")
    idx = int(item_id)
    if self._counts[idx] < qty:
      return False
    self._counts[idx] -= qty
    return True

  def can_afford_materials(
    self, ingredients: Tuple[Ingredient, ...], batch_size: int
  ) -> bool:
    for ing in ingredients:
      if self._counts[int(ing.item_id)] < ing.quantity * batch_size:
        return False
    return True

  def max_affordable_batch_materials(
    self, ingredients: Tuple[Ingredient, ...]
  ) -> int:
    if not ingredients:
      return 0
    max_batch = float("inf")
    for ing in ingredients:
      available = self._counts[int(ing.item_id)]
      max_batch = min(max_batch, available // ing.quantity)
    return int(max_batch)

  def as_array(self) -> np.ndarray:
    return self._counts.copy()

  def reset(self) -> None:
    self._counts[:] = 0

  def set_counts(self, counts: np.ndarray) -> None:
    self._counts[:] = counts
