from __future__ import annotations

import numpy as np
import pytest

from src.core.items import ItemId, Ingredient
from src.core.inventory import Stash


class TestStash:
  def test_initial_empty(self, fresh_stash: Stash):
    assert fresh_stash.get(ItemId.COTTON) == 0

  def test_add(self, fresh_stash: Stash):
    fresh_stash.add(ItemId.COTTON, 10)
    assert fresh_stash.get(ItemId.COTTON) == 10

  def test_add_multiple(self, fresh_stash: Stash):
    fresh_stash.add(ItemId.COTTON, 10)
    fresh_stash.add(ItemId.COTTON, 5)
    assert fresh_stash.get(ItemId.COTTON) == 15

  def test_remove_success(self, fresh_stash: Stash):
    fresh_stash.add(ItemId.COTTON, 10)
    assert fresh_stash.remove(ItemId.COTTON, 5) is True
    assert fresh_stash.get(ItemId.COTTON) == 5

  def test_remove_exact(self, fresh_stash: Stash):
    fresh_stash.add(ItemId.COTTON, 10)
    assert fresh_stash.remove(ItemId.COTTON, 10) is True
    assert fresh_stash.get(ItemId.COTTON) == 0

  def test_remove_insufficient(self, fresh_stash: Stash):
    fresh_stash.add(ItemId.COTTON, 5)
    assert fresh_stash.remove(ItemId.COTTON, 10) is False
    assert fresh_stash.get(ItemId.COTTON) == 5

  def test_add_negative_raises(self, fresh_stash: Stash):
    with pytest.raises(ValueError):
      fresh_stash.add(ItemId.COTTON, -1)

  def test_remove_negative_raises(self, fresh_stash: Stash):
    with pytest.raises(ValueError):
      fresh_stash.remove(ItemId.COTTON, -1)

  def test_can_afford_materials(self, fresh_stash: Stash):
    fresh_stash.add(ItemId.COTTON, 30)
    ingredients = (Ingredient(ItemId.COTTON, 3),)
    assert fresh_stash.can_afford_materials(ingredients, 10) is True
    assert fresh_stash.can_afford_materials(ingredients, 11) is False

  def test_can_afford_multiple_ingredients(self, fresh_stash: Stash):
    fresh_stash.add(ItemId.STRING, 10)
    fresh_stash.add(ItemId.WOOD, 10)
    ingredients = (
      Ingredient(ItemId.STRING, 2),
      Ingredient(ItemId.WOOD, 2),
    )
    assert fresh_stash.can_afford_materials(ingredients, 5) is True
    assert fresh_stash.can_afford_materials(ingredients, 6) is False

  def test_max_affordable_batch_materials(self, fresh_stash: Stash):
    fresh_stash.add(ItemId.COTTON, 15)
    ingredients = (Ingredient(ItemId.COTTON, 3),)
    assert fresh_stash.max_affordable_batch_materials(ingredients) == 5

  def test_as_array(self, fresh_stash: Stash):
    fresh_stash.add(ItemId.COTTON, 5)
    arr = fresh_stash.as_array()
    assert arr.shape == (23,)
    assert arr[0] == 5
    assert arr.dtype == np.int32

  def test_as_array_is_copy(self, fresh_stash: Stash):
    fresh_stash.add(ItemId.COTTON, 5)
    arr = fresh_stash.as_array()
    arr[0] = 999
    assert fresh_stash.get(ItemId.COTTON) == 5

  def test_reset(self, fresh_stash: Stash):
    fresh_stash.add(ItemId.COTTON, 100)
    fresh_stash.reset()
    assert fresh_stash.get(ItemId.COTTON) == 0

  def test_set_counts(self, fresh_stash: Stash):
    counts = np.zeros(23, dtype=np.int32)
    counts[0] = 42
    fresh_stash.set_counts(counts)
    assert fresh_stash.get(ItemId.COTTON) == 42
