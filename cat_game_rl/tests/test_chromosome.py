from __future__ import annotations

import numpy as np
import pytest

from src.core.items import CraftingTree, ItemId, ITEM_NAME_TO_ID, CRAFTABLE_ITEM_IDS
from src.ga.chromosome import Chromosome, MAX_TICKS


@pytest.fixture
def targets():
  return {ItemId.STRING: 5}


class TestChromosome:
  def test_create_random_shape(self):
    genes = Chromosome.create_random(density=0.05, max_batch=10)
    assert genes.shape == (MAX_TICKS, 19)
    assert genes.dtype == np.uint8

  def test_create_random_sparsity(self):
    genes = Chromosome.create_random(density=0.05, max_batch=10)
    nonzero_frac = np.count_nonzero(genes) / genes.size
    assert 0.01 < nonzero_frac < 0.15

  def test_decode_all_zeros(self):
    genes = np.zeros((MAX_TICKS, 19), dtype=np.uint8)
    schedule = Chromosome.decode(genes)
    assert len(schedule) == MAX_TICKS
    for tick_actions in schedule:
      assert len(tick_actions) == 0

  def test_decode_single_action(self):
    genes = np.zeros((MAX_TICKS, 19), dtype=np.uint8)
    genes[0, 0] = 5  # STRING at tick 0, batch 5
    schedule = Chromosome.decode(genes)
    assert CRAFTABLE_ITEM_IDS[0] in schedule[0]
    assert schedule[0][CRAFTABLE_ITEM_IDS[0]] == 5

  def test_evaluate_returns_three_floats(
    self, crafting_tree: CraftingTree, targets: dict
  ):
    genes = np.zeros((MAX_TICKS, 19), dtype=np.uint8)
    result = Chromosome.evaluate(genes, crafting_tree, targets)
    assert len(result) == 3
    assert all(isinstance(v, float) for v in result)

  def test_all_zeros_high_penalty(
    self, crafting_tree: CraftingTree, targets: dict
  ):
    genes = np.zeros((MAX_TICKS, 19), dtype=np.uint8)
    cost, time, waste = Chromosome.evaluate(genes, crafting_tree, targets)
    assert time == MAX_TICKS + 1  # penalty tick

  def test_crafting_string_reduces_time(
    self, crafting_tree: CraftingTree, targets: dict
  ):
    genes = np.zeros((MAX_TICKS, 19), dtype=np.uint8)
    # Craft string every tick for first 10 ticks
    for t in range(10):
      genes[t, 0] = 5  # STRING batch=5
    cost, time, waste = Chromosome.evaluate(genes, crafting_tree, targets)
    assert time < MAX_TICKS + 1  # should complete

  def test_waste_tracks_excess(self, crafting_tree: CraftingTree):
    targets = {ItemId.STRING: 1}
    genes = np.zeros((MAX_TICKS, 19), dtype=np.uint8)
    genes[0, 0] = 10  # STRING batch=10, target is only 1
    cost, time, waste = Chromosome.evaluate(genes, crafting_tree, targets)
    assert waste >= 9  # at least 9 excess strings
