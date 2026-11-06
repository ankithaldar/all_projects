#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Tests for genetic operators.'''

from __future__ import annotations

from unittest import mock

import numpy as np
import pytest

from src.core.items import NUM_CRAFTABLE
from src.ga.chromosome import MAX_TICKS
from src.ga.operators import GaOperators


def _genes(value=0):
  '''Return a small gene array filled with a value.'''
  return np.full((MAX_TICKS, NUM_CRAFTABLE), value, dtype=np.uint8)


def _operators(crafting_tree):
  '''Build operators with a deterministic config.'''
  config = {
    'time_block_size': 48,
    'mut_batch_indpb': 0.5,
    'mut_time_shift_prob': 0.15,
    'max_batch_size': 20,
  }
  return GaOperators(crafting_tree, config)


class TestCxTimeBlock:
  '''Tests for time-block crossover.'''

  def test_swaps_blocks_and_preserves_shape(
    self, crafting_tree
  ):
    '''Crossover swaps some blocks between parents.'''
    ops = _operators(crafting_tree)
    ind1 = _genes(1)
    ind2 = _genes(9)
    out1, out2 = ops.cx_time_block(ind1.copy(), ind2.copy())
    assert out1.shape == (MAX_TICKS, NUM_CRAFTABLE)
    assert out2.shape == (MAX_TICKS, NUM_CRAFTABLE)
    # Every cell must come from one of the two parents.
    assert set(np.unique(out1)).issubset({1, 9})
    assert set(np.unique(out2)).issubset({1, 9})

  def test_no_swap_when_random_high(self, crafting_tree):
    '''With random() >= 0.5 for every block nothing changes.'''
    ops = _operators(crafting_tree)
    ind1 = _genes(1)
    ind2 = _genes(9)
    with mock.patch(
      'src.ga.operators.random.random', return_value=0.99
    ):
      out1, out2 = ops.cx_time_block(ind1.copy(), ind2.copy())
    assert np.all(out1 == 1)
    assert np.all(out2 == 9)

  def test_full_swap_when_random_low(self, crafting_tree):
    '''With random() < 0.5 for every block parents swap fully.'''
    ops = _operators(crafting_tree)
    ind1 = _genes(1)
    ind2 = _genes(9)
    with mock.patch(
      'src.ga.operators.random.random', return_value=0.0
    ):
      out1, out2 = ops.cx_time_block(ind1.copy(), ind2.copy())
    assert np.all(out1 == 9)
    assert np.all(out2 == 1)


class TestMutBatchSize:
  '''Tests for batch-size mutation.'''

  def test_mutation_changes_some_genes(self, crafting_tree):
    '''Mutation with high probability changes values.'''
    ops = _operators(crafting_tree)
    ind = _genes(3)
    (mutated,) = ops.mut_batch_size(ind.copy())
    assert mutated.shape == ind.shape
    # With indpb=0.5 some genes almost surely changed, but the
    # operator must never exceed max_batch.
    assert mutated.max() <= 20

  def test_zero_probability_is_identity(self, crafting_tree):
    '''Zero mutation probability leaves the individual intact.'''
    config = {'mut_batch_indpb': 0.0, 'max_batch_size': 20}
    ops = GaOperators(crafting_tree, config)
    ind = _genes(7)
    (mutated,) = ops.mut_batch_size(ind.copy())
    assert np.all(mutated == 7)

  def test_values_within_max_batch(self, crafting_tree):
    '''Mutated values stay within the configured cap.'''
    config = {'mut_batch_indpb': 1.0, 'max_batch_size': 5}
    ops = GaOperators(crafting_tree, config)
    (mutated,) = ops.mut_batch_size(_genes(0))
    assert mutated.max() <= 5


class TestMutTimeShift:
  '''Tests for time-shift mutation.'''

  def test_skipped_when_random_high(self, crafting_tree):
    '''A random draw above the probability leaves genes intact.'''
    ops = _operators(crafting_tree)
    ind = _genes(4)
    with mock.patch(
      'src.ga.operators.random.random', return_value=0.99
    ):
      (mutated,) = ops.mut_time_shift(ind.copy())
    assert np.all(mutated == 4)

  def test_positive_shift_moves_down(self, crafting_tree):
    '''A positive shift moves a column towards later ticks.'''
    ops = _operators(crafting_tree)
    ind = np.zeros((MAX_TICKS, NUM_CRAFTABLE), dtype=np.uint8)
    ind[0, 0] = 6
    with mock.patch(
      'src.ga.operators.random.random', return_value=0.0
    ), mock.patch(
      'src.ga.operators.random.randint', side_effect=[0, 3]
    ):
      (mutated,) = ops.mut_time_shift(ind.copy())
    assert mutated[0, 0] == 0
    assert mutated[3, 0] == 6

  def test_negative_shift_moves_up(self, crafting_tree):
    '''A negative shift moves a column towards earlier ticks.'''
    ops = _operators(crafting_tree)
    ind = np.zeros((MAX_TICKS, NUM_CRAFTABLE), dtype=np.uint8)
    ind[10, 2] = 5
    with mock.patch(
      'src.ga.operators.random.random', return_value=0.0
    ), mock.patch(
      'src.ga.operators.random.randint', side_effect=[2, -4]
    ):
      (mutated,) = ops.mut_time_shift(ind.copy())
    assert mutated[10, 2] == 0
    assert mutated[6, 2] == 5

  def test_zero_shift_is_identity(self, crafting_tree):
    '''A zero shift leaves the individual intact.'''
    ops = _operators(crafting_tree)
    ind = _genes(2)
    with mock.patch(
      'src.ga.operators.random.random', return_value=0.0
    ), mock.patch(
      'src.ga.operators.random.randint', side_effect=[1, 0]
    ):
      (mutated,) = ops.mut_time_shift(ind.copy())
    assert np.all(mutated == 2)


class TestMutate:
  '''Tests for the combined mutate operator.'''

  def test_mutate_returns_tuple_and_shape(self, crafting_tree):
    '''Combined mutation returns a one-tuple of same shape.'''
    ops = _operators(crafting_tree)
    ind = _genes(1)
    result = ops.mutate(ind.copy())
    assert isinstance(result, tuple)
    assert len(result) == 1
    assert result[0].shape == ind.shape

  def test_mutate_with_noop_shift(self, crafting_tree):
    '''Mutation still works when the shift is skipped.'''
    ops = _operators(crafting_tree)
    with mock.patch(
      'src.ga.operators.random.random', return_value=0.99
    ):
      (mutated,) = ops.mutate(_genes(1))
    assert mutated.shape == (MAX_TICKS, NUM_CRAFTABLE)


if __name__ == '__main__':
  pytest.main([__file__])
