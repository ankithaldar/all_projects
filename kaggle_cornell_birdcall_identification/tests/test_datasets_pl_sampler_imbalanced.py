#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Tests of the inverse-frequency sampler.

The weight of a row is ``1 / count`` of its species, and
``len(dataset)`` indices are drawn **with replacement**, so every species
holds an equal share of the total weight.  The draw is reproducible: the
generator is derived from ``(seed, epoch)``.
'''

import pytest

from birdcall.datasets.pl_sampler_imbalanced import ImbalancedDatasetSampler


def test_sampler_balances_the_classes(tmp_path, make_dataset, fold_species):
  '''Every species ends up with the same share of the draws.'''
  dataset = make_dataset(tmp_path, is_training=True, counts=(12, 6, 2))
  sampler = ImbalancedDatasetSampler(dataset, dataset.get_label, seed=7)
  assert len(sampler) == len(dataset)
  totals = {species: 0 for species in fold_species[:3]}
  for epoch in range(300):
    sampler.set_epoch(epoch)
    for index in sampler:
      totals[dataset.get_label(dataset, index)] += 1
  draws = sum(totals.values())
  for species, total in totals.items():
    assert total == pytest.approx(draws / 3.0, rel=0.15)


def test_sampler_is_reproducible_for_a_seed_and_epoch(tmp_path,
                                                       make_dataset):
  '''The same seed and the same epoch give the same draw.'''
  dataset = make_dataset(tmp_path, is_training=True, counts=(12, 6, 2))
  first = ImbalancedDatasetSampler(dataset, dataset.get_label, seed=7)
  second = ImbalancedDatasetSampler(dataset, dataset.get_label, seed=7)
  first.set_epoch(4)
  second.set_epoch(4)
  assert list(first) == list(second)
  second.set_epoch(5)
  assert list(first) != list(second)


def test_the_weights_are_the_inverse_of_the_counts(tmp_path, make_dataset,
                                                   fold_species):
  '''A row weighs ``1 / count`` of its species.'''
  dataset = make_dataset(tmp_path, is_training=True, counts=(4, 2, 1))
  sampler = ImbalancedDatasetSampler(dataset, dataset.get_label, seed=7)
  assert len(sampler.weights) == len(dataset)
  assert sampler.label_to_count[fold_species[0]] == 4
  assert sampler.label_to_count[fold_species[1]] == 2
  assert sampler.label_to_count[fold_species[2]] == 1
  for index in range(len(dataset)):
    label = dataset.get_label(dataset, index)
    assert float(sampler.weights[index]) == pytest.approx(
        1.0 / sampler.label_to_count[label])


def test_the_sampler_is_not_a_distributed_sampler(tmp_path, make_dataset):
  '''Lightning shards a custom sampler; it does not replace it.'''
  from torch.utils.data import DistributedSampler
  dataset = make_dataset(tmp_path, is_training=True)
  sampler = ImbalancedDatasetSampler(dataset, dataset.get_label, seed=7)
  assert not isinstance(sampler, DistributedSampler)
