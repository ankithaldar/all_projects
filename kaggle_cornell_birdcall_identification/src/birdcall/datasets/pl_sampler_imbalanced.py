#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''The inverse-frequency sampler of the SED pipeline.

The original vendored this from ``ufoym/imbalanced-dataset-sampler``: the
weight of a row is ``1 / count`` of its primary species, and
``len(dataset)`` indices are drawn **with replacement**.  Every species
therefore holds an equal share of the total weight, which is what keeps the
rare classes from being drowned out by the common ones.

Two changes are made on purpose:

* The draw is reproducible.  The original drew from the global torch RNG
  on every ``__iter__``; here the generator is derived from
  ``(seed, epoch)`` and ``set_epoch`` advances the epoch counter.
* The sampler is **not** a ``DistributedSampler``.  Lightning shards a
  custom sampler rather than replacing it, so the weighting survives a
  DDP run; a ``DistributedSampler`` would have replaced it.
'''

import torch
from torch.utils.data import Sampler

from birdcall.helpers.pl_seed import make_torch_generator


class ImbalancedDatasetSampler(Sampler):
  '''Draws the training indices with inverse-frequency weights.

  Attributes:
    weights: The per-row weight, ``1 / count`` of the row's species.
    label_to_count: How many rows each species has.
  '''

  def __init__(self, dataset, get_label, seed: int = 42) -> None:
    '''Builds the weight vector of a dataset.

    Args:
      dataset: The dataset to sample.
      get_label: The label callback, called as ``get_label(dataset, idx)``
        and returning the primary species code.
      seed: The root seed of the draw.
    '''
    self.seed = seed
    self.epoch = 0
    self.indices = list(range(len(dataset)))
    self.num_samples = len(self.indices)
    self.label_to_count: dict[str, int] = {}
    for index in self.indices:
      label = get_label(dataset, index)
      self.label_to_count[label] = self.label_to_count.get(label, 0) + 1
    self.weights = torch.DoubleTensor([
        1.0 / self.label_to_count[get_label(dataset, index)]
        for index in self.indices])

  def set_epoch(self, epoch: int) -> None:
    '''Sets the epoch the next draw belongs to.

    Args:
      epoch: The epoch index.
    '''
    self.epoch = epoch

  def __iter__(self):
    '''Draws one epoch of indices.

    Returns:
      An iterator of ``len(dataset)`` indices, with replacement.
    '''
    generator = make_torch_generator(self.seed, self.epoch)
    drawn = torch.multinomial(self.weights, self.num_samples,
                              replacement=True, generator=generator)
    return iter(drawn.tolist())

  def __len__(self) -> int:
    '''Returns the number of indices an epoch holds.

    Returns:
      The size of the dataset.
    '''
    return self.num_samples
