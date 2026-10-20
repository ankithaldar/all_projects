#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Tests of the mixup coefficient generator and of the mixing kernel.'''

import numpy as np
import pytest
import torch

from birdcall.augmentations.mixup import do_mixup
from birdcall.augmentations.mixup import Mixup


def test_the_lambda_is_drawn_per_pair():
  mixup = Mixup(mixup_alpha=1.0)
  lambdas = mixup.get_lambda(batch_size=8, device='cpu')
  assert lambdas.shape == (8,)
  assert lambdas.dtype == torch.float32
  for index in range(0, 8, 2):
    assert float(lambdas[index]) + float(lambdas[index + 1]) == pytest.approx(
        1.0)


def test_the_same_seed_gives_the_same_sequence():
  first = Mixup(mixup_alpha=1.0).get_lambda(batch_size=16, device='cpu')
  second = Mixup(mixup_alpha=1.0).get_lambda(batch_size=16, device='cpu')
  assert torch.equal(first, second)


def test_a_different_seed_gives_a_different_sequence():
  first = Mixup(mixup_alpha=1.0, random_seed=1).get_lambda(
      batch_size=16, device='cpu')
  second = Mixup(mixup_alpha=1.0, random_seed=2).get_lambda(
      batch_size=16, device='cpu')
  assert not torch.equal(first, second)


def test_do_mixup_halves_the_batch_and_mixes_the_pairs():
  labels = torch.zeros(8, 264)
  labels[0, 3] = 1.0
  labels[1, 7] = 1.0
  lambdas = torch.tensor([0.25, 0.75, 0.5, 0.5, 0.25, 0.75, 0.5, 0.5])
  mixed = do_mixup(labels, lambdas)
  assert mixed.shape == (4, 264)
  expected = (labels[0::2] * lambdas[0::2].unsqueeze(1) +
              labels[1::2] * lambdas[1::2].unsqueeze(1))
  assert torch.allclose(mixed, expected)


def test_do_mixup_is_rank_agnostic():
  spectrum = torch.arange(48, dtype=torch.float32).reshape(2, 2, 3, 4)
  lambdas = torch.tensor([0.25, 0.75])
  mixed = do_mixup(spectrum, lambdas)
  assert mixed.shape == (1, 2, 3, 4)
  expected = (spectrum[0::2] * lambdas[0::2].reshape(1, 1, 1, 1) +
              spectrum[1::2] * lambdas[1::2].reshape(1, 1, 1, 1))
  assert torch.allclose(mixed, expected)


def test_do_mixup_works_on_numpy_arrays():
  values = np.arange(8, dtype=np.float32).reshape(4, 2)
  lambdas = np.array([0.25, 0.75, 0.5, 0.5], dtype=np.float32)
  mixed = do_mixup(values, lambdas)
  assert mixed.shape == (2, 2)
  expected = (values[0::2] * lambdas[0::2].reshape(2, 1) +
              values[1::2] * lambdas[1::2].reshape(2, 1))
  assert np.allclose(mixed, expected)
