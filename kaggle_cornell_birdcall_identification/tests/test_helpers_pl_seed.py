#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Tests of the torch seeding helpers.

This module is the only place in the ``pl_`` surface allowed to call
``torch.manual_seed``.
'''

import numpy as np
import torch

from birdcall.helpers.pl_seed import make_torch_generator
from birdcall.helpers.pl_seed import seed_everything
from birdcall.helpers.pl_seed import worker_init_fn
from birdcall.helpers.seed import derive_seed


def test_the_seed_everything_returns_the_derived_seed():
  '''The seed is derived from the base seed and the rank.'''
  seed = seed_everything(42, 0)
  assert seed == derive_seed(42, 0)
  assert seed_everything(42, 1) == derive_seed(42, 1)


def test_two_ranks_do_not_draw_identical_sequences():
  '''The rank keeps the ranks apart.'''
  seed_everything(42, 0)
  first = torch.rand(4)
  seed_everything(42, 1)
  second = torch.rand(4)
  assert not torch.equal(first, second)


def test_the_same_seed_gives_the_same_torch_sequence():
  '''The seeding is reproducible.'''
  seed_everything(42, 0)
  first = torch.rand(4)
  seed_everything(42, 0)
  second = torch.rand(4)
  assert torch.equal(first, second)


def test_the_numpy_stream_is_seeded_too():
  '''The numpy stream is seeded with the same value.'''
  seed_everything(42, 0)
  first = np.random.rand(4)
  seed_everything(42, 0)
  second = np.random.rand(4)
  assert np.array_equal(first, second)


def test_the_torch_generator_is_reproducible():
  '''The generator is derived from the base seed and the parts.'''
  first = torch.rand(4, generator=make_torch_generator(42, 1))
  second = torch.rand(4, generator=make_torch_generator(42, 1))
  assert torch.equal(first, second)


def test_the_torch_generator_differs_per_consumer():
  '''The parts separate the streams.'''
  first = torch.rand(4, generator=make_torch_generator(42, 1))
  second = torch.rand(4, generator=make_torch_generator(42, 2))
  assert not torch.equal(first, second)


def test_the_worker_init_seeds_every_global_rng():
  '''A worker seeds numpy, the stdlib random and torch.'''
  worker_init_fn(0, 42, 0)
  first = (np.random.rand(), torch.rand(1).item())
  worker_init_fn(1, 42, 0)
  second = (np.random.rand(), torch.rand(1).item())
  assert first != second
