#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Tests of the shared seeding primitives.

A seed is a pure function of an integer base seed and a tuple of integer
parts.  Nothing depends on wall-clock time, process id, or the order in
which work items happen to be scheduled.
'''

import random

import numpy as np
import pytest

from birdcall.helpers.seed import SEED_MODULUS
from birdcall.helpers.seed import derive_seed
from birdcall.helpers.seed import make_generator
from birdcall.helpers.seed import seed_worker


def test_the_derived_seed_is_a_32_bit_integer():
  '''A seed is a non-negative 32 bit integer.'''
  seed = derive_seed(42, 0, 0)
  assert isinstance(seed, int)
  assert 0 <= seed < SEED_MODULUS


def test_the_derived_seed_is_a_pure_function():
  '''The same inputs give the same seed.'''
  assert derive_seed(42, 3, 7) == derive_seed(42, 3, 7)


def test_different_parts_give_different_seeds():
  '''The parts identify the consumer of the seed.'''
  assert derive_seed(42, 0) != derive_seed(42, 1)
  assert derive_seed(42, 0) != derive_seed(43, 0)
  assert derive_seed(42, 0, 1) != derive_seed(42, 1, 0)


def test_the_parts_are_not_concatenated():
  '''The parts are separated, so `1, 23` and `12, 3` differ.'''
  assert derive_seed(42, 1, 23) != derive_seed(42, 12, 3)


def test_a_negative_base_seed_still_gives_a_positive_seed():
  '''The derivation is total.'''
  assert derive_seed(-1) >= 0


def test_the_generator_is_reproducible():
  '''The same seed gives the same sequence.'''
  first = make_generator(42, 1).random(5)
  second = make_generator(42, 1).random(5)
  assert np.array_equal(first, second)


def test_the_generator_differs_per_consumer():
  '''The parts separate the streams.'''
  first = make_generator(42, 1).random(5)
  second = make_generator(42, 2).random(5)
  assert not np.array_equal(first, second)


def test_the_worker_seeds_numpy_and_the_stdlib_random():
  '''The audiomentations transforms draw from the stdlib random.'''
  seed = seed_worker(0, 42, 0)
  first = (np.random.rand(), random.random())
  seed_worker(1, 42, 0)
  second = (np.random.rand(), random.random())
  assert first != second
  assert isinstance(seed, int)


def test_the_worker_seed_depends_on_the_epoch():
  '''The epoch is part of the worker's seed.'''
  assert seed_worker(0, 42, 0) != seed_worker(0, 42, 1)
