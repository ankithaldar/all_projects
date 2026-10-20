#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Tests of the TensorFlow seeding helpers.

This module is the only place in the ``tf_`` surface allowed to call
``tf.random.set_seed``.
'''

import numpy as np
import tensorflow as tf

from birdcall.helpers.seed import derive_seed
from birdcall.helpers.tf_seed import CUBLAS_WORKSPACE_CONFIG
from birdcall.helpers.tf_seed import enable_determinism
from birdcall.helpers.tf_seed import seed_everything
from birdcall.helpers.tf_seed import worker_init_fn


def test_the_seed_everything_returns_the_derived_seed():
  '''The seed is derived from the base seed and the replica index.'''
  seed = seed_everything(42, 0)
  assert seed == derive_seed(42, 0)
  assert seed_everything(42, 1) == derive_seed(42, 1)


def test_the_tensorflow_stream_is_seeded():
  '''The Keras backend draws from the seeded TensorFlow stream.'''
  seed_everything(42, 0)
  first = tf.random.uniform((4,)).numpy()
  seed_everything(42, 0)
  second = tf.random.uniform((4,)).numpy()
  assert np.array_equal(first, second)


def test_two_replicas_do_not_draw_identical_sequences():
  '''The replica index keeps the replicas apart.'''
  seed_everything(42, 0)
  first = tf.random.uniform((4,)).numpy()
  seed_everything(42, 1)
  second = tf.random.uniform((4,)).numpy()
  assert not np.array_equal(first, second)


def test_the_numpy_stream_is_seeded_too():
  '''The numpy stream is seeded with the same value.'''
  seed_everything(42, 0)
  first = np.random.rand(4)
  seed_everything(42, 0)
  second = np.random.rand(4)
  assert np.array_equal(first, second)


def test_the_worker_init_seeds_the_global_rngs():
  '''A tf.data worker seeds numpy and the stdlib random.'''
  worker_init_fn(0, 42, 0)
  first = np.random.rand()
  worker_init_fn(1, 42, 0)
  second = np.random.rand()
  assert first != second


def test_the_determinism_can_be_enabled():
  '''Requesting deterministic kernels does not raise.'''
  enable_determinism()
  assert CUBLAS_WORKSPACE_CONFIG == ':4096:8'
