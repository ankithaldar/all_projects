#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''TensorFlow specific seeding for the Keras backend.

This module is the only place in the ``tf_`` surface allowed to call
``tf.random.set_seed``.  ``helpers/seed.py`` owns the framework-agnostic
derivations and the numpy/stdlib seeding.
'''

import os

import numpy as np
import tensorflow as tf

from birdcall.helpers.seed import derive_seed
from birdcall.helpers.seed import seed_worker

CUBLAS_WORKSPACE_CONFIG = ':4096:8'


def enable_determinism() -> None:
  '''Requests deterministic TensorFlow kernels.

  The CUDA cublas workspace has to be configured before the first CUDA
  operation, so this is a no-op when no GPU is present.
  '''
  os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', CUBLAS_WORKSPACE_CONFIG)
  tf.config.experimental.enable_op_determinism()


def seed_everything(base_seed: int, rank: int = 0,
                    deterministic: bool = True) -> int:
  '''Seeds every global RNG the Keras backend uses.

  Args:
    base_seed: The root seed of the experiment.
    rank: The distributed replica index, so replicas do not draw identical
      sequences.
    deterministic: Whether to request deterministic kernels.

  Returns:
    The derived seed for this replica.
  '''
  seed = derive_seed(base_seed, rank)
  np.random.seed(seed)
  tf.random.set_seed(seed)
  if deterministic:
    enable_determinism()
  return seed


def worker_init_fn(worker_id: int, base_seed: int, epoch: int = 0) -> None:
  '''Seeds numpy and the stdlib random module in a tf.data worker.

  Args:
    worker_id: The tf.data worker index.
    base_seed: The root seed of the experiment.
    epoch: The epoch the worker is starting.
  '''
  seed_worker(worker_id, base_seed, epoch)
