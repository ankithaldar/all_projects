#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Deterministic seeding primitives shared by both framework backends.

This module owns every seed *derivation* in the project.  It is deliberately
free of torch and tensorflow imports so that it stays usable from the
shared surface; the framework-specific seeding lives in ``helpers/pl_seed.py``
and ``helpers/tf_seed.py``.

Two rules are enforced here:

* A seed is a pure function of an integer base seed and a tuple of integer
  parts.  Nothing depends on wall-clock time, process id, or the order in
  which work items happen to be scheduled.
* Per-item randomness is derived from ``(seed, epoch, index)``, so the value
  drawn for item *i* does not change when the number of dataloader workers
  changes.

The stdlib ``random`` module is seeded alongside numpy because the
audiomentations transforms draw their parameters from ``random``; seeding
numpy alone would leave the waveform augmentation irreproducible.
'''

import hashlib
import random

import numpy as np

SEED_MODULUS = 2 ** 32


def derive_seed(base_seed: int, *parts: int) -> int:
  '''Derives a stable child seed from a base seed and integer parts.

  Args:
    base_seed: The root seed of the experiment.
    *parts: Additional integers identifying the consumer of the seed, for
      example an epoch number or a dataset index.

  Returns:
    A non-negative 32 bit integer seed.
  '''
  digest = hashlib.blake2b(digest_size=8)
  digest.update(str(int(base_seed)).encode('utf-8'))
  for part in parts:
    digest.update(b'|')
    digest.update(str(int(part)).encode('utf-8'))
  return int.from_bytes(digest.digest(), 'big') % SEED_MODULUS


def make_generator(base_seed: int, *parts: int) -> np.random.Generator:
  '''Builds a numpy generator from a derived seed.

  Args:
    base_seed: The root seed of the experiment.
    *parts: Additional integers identifying the consumer of the generator.

  Returns:
    A seeded ``numpy.random.Generator``.
  '''
  return np.random.default_rng(derive_seed(base_seed, *parts))


def seed_worker(worker_id: int, base_seed: int, epoch: int = 0) -> int:
  '''Seeds the global numpy and stdlib random streams of a worker.

  Args:
    worker_id: The dataloader worker index.
    base_seed: The root seed of the experiment.
    epoch: The epoch the worker is starting.

  Returns:
    The derived seed, so a caller can log or assert on it.
  '''
  seed = derive_seed(base_seed, epoch, worker_id)
  np.random.seed(seed)
  random.seed(seed)
  return seed
