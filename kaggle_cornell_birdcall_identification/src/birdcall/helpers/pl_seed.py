#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Torch specific seeding for the PyTorch Lightning backend.

This module is the only place in the ``pl_`` surface allowed to call
``torch.manual_seed``.  ``helpers/seed.py`` owns the framework-agnostic
derivations and the numpy/stdlib seeding.
'''

import numpy as np
import torch

from birdcall.helpers.seed import derive_seed
from birdcall.helpers.seed import seed_worker


def seed_everything(base_seed: int, rank: int = 0,
                    deterministic: bool = True) -> int:
  '''Seeds every global RNG the torch backend uses.

  Args:
    base_seed: The root seed of the experiment.
    rank: The distributed rank, so that ranks do not draw identical
      sequences.  The original seeded ``42 + rank``; the offset is kept.
    deterministic: Whether to request deterministic kernels.

  Returns:
    The derived seed for this rank.
  '''
  seed = derive_seed(base_seed, rank)
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
  if deterministic:
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
  return seed


def make_torch_generator(base_seed: int, *parts: int) -> torch.Generator:
  '''Builds a seeded torch generator from a derived seed.

  Args:
    base_seed: The root seed of the experiment.
    *parts: Additional integers identifying the consumer of the generator.

  Returns:
    A seeded ``torch.Generator``.
  '''
  generator = torch.Generator()
  generator.manual_seed(derive_seed(base_seed, *parts))
  return generator


def worker_init_fn(worker_id: int, base_seed: int, epoch: int = 0) -> None:
  '''Seeds numpy, the stdlib random module and torch in a dataloader worker.

  Args:
    worker_id: The dataloader worker index.
    base_seed: The root seed of the experiment.
    epoch: The epoch the worker is starting.
  '''
  seed = seed_worker(worker_id, base_seed, epoch)
  torch.manual_seed(seed)
