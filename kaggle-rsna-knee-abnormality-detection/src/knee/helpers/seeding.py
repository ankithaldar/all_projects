#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Reproducibility helpers."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = False) -> None:
  """Seed python/numpy/torch (+CUDA) and configure cudnn.

  Deterministic mode trades speed for bit-exact runs; default off since
  AUC differences from nondeterminism are <0.001 but speed cost is ~30%.
  """
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  os.environ['PYTHONHASHSEED'] = str(seed)
  torch.backends.cudnn.benchmark = not deterministic
  if deterministic:  # pragma: no cover - rarely enabled in practice
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def worker_init_fn(worker_id: int) -> None:
  """Distinct-but-derived seed per dataloader worker."""
  seed = torch.initial_seed() % 2**32
  np.random.seed(seed + worker_id)
  random.seed(seed + worker_id)
