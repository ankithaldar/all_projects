#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""2.5D-consistent augmentations.

Constraint: every slice in a stack must receive the SAME spatial
transform, otherwise the model sees physically impossible volumes.
We achieve this by reseeding the augmentation RNG once per slice with a
seed drawn per stack (works across albumentations versions), capturing
and restoring global RNG state around each item so dataloader workers
stay reproducible.

YAML declares plain albumentations specs:
  - {target: albumentations.Affine, params: {...}}
No HorizontalFlip is permitted anywhere (see schema/blueprint).
"""

from __future__ import annotations

import random
from collections.abc import Callable

import albumentations as A
import numpy as np

from knee.config_params.loader import resolve_target
from knee.config_params.schema import AugmentConfig, AugmentItem

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class SliceStackTransform:
  """Callable: np.ndarray (C,H,W) uint8 -> float32 tensor-ready array."""

  def __init__(self, items: list[AugmentItem]) -> None:
    """Compile the albumentations pipeline once.

    Args:
        items: Declarative transform specs from the experiment config.
    """
    self._compose = (
      A.Compose([self._build(it) for it in items]) if items else None
    )

  @staticmethod
  def _build(item: AugmentItem):
    """Instantiate one albumentations transform from its spec.

    Args:
        item: Declarative spec with class target path and params.

    Returns:
        A configured albumentations transform instance.
    """
    cls = resolve_target(item.target)
    return cls(**item.params)

  def __call__(self, stack: np.ndarray) -> np.ndarray:
    if self._compose is None:
      return stack.astype(np.float32) / 255.0
    c = stack.shape[0]
    state = random.getstate(), np.random.get_state()  # preserve worker RNG
    seed = random.randrange(2**31)
    out = np.empty_like(stack, dtype=np.uint8)
    try:
      for i in range(c):
        random.seed(seed)
        np.random.seed(seed % (2**32))
        out[i] = self._compose(image=stack[i])['image']
    finally:
      random.setstate(state[0])
      np.random.set_state(state[1])
    return out.astype(np.float32) / 255.0


def _normalize_imagenet(stack: np.ndarray, in_chans: int) -> np.ndarray:
  """Tile ImageNet stats across channel groups of 3 (grayscale MRI)."""
  mean = np.tile(np.array(IMAGENET_MEAN, dtype=np.float32), in_chans // 3 + 1)[
    :in_chans
  ]
  std = np.tile(np.array(IMAGENET_STD, dtype=np.float32), in_chans // 3 + 1)[
    :in_chans
  ]
  return (stack - mean[:, None, None]) / std[:, None, None]


def build_transform(
  augment_cfg: AugmentConfig, split: str, in_chans: int
) -> Callable[[np.ndarray], np.ndarray]:
  """Returns stack-level transform pipeline for 'train' or 'valid'."""
  items = getattr(augment_cfg, split)
  stack_tf = SliceStackTransform(items)

  def apply(stack: np.ndarray) -> np.ndarray:
    out = stack_tf(stack)
    return _normalize_imagenet(out, in_chans)

  return apply
