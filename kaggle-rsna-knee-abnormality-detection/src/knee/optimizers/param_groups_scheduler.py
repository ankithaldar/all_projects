#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Differential-LR parameter groups + warmup-cosine schedule.

The student trains a pretrained encoder far below the head LR
(backbone x ``backbone_lr_scale``, default 0.25) and anneals all groups
with a short linear warmup into a cosine decay. The scheduler is stepped
manually by :meth:`KneeStudyLitModule.on_train_epoch_start` with a
fractional epoch so epoch-level semantics are exact.
"""

from __future__ import annotations

import math

from torch import nn


def build_param_groups(
  model: nn.Module,
  base_lr: float,
  weight_decay: float = 1e-2,
  backbone_lr_scale: float = 0.25,
) -> list[dict]:
  """Split model parameters into backbone vs head learning-rate groups.

  Args:
      model: Composite study model; parameters named ``encoder.*`` form
          the backbone group.
      base_lr: Head (and global baseline) learning rate.
      weight_decay: Weight decay shared by both groups.
      backbone_lr_scale: Multiplier applied to ``base_lr`` for the
          pretrained encoder.

  Returns:
      Two param-group dicts ready for an optimizer constructor.
  """
  backbone, head = [], []
  for name, param in model.named_parameters():
    if not param.requires_grad:
      continue
    target = backbone if name.startswith('encoder.') else head
    target.append(param)
  return [
    {
      'params': backbone or head,
      'lr': base_lr * backbone_lr_scale,
      'weight_decay': weight_decay,
    },
    {'params': head, 'lr': base_lr, 'weight_decay': weight_decay},
  ]


class WarmupCosineScheduler:
  """Linear warmup followed by cosine annealing to a floor fraction.

  Works on raw optimizer objects (also Lookahead wrappers: their
  param_groups dicts are shared with the inner optimizer).

  Args:
      optimizer: Any object exposing ``param_groups``.
      warmup_epochs: Linear ramp length in epochs.
      min_lr_scale: Final LR as a fraction of each group's base LR.
      total_epochs: Total training epochs used by the cosine tail.
  """

  def __init__(
    self,
    optimizer,
    warmup_epochs: float = 1.0,
    min_lr_scale: float = 0.01,
    total_epochs: int = 12,
  ) -> None:
    if warmup_epochs < 0:
      raise ValueError('warmup_epochs must be non-negative')
    if not 0 < min_lr_scale <= 1:
      raise ValueError('min_lr_scale must lie in (0, 1]')
    self.optimizer = optimizer
    self.warmup_epochs = max(warmup_epochs, 0.0)
    self.min_lr_scale = min_lr_scale
    self.total_epochs = max(int(total_epochs), 1)
    self.base_lrs = [g['lr'] for g in optimizer.param_groups]
    self.last_epoch = -1

  def _factor(self, epoch: float) -> float:
    """LR multiplier at a fractional epoch."""
    if self.warmup_epochs > 0 and epoch < self.warmup_epochs:
      return max(epoch, 0.0) / self.warmup_epochs
    span = max(self.total_epochs - self.warmup_epochs, 1e-6)
    progress = min(max(epoch - self.warmup_epochs, 0.0) / span, 1.0)
    cos = 0.5 * (1.0 + math.cos(math.pi * progress))
    return self.min_lr_scale + (1.0 - self.min_lr_scale) * cos

  def step(self, epoch: float) -> None:
    """Set every group's LR for the given fractional epoch.

    Args:
        epoch: Current epoch as a float (e.g. ``epoch + 0.5``).
    """
    factor = self._factor(float(epoch))
    groups = self.optimizer.param_groups
    for group, base in zip(groups, self.base_lrs, strict=True):
      group['lr'] = base * factor
    self.last_epoch = int(epoch)

  def get_last_lr(self) -> list[float]:
    """Return current per-group LRs (Lightning LR-monitor friendly)."""
    return [g['lr'] for g in self.optimizer.param_groups]
