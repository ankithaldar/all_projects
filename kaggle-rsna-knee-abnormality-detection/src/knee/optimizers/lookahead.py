#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Lookahead optimizer wrapper (Zhang et al., 2019).

Maintains slow weights interpolated every ``k`` fast steps, improving
stability across heterogeneous weak-label batches without changing the
inner optimizer's per-step semantics. ``param_groups`` dicts are shared
with the inner optimizer, so LR schedulers mutating group ``lr`` remain
fully effective.
"""

from __future__ import annotations

from typing import Any

import torch
from torch.optim import Optimizer


class Lookahead(Optimizer):
  """Wrap any base optimizer with slow-weight lookahead.

  Args:
      optimizer: Fast inner optimizer (AdamW, SGD, ...).
      k: Fast steps between slow-weight syncs.
      alpha: Slow-weight interpolation toward fast weights.

  Raises:
      ValueError: If ``k < 1`` or ``alpha`` outside ``(0, 1]``.
  """

  def __init__(
    self, optimizer: Optimizer, k: int = 5, alpha: float = 0.5
  ) -> None:
    """Snapshot initial slow weights and share param-group dicts.

    Args:
        optimizer: Inner optimizer to wrap.
        k: Sync interval in fast steps.
        alpha: Interpolation factor in ``(0, 1]``.
    """
    if k < 1:
      raise ValueError('k must be >= 1')
    if not 0 < alpha <= 1:
      raise ValueError('alpha must be in (0, 1]')
    self.optimizer = optimizer
    self.k = k
    self.alpha = alpha
    self._step_count = 0
    self._slow: list[list[torch.Tensor]] = [
      [p.detach().clone() for p in group['params']]
      for group in optimizer.param_groups
    ]
    super().__init__(optimizer.param_groups, {'k': k, 'alpha': alpha})

  @torch.no_grad()
  def step(self, closure: Any = None) -> Any:  # type: ignore[override]
    """Advance the fast optimizer; sync slow weights every k-th step.

    On sync each parameter moves to ``slow = slow*(1-alpha) + fast*alpha``
    and the fast weights are pulled back to the slow values.

    Args:
        closure: Optional closure forwarded to the inner optimizer.

    Returns:
        The inner optimizer's step return value.
    """
    loss = self.optimizer.step(closure)
    self._step_count += 1
    if self._step_count % self.k == 0:
      for slow_group, group in zip(
        self._slow, self.optimizer.param_groups, strict=True
      ):
        for slow, p in zip(slow_group, group['params'], strict=True):
          slow.mul_(1.0 - self.alpha).add_(p.detach(), alpha=self.alpha)
          p.copy_(slow)
    return loss

  def zero_grad(self, set_to_none: bool = True) -> None:
    """Delegate gradient zeroing to the inner optimizer.

    Args:
        set_to_none: Forwarded unchanged.
    """
    self.optimizer.zero_grad(set_to_none=set_to_none)
