#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Gradient control toolkit: clipping strategies + gradient noise injection.

All strategies implement :class:`ClipStrategy` and operate on parameter
iterables directly (Strategy pattern), selected via YAML ComponentSpec.
Noise injection follows Neelakantan et al. (2015) with a decaying
schedule relative to global step.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Iterator
from typing import Protocol

import numpy as np
import torch
from torch import nn


class ClipStrategy(Protocol):
  """Interface every clipping strategy fulfils."""

  def clip(self, parameters: Iterable[nn.Parameter]) -> dict[str, float]:
    """Clip gradients in place and report diagnostics.

    Args:
        parameters: Trainable parameters whose ``.grad`` is processed.

    Returns:
        Metrics dict (at minimum ``grad_norm`` pre-clip).
    """


def _param_iter(parameters: Iterable[nn.Parameter]) -> Iterator[nn.Parameter]:
  """Yield parameters carrying a non-None gradient.

  Args:
      parameters: Raw trainable parameter iterable.

  Yields:
      Parameters with gradients attached.
  """
  for p in parameters:
    if p.grad is not None:
      yield p


class NormClipping:
  """Classic L2-norm rescaling to a fixed ceiling (delegates to torch).

  Args:
      max_norm: Target maximum global gradient norm.
      error_if_nonfinite: Raise on NaN/inf instead of silently skipping.
  """

  def __init__(
    self, max_norm: float = 10.0, error_if_nonfinite: bool = False
  ) -> None:
    """Store the clipping ceiling.

    Args:
        max_norm: Maximum allowed global norm.
        error_if_nonfinite: Whether non-finite gradients should raise.
    """
    self.max_norm = max_norm
    self.error_if_nonfinite = error_if_nonfinite

  def clip(self, parameters: Iterable[nn.Parameter]) -> dict[str, float]:
    """Rescale gradients when their global norm exceeds ``max_norm``.

    Args:
        parameters: Trainable parameters.

    Returns:
        Dict with pre-clip ``grad_norm`` and applied ``scale``.
    """
    params = list(_param_iter(parameters))
    total = torch.norm(torch.stack([p.grad.norm(2) for p in params]))
    torch.nn.utils.clip_grad_norm_(
      params, self.max_norm, error_if_nonfinite=self.error_if_nonfinite
    )
    scale = min(1.0, float(self.max_norm) / (float(total) + 1e-12))
    return {'grad_norm': float(total), 'scale': scale}


class PercentileClipping:
  """Clip to the rolling q-th percentile of recent gradient norms.

  Adapts to the natural noise scale of training instead of a hand-tuned
  constant: gradients beyond what the model 'usually' sees get damped,
  typical gradients pass untouched.

  Args:
      percentile: Percentile threshold in ``(0, 100]``.
      history_size: Number of recent steps tracked.
      min_norm: Floor so early tiny norms do not crush learning.
  """

  def __init__(
    self,
    percentile: float = 90.0,
    history_size: int = 200,
    min_norm: float = 1e-2,
  ) -> None:
    """Initialize the rolling statistics window.

    Args:
        percentile: Threshold percentile over recent norms.
        history_size: Window length for the percentile estimate.
        min_norm: Minimum effective clipping norm.
    """
    if not 0 < percentile <= 100:
      raise ValueError('percentile must be in (0, 100]')
    self.percentile = percentile
    self.min_norm = min_norm
    self.history: deque[float] = deque(maxlen=history_size)

  def clip(self, parameters: Iterable[nn.Parameter]) -> dict[str, float]:
    """Scale gradients against the rolling percentile threshold.

    Args:
        parameters: Trainable parameters.

    Returns:
        Diagnostics including current ``threshold`` and ``scale``.
    """
    params = list(_param_iter(parameters))
    if not params:
      return {'grad_norm': 0.0, 'scale': 1.0, 'threshold': 0.0}
    total = torch.norm(torch.stack([p.grad.norm(2) for p in params]))
    norm = float(total)
    threshold = (
      float(np_percentile(list(self.history), self.percentile))
      if len(self.history) >= 5
      else max(norm, self.min_norm)
    )
    threshold = max(threshold, self.min_norm)
    scale = min(1.0, threshold / (norm + 1e-12))
    if scale < 1.0:
      torch.nn.utils.clip_grad_norm_(params, threshold)
    self.history.append(min(norm, 1e6))
    return {'grad_norm': norm, 'scale': scale, 'threshold': threshold}


def np_percentile(values: list[float], q: float) -> float:
  """Compute a percentile without importing numpy at module scope.

  Args:
      values: Sample values.
      q: Desired percentile in ``(0, 100]``.

  Returns:
      Interpolated percentile estimate.
  """
  return float(np.percentile(np.asarray(values, dtype=np.float64), q))


class AdaptiveClipping:
  """Norm ceiling adapted from an EMA of observed gradient norms.

  Ceiling = ``multiplier * ema(grad_norm)``: transient spikes are cut
  hard, sustained distribution shifts move the ceiling smoothly.

  Args:
      multiplier: Safety factor above the EMA.
      ema_decay: Decay for the running norm estimate.
      warmup_norm: Initial ceiling before enough observations accrue.
      warmup_steps: Observations required before trusting the EMA.
  """

  def __init__(
    self,
    multiplier: float = 1.5,
    ema_decay: float = 0.98,
    warmup_norm: float = 10.0,
    warmup_steps: int = 20,
  ) -> None:
    """Configure the adaptive ceiling estimator.

    Args:
        multiplier: Factor above EMA used as ceiling.
        ema_decay: EMA decay rate.
        warmup_norm: Fixed ceiling during warmup.
        warmup_steps: Steps before adaptation kicks in.
    """
    self.multiplier = multiplier
    self.ema_decay = ema_decay
    self.warmup_norm = warmup_norm
    self.warmup_steps = warmup_steps
    self._ema: float | None = None
    self._steps = 0

  def clip(self, parameters: Iterable[nn.Parameter]) -> dict[str, float]:
    """Update the EMA and clip to ``multiplier * ema``.

    Args:
        parameters: Trainable parameters.

    Returns:
        Diagnostics incl. current adaptive ceiling.
    """
    params = list(_param_iter(parameters))
    if not params:
      return {'grad_norm': 0.0, 'scale': 1.0, 'ceiling': 0.0}
    total = float(torch.norm(torch.stack([p.grad.norm(2) for p in params])))
    self._ema = (
      total
      if self._ema is None
      else self.ema_decay * self._ema + (1 - self.ema_decay) * total
    )
    self._steps += 1
    ceiling = (
      self.warmup_norm
      if self._steps < self.warmup_steps
      else self.multiplier * self._ema
    )
    scale = min(1.0, ceiling / (total + 1e-12))
    if scale < 1.0:
      torch.nn.utils.clip_grad_norm_(params, ceiling)
    return {'grad_norm': total, 'scale': scale, 'ceiling': float(ceiling)}


class GradientNoiseInjector:
  """Inject decaying Gaussian noise into gradients (regularization).

  Noise std decays as ``eta / (1 + step)^gamma`` relative to each
  tensor's own norm, encouraging exploration of flatter minima.

  Args:
      eta: Base noise magnitude.
      gamma: Decay exponent (0.55 recommended by the paper).
  """

  def __init__(self, eta: float = 0.01, gamma: float = 0.55) -> None:
    """Store noise schedule constants.

    Args:
        eta: Base scale.
        gamma: Decay exponent.
    """
    if eta <= 0:
      raise ValueError('eta must be positive')
    self.eta = eta
    self.gamma = gamma
    self._step = 0

  def inject(self, parameters: Iterable[nn.Parameter]) -> None:
    """Add scheduled noise to every gradient tensor in place.

    Args:
        parameters: Trainable parameters after backward, before step.
    """
    self._step += 1
    std_scale = self.eta / float((1 + self._step) ** self.gamma)
    for p in _param_iter(parameters):
      noise = torch.randn_like(p.grad) * (p.grad.abs().mean() + 1e-8)
      p.grad.add_(noise, alpha=std_scale)
