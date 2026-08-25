#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Multi-label classification losses.

Two members, both consumed through ``loss.criterion`` ComponentSpecs:

- :class:`AsymmetricLoss` (ASL, Ben-Baruch et al. 2021) -- the supervised
  criterion for hard gold labels; down-weights easy negatives via
  ``gamma_neg`` and applies probability-margin shifting via ``clip``.
- :class:`SoftBCEWithLogits` -- temperature-scaled BCE against soft
  teacher probabilities (the distillation term).

Both return per-sample scalars with ``reduction='none'`` because
:class:`knee.engines.study_lit_module.KneeStudyLitModule` weights each
study by its label-trust weight before pooling the batch loss.
"""

from __future__ import annotations

import torch
from torch import nn


class AsymmetricLoss(nn.Module):
  """ASL for multi-label targets: focal on both sides + negative margin.

  Args:
      gamma_neg: Focusing exponent applied to easy negatives.
      gamma_pos: Focusing exponent applied to positives.
      clip: Probability shift for negatives (acts as a hard threshold).
      eps: Numerical floor inside logarithms.

  Raises:
      ValueError: If gammas are negative or clip is outside [0, 1).
  """

  def __init__(
    self,
    gamma_neg: float = 2.0,
    gamma_pos: float = 0.2,
    clip: float = 0.05,
    eps: float = 1e-6,
  ) -> None:
    super().__init__()
    if gamma_neg < 0 or gamma_pos < 0:
      raise ValueError('gammas must be non-negative')
    if not 0 <= clip < 1:
      raise ValueError('clip must lie in [0, 1)')
    self.gamma_neg = gamma_neg
    self.gamma_pos = gamma_pos
    self.clip = clip
    self.eps = eps

  def forward(
    self,
    logits: torch.Tensor,
    targets: torch.Tensor,
    reduction: str = 'none',
  ) -> torch.Tensor:
    """Compute ASL between logits and binary targets.

    Args:
        logits: Raw model outputs ``(B, C)``.
        targets: Binary targets ``(B, C)``, broadcastable.
        reduction: 'none' | 'mean' | 'sum'; 'none' yields per-sample
            scalars (mean over classes).

    Returns:
        Reduced loss tensor.
    """
    x_sigmoid = torch.sigmoid(logits)
    xs_pos = x_sigmoid
    xs_neg = 1.0 - x_sigmoid
    if self.clip > 0:
      xs_neg = (xs_neg + self.clip).clamp(max=1.0)
    los_pos = targets * torch.log(xs_pos.clamp_min(self.eps))
    los_neg = (1.0 - targets) * torch.log(xs_neg.clamp_min(self.eps))
    loss = -(los_pos + los_neg)
    if self.gamma_neg > 0 or self.gamma_pos > 0:
      focal = torch.pow(1.0 - xs_pos, self.gamma_pos) * targets + torch.pow(
        xs_neg, self.gamma_neg
      ) * (1.0 - targets)
      loss = loss * focal
    per_sample = loss.mean(dim=-1)
    if reduction == 'mean':
      return per_sample.mean()
    if reduction == 'sum':
      return per_sample.sum()
    return per_sample


class SoftBCEWithLogits(nn.Module):
  """Temperature-scaled BCE against soft probability targets.

  Distillation term of the student: logits are divided by ``T`` before
  sigmoid so teacher confidence is softened consistently.

  Args:
      temperature: Divisor applied to logits (>0); T=2 per BLUEPRINT.
      eps: Clamp range edge for probability stability.
  """

  def __init__(self, temperature: float = 2.0, eps: float = 1e-6) -> None:
    """Validate and store the temperature.

    Args:
        temperature: Positive logit divisor.
        eps: Numerical clamp epsilon.

    Raises:
        ValueError: If temperature is not positive.
    """
    super().__init__()
    if temperature <= 0:
      raise ValueError('temperature must be positive')
    self.temperature = temperature
    self.eps = eps

  def forward(
    self,
    logits: torch.Tensor,
    targets: torch.Tensor,
    reduction: str = 'none',
  ) -> torch.Tensor:
    """Compute elementwise BCE(sigmoid(logits / T), soft targets).

    Args:
        logits: Raw model outputs ``(B, C)``.
        targets: Soft probabilities ``(B, C)`` in [0, 1].
        reduction: 'none' | 'mean' | 'sum'.

    Returns:
        Reduced loss tensor; 'none' is per-sample (mean over classes).
    """
    probs = torch.sigmoid(logits / self.temperature)
    probs = probs.clamp(self.eps, 1.0 - self.eps)
    loss = -(
      targets * torch.log(probs) + (1.0 - targets) * torch.log(1.0 - probs)
    )
    per_sample = loss.mean(dim=-1)
    if reduction == 'mean':
      return per_sample.mean()
    if reduction == 'sum':
      return per_sample.sum()
    return per_sample
