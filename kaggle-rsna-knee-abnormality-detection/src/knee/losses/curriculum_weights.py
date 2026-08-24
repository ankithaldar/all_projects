#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Curriculum learning controller + dynamic loss weighting.

Curriculum: task difficulty grows over training. For weak supervision the
natural axis is label trust - early epochs train on gold/high-confidence
seeds only; soft teacher labels enter as the student matures.

Dynamic weighting: DWA (Dynamic Weight Averaging, Liu et al. 2019)
balances the supervised and distillation terms by their relative rates
of change, replacing the hand-tuned ramp.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np


class CurriculumController(Protocol):
  """Interface for difficulty schedules consumed by the LightningModule."""

  def weight_floor(self, epoch: int) -> float:
    """Minimum ``sample_weight`` a study needs to contribute this epoch.

    Args:
        epoch: Current (zero-based) training epoch.

    Returns:
        Floor in ``[0, 1]``; 0.0 means everything participates.
    """


class ConfidenceRampCurriculum:
  """Ramps the sample-weight floor downward: easy first, hard later.

  Epochs ``0..hold_epochs`` keep only gold/rule seeds (weight >= floor);
  afterwards the floor decays linearly to zero so soft-label studies join
  gradually, ordered implicitly by their builder-assigned weights.

  Args:
      start_floor: Initial weight floor.
      hold_epochs: Epochs at full strictness.
      ramp_epochs: Epochs over which the floor reaches zero after hold.
  """

  def __init__(
    self, start_floor: float = 0.9, hold_epochs: int = 2, ramp_epochs: int = 4
  ) -> None:
    """Validate and store the schedule shape.

    Args:
        start_floor: Floor used during the hold phase.
        hold_epochs: Length of the strict phase.
        ramp_epochs: Length of the decay phase.

    Raises:
        ValueError: If any parameter is outside its valid range.
    """
    if not 0 <= start_floor <= 1:
      raise ValueError('start_floor must be within [0, 1]')
    if hold_epochs < 0 or ramp_epochs < 0:
      raise ValueError('epochs must be non-negative')
    self.start_floor = start_floor
    self.hold_epochs = hold_epochs
    self.ramp_epochs = max(ramp_epochs, 1)

  def weight_floor(self, epoch: int) -> float:
    """Compute the floor for an epoch per the hold-then-decay schedule.

    Args:
        epoch: Current epoch index.

    Returns:
        Weight floor for this epoch's batches.
    """
    if epoch < self.hold_epochs:
      return self.start_floor
    progress = min((epoch - self.hold_epochs) / self.ramp_epochs, 1.0)
    return self.start_floor * (1.0 - progress)


class DynamicLossWeighter:
  """DWA balancing of multiple loss terms by their descent rates.

  Weights emphasize the term whose loss is decreasing slowest relative
  to its own history, preventing one term (e.g. distillation) from
  dominating gradient magnitude.

  Args:
      temperature: Softening exponent T in DWA; larger => flatter weights.
      initial_weights: Starting weights per term.
  """

  def __init__(
    self,
    temperature: float = 2.0,
    initial_weights: tuple[float, ...] = (1.0, 1.0),
  ) -> None:
    """Initialize histories with equal-cost baselines.

    Args:
        temperature: DWA temperature (>0).
        initial_weights: Prior weights per tracked loss term.
    """
    if temperature <= 0:
      raise ValueError('temperature must be positive')
    self.temperature = temperature
    self._last_cost = np.ones(len(initial_weights), dtype=np.float64)
    self._weights = np.asarray(initial_weights, dtype=np.float64)

  def update(self, losses: tuple[float, ...]) -> tuple[float, ...]:
    """Recompute weights from this epoch's term losses.

    Args:
        losses: Scalar loss value per tracked term.

    Returns:
        New normalized weights (same order as inputs).
    """
    cost = np.asarray(losses, dtype=np.float64).clip(min=1e-8)
    ratio = cost / self._last_cost
    raw = np.exp(self._weights / self.temperature * -np.log(ratio))
    self._weights = len(losses) * raw / raw.sum()
    self._last_cost = cost
    return tuple(float(w) for w in self._weights)
