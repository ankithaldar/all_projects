#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Runtime adaptivity: batch-size tuning, accumulation scheduling, online HP.

All three are Lightning callbacks with conservative heuristics and hard
safety bounds; each can be disabled by omitting it from the YAML
callback list (Open/Closed - no engine edits required).
"""

from __future__ import annotations

from typing import Any

import lightning.pytorch as pl
import torch

from knee.helpers.logging_utils import get_logger


def _memory_fraction() -> float:
  """Peak-to-total GPU memory usage fraction, or 0.0 without CUDA.

  Returns:
      Fraction in ``[0, 1]``; 0.0 on CPU-only runs.
  """
  if not torch.cuda.is_available():
    return 0.0
  peak = torch.cuda.max_memory_allocated()
  total = torch.cuda.get_device_properties(0).total_memory
  return float(peak / max(total, 1))


class AdaptiveBatchSizeCallback(pl.Callback):
  """Grow the per-study batch size when GPU memory is under-utilized.

  Doubles ``batch_size`` at epoch boundaries while peak memory stays
  below ``target_fraction`` of device memory, bounded by ``max_size``.
  The DataModule must expose ``set_batch_size(int)``.

  Args:
      target_fraction: Peak-memory fraction below which scaling triggers.
      max_size: Hard batch-size ceiling.
      min_epochs_between: Cooldown between growth steps.
  """

  def __init__(
    self,
    target_fraction: float = 0.55,
    max_size: int = 32,
    min_epochs_between: int = 1,
  ) -> None:
    """Validate growth policy parameters.

    Args:
        target_fraction: Utilization threshold for doubling.
        max_size: Ceiling on batch size.
        min_epochs_between: Growth cooldown in epochs.

    Raises:
        ValueError: If thresholds are outside valid ranges.
    """
    super().__init__()
    if not 0 < target_fraction < 1:
      raise ValueError('target_fraction must be in (0, 1)')
    self.target_fraction = target_fraction
    self.max_size = max_size
    self.min_epochs_between = min_epochs_between
    self._last_growth_epoch = -min_epochs_between
    self._log = get_logger('knee.adapt.batch')

  def on_train_epoch_start(
    self, trainer: pl.Trainer, pl_module: pl.LightningModule
  ) -> None:
    """Consider one batch-size growth step per epoch.

    Args:
        trainer: Active trainer.
        pl_module: Unused module handle.
    """
    del pl_module
    epoch = trainer.current_epoch
    if epoch - self._last_growth_epoch < self.min_epochs_between:
      return
    dm = trainer.datamodule
    if dm is None or not hasattr(dm, 'set_batch_size'):
      return
    current = int(
      getattr(dm, 'batch_size', 0)
      or getattr(dm.hparams['dm_cfg'], 'batch_size', 0)
    )
    if (
      _memory_fraction() < self.target_fraction and current * 2 <= self.max_size
    ):
      new_size = current * 2
      dm.set_batch_size(new_size)
      torch.cuda.reset_peak_memory_stats()
      self._last_growth_epoch = epoch
      trainer.reset_train_dataloader()
      self._log.info('epoch %d: batch_size %d -> %d', epoch, current, new_size)


class SmartAccumulationCallback(pl.Callback):
  """Tune gradient accumulation to balance effective batch vs throughput.

  Halves accumulation when memory is tight (risking OOM), doubles it
  when the GPU is idle-ish, always keeping powers of two within
  configured bounds.

  Args:
      low_fraction: Below this peak fraction, accumulate more.
      high_fraction: Above this, accumulate less.
      min_accum / max_accum: Hard clamps.
  """

  def __init__(
    self,
    low_fraction: float = 0.35,
    high_fraction: float = 0.85,
    min_accum: int = 1,
    max_accum: int = 64,
  ) -> None:
    """Store the adjustment band.

    Args:
        low_fraction: Grow threshold.
        high_fraction: Shrink threshold.
        min_accum: Lower clamp.
        max_accum: Upper clamp.

    Raises:
        ValueError: If thresholds are inverted.
    """
    super().__init__()
    if low_fraction >= high_fraction:
      raise ValueError('low_fraction must be < high_fraction')
    self.low = low_fraction
    self.high = high_fraction
    self.min_accum = min_accum
    self.max_accum = max_accum
    self._log = get_logger('knee.adapt.accum')

  def on_train_epoch_end(
    self, trainer: pl.Trainer, pl_module: pl.LightningModule
  ) -> None:
    """Adjust the Trainer's accumulation factor from memory pressure.

    Args:
        trainer: Active trainer.
        pl_module: Unused module handle.
    """
    del pl_module
    frac = _memory_fraction()
    current = int(trainer.accumulate_grad_batches)
    if frac > self.high and current // 2 >= self.min_accum:
      target = current // 2
    elif frac < self.low and current * 2 <= self.max_accum:
      target = current * 2
    else:
      return
    try:
      trainer.accumulate_grad_batches = target
      torch.cuda.reset_peak_memory_stats()
      self._log.info('accumulate_grad_batches %d -> %d', current, target)
    except AttributeError:  # older PL internals: fail safe, keep running
      self._log.warning('cannot adjust accumulation on this Lightning version')


class OnlineHPTuner(pl.Callback):
  """Plateau-driven LR decay (+ optional momentum/wd nudges) at runtime.

  When ``val_macro_auc`` stalls for ``patience`` validations, every
  param-group LR is multiplied by ``lr_decay`` up to ``max_reductions``
  times. Optional gentle weight-decay growth keeps regularization
  relevant as LRs shrink.

  Args:
      patience: Validations without improvement before decaying.
      lr_decay: Multiplicative LR factor.
      max_reductions: Lifetime cap on decay events.
      wd_growth: Additive weight-decay increment per event (relative).
      monitor: Metric name tracked.
  """

  def __init__(
    self,
    patience: int = 2,
    lr_decay: float = 0.5,
    max_reductions: int = 3,
    wd_growth: float = 0.0,
    monitor: str = 'val_macro_auc',
  ) -> None:
    """Initialize plateau bookkeeping.

    Args:
        patience: Stall length triggering a decay event.
        lr_decay: Decay factor applied to all groups.
        max_reductions: Maximum number of events.
        wd_growth: Relative weight-decay increase per event.
        monitor: Metric to watch.
    """
    super().__init__()
    self.patience = patience
    self.lr_decay = lr_decay
    self.max_reductions = max_reductions
    self.wd_growth = wd_growth
    self.monitor = monitor
    self._best: float | None = None
    self._stall = 0
    self._events = 0
    self._log = get_logger('knee.adapt.hp')

  def on_validation_epoch_end(
    self, trainer: pl.Trainer, pl_module: pl.LightningModule
  ) -> None:
    """Evaluate the plateau condition and apply one tuning step.

    Args:
        trainer: Trainer whose optimizers are adjusted.
        pl_module: Unused module handle.
    """
    del pl_module
    score = trainer.callback_metrics.get(self.monitor)
    if score is None:
      return
    score = float(score)
    if self._best is None or score > self._best:
      self._best = score
      self._stall = 0
      return
    self._stall += 1
    if self._stall < self.patience or self._events >= self.max_reductions:
      return
    optimizers: list[Any] = (
      trainer.optimizers
      if isinstance(trainer.optimizers, list)
      else [trainer.optimizers]
    )
    for opt in optimizers:
      inner = getattr(opt, 'optimizer', opt)  # unwrap Lookahead
      for group in inner.param_groups:
        group['lr'] *= self.lr_decay
        if self.wd_growth and 'weight_decay' in group:
          group['weight_decay'] *= 1.0 + self.wd_growth
    self._events += 1
    self._stall = 0
    self._log.info(
      'plateau #%d: lrs scaled by %.2f', self._events, self.lr_decay
    )
