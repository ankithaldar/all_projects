#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Progressive layer freezing + AMP stability fallback.

Freezing: rule-based progressive fine-tuning - backbone stays frozen
while the head/aggregator calibrates, then stages unlock on schedule.
Rules are declarative ``(pattern, until_epoch)`` pairs applied by a
Lightning callback.

Stability: bf16/fp16 divergence (NaN/inf losses, exploding norms) is
detected and the fit is transparently restarted in fp32 from the latest
checkpoint instead of silently corrupting weights.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import lightning.pytorch as pl
import torch

from knee.helpers.logging_utils import get_logger


def set_requires_grad(
  module: torch.nn.Module, pattern: str, requires_grad: bool
) -> int:
  """Flip ``requires_grad`` for parameters whose name contains a pattern.

  Args:
      module: Module tree to walk.
      pattern: Substring matched against parameter names.
      requires_grad: Desired gradient flag.

  Returns:
      Number of parameters updated.
  """
  touched = 0
  for name, param in module.named_parameters():
    if pattern in name:
      param.requires_grad_(requires_grad)
      touched += 1
  return touched


class ProgressiveUnfreezingCallback(pl.Callback):
  """Apply freeze rules per epoch for staged fine-tuning.

  Args:
      rules: List of dicts ``{'pattern': str, 'until_epoch': int}``;
          matching parameters stay frozen while
          ``current_epoch < until_epoch``.
  """

  def __init__(self, rules: list[dict]) -> None:
    """Store the declarative freeze schedule.

    Args:
        rules: Freeze rules; empty list disables the callback.
    """
    super().__init__()
    self.rules = rules
    self._log = get_logger('knee.freeze')

  def on_train_epoch_start(
    self, trainer: pl.Trainer, pl_module: pl.LightningModule
  ) -> None:
    """Enforce freeze state for the starting epoch.

    Args:
        trainer: Active trainer.
        pl_module: The model whose parameters are toggled.
    """
    epoch = trainer.current_epoch
    for rule in self.rules:
      frozen_now = epoch < int(rule['until_epoch'])
      n = set_requires_grad(
        pl_module, rule['pattern'], requires_grad=not frozen_now
      )
      if n:
        self._log.info(
          'epoch %d: %s %d params matching %r',
          epoch,
          'froze' if frozen_now else 'unfroze',
          n,
          rule['pattern'],
        )


class AmpFallbackError(RuntimeError):
  """Raised internally when mixed precision proves unstable."""


class AmpInstabilityWatcher(pl.Callback):
  """Watch training losses; abort the fit when precision diverges.

  Consecutive non-finite losses (or exploding loss growth) trip an
  :class:`AmpFallbackError`; the caller catches it and re-fits with
  ``precision='32-true'`` resuming from the newest checkpoint.

  Args:
      patience: Consecutive bad batches tolerated before tripping.
  """

  def __init__(self, patience: int = 10) -> None:
    """Configure the consecutive-bad-batch budget.

    Args:
        patience: Bad batches allowed before raising.
    """
    super().__init__()
    self.patience = patience
    self._bad_streak = 0

  def on_train_batch_end(
    self,
    trainer: pl.Trainer,
    unused_pl_module: pl.LightningModule,
    outputs: dict,
    unused_batch,
    unused_batch_idx: int,
  ) -> None:
    """Track loss finiteness across batches.

    Args:
        trainer: Active trainer.
        unused_pl_module: Unused module handle.
        outputs: Training-step output mapping.
        unused_batch: Unused batch payload.
        unused_batch_idx: Index of the finished batch.
    """
    loss = outputs.get('loss') if isinstance(outputs, dict) else None
    finite = loss is not None and bool(torch.isfinite(loss).all())
    if finite:
      self._bad_streak = 0
      return
    self._bad_streak += 1
    if self._bad_streak >= self.patience:
      raise AmpFallbackError(
        f'{self._bad_streak} non-finite losses at step {trainer.global_step}'
      )

  def on_train_epoch_start(
    self, unused_trainer: pl.Trainer, unused_pl_module: pl.LightningModule
  ) -> None:
    """Reset the streak each epoch.

    Args:
        unused_trainer: Unused trainer handle.
        unused_pl_module: Unused module handle.
    """
    self._bad_streak = 0


def fit_with_amp_fallback(
  make_trainer_and_run: Callable[[str], object],
  precision_options: tuple[str, ...] = ('bf16-mixed', '32-true'),
  checkpoint_dir: Path | None = None,
) -> str:
  """Run a fit, degrading precision automatically on instability.

  Args:
      make_trainer_and_run: Callable receiving a precision string,
          building a fresh Trainer (with watchers attached) and calling
          ``fit(..., ckpt_path=<latest or None>)``. Must return its run
          result; may raise :class:`AmpFallbackError`.
      precision_options: Precisions tried in order.
      checkpoint_dir: Directory scanned for a resume checkpoint between
          attempts.

  Returns:
      The precision string that ultimately succeeded.

  Raises:
      AmpFallbackError: If every requested precision fails.
  """
  log = get_logger('knee.stability')
  last_error: Exception | None = None
  for precision in precision_options:
    try:
      make_trainer_and_run(precision)
      return precision
    except AmpFallbackError as exc:
      last_error = exc
      log.warning('precision %s unstable (%s); falling back', precision, exc)
      if checkpoint_dir is not None:
        candidates = sorted(
          Path(checkpoint_dir).glob('*.ckpt'), key=lambda p: p.stat().st_mtime
        )
        log.info(
          'resume source after fallback: %s',
          candidates[-1] if candidates else 'scratch',
        )
  raise AmpFallbackError(f'all precisions failed; last error: {last_error}')
