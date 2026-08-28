#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Epoch/batch progress surfaced through the production log pipeline.

Lightning's tqdm bar is console-only and disappears under process
launchers; this callback mirrors the same signal into the flat run log
at a configurable cadence, plus GPU memory sampling, so pace and
failures are visible in exactly one place:

    epoch 1 | batch 25/441 | step 25 | train_loss 0.3123 |
    lr 2.5e-04 | gpu 8.1/14.5 GiB
"""

from __future__ import annotations

import time

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

from knee.helpers.utils import get_logger

_LOGGER = get_logger('progress')


class ProgressLogCallback(Callback):
  """Log batch/epoch progress + loss/lr/gpu-memory every N steps."""

  def __init__(
    self,
    log_every_n_steps: int = 25,
    log_gpu_mem: bool = True,
  ) -> None:
    """Configure cadence.

    Args:
        log_every_n_steps: Emit a line every this many optimizer steps.
        log_gpu_mem: Include torch CUDA memory snapshot when available.
    """
    super().__init__()
    self.log_every_n_steps = max(1, int(log_every_n_steps))
    self.log_gpu_mem = log_gpu_mem
    self._epoch_start = 0.0

  # -- helpers ----------------------------------------------------------

  @staticmethod
  def _gpu_mem_gib() -> str:
    """Snapshot GPU memory when CUDA is live.

    Returns:
        'allocated/total GiB' or 'n/a' outside CUDA.
    """
    try:
      import torch  # pylint: disable=import-outside-toplevel

      if not torch.cuda.is_available():
        return 'n/a'
      allocated = torch.cuda.memory_allocated() / 1024**3
      total = torch.cuda.get_device_properties(0).total_memory / 1024**3
      return f'{allocated:.1f}/{total:.1f} GiB'
    except (ImportError, OSError, RuntimeError):
      return 'n/a'

  @staticmethod
  def _batches_total(trainer: pl.Trainer) -> str:
    """Batches per epoch as 'N' or '?' when unknown.

    Args:
        trainer: Active trainer.

    Returns:
        String usable inline in the progress line.
    """
    try:
      return str(int(trainer.num_training_batches))
    except (AttributeError, TypeError, ValueError):
      return '?'

  @staticmethod
  def _fmt(value) -> str:
    """Format an optional metric value for the log line.

    Args:
        value: Metric or None.

    Returns:
        '0.3123' style string, or '-' when absent.
    """
    if value is None:
      return '-'
    try:
      return f'{float(value):.4f}'
    except (TypeError, ValueError):
      return '-'

  # -- hooks ------------------------------------------------------------

  def on_train_epoch_start(
    self, trainer: pl.Trainer, pl_module: pl.LightningModule
  ) -> None:
    """Record epoch start time.

    Args:
        trainer: Active trainer.
        pl_module: Active module.
    """
    del pl_module
    self._epoch_start = time.time()

  def on_train_batch_end(
    self,
    trainer: pl.Trainer,
    pl_module: pl.LightningModule,
    outputs: object,
    batch: object,
    batch_idx: int,
  ) -> None:
    """Emit a progress line every log_every_n_steps optimizer steps.

    Args:
        trainer: Active trainer.
        pl_module: Active module.
        outputs: Step output (unused).
        batch: Current batch (unused).
        batch_idx: Zero-based batch index within the epoch.
    """
    del outputs, batch
    if not trainer.is_global_zero:
      return
    step = int(trainer.global_step)
    if step == 0 or step % self.log_every_n_steps != 0:
      return
    loss = trainer.callback_metrics.get('train/loss')
    lr = None
    try:
      optimizers = pl_module.optimizers()
      items = (
        optimizers if isinstance(optimizers, (list, tuple)) else [optimizers]
      )
      items = [o for o in items if o is not None]
      if items:
        raw = getattr(items[0], 'optimizer', items[0])
        groups = getattr(raw, 'param_groups', [])
        lr = groups[0].get('lr') if groups else None
    except Exception:  # pylint: disable=broad-except
      lr = None
    gpu = self._gpu_mem_gib() if self.log_gpu_mem else 'off'
    _LOGGER.info(
      'epoch %d | batch %d/%s | step %d | train_loss %s | lr %s | gpu %s',
      trainer.current_epoch,
      batch_idx + 1,
      self._batches_total(trainer),
      step,
      self._fmt(loss),
      self._fmt(lr),
      gpu,
    )

  def on_validation_epoch_end(
    self, trainer: pl.Trainer, pl_module: pl.LightningModule
  ) -> None:
    """Log validation metrics when validation produced any.

    Args:
        trainer: Active trainer.
        pl_module: Active module.
    """
    del pl_module
    if not trainer.is_global_zero or trainer.sanity_checking:
      return
    metrics = trainer.callback_metrics
    auc = self._fmt(metrics.get('val/auc_macro'))
    loss = self._fmt(metrics.get('val/loss'))
    _LOGGER.info(
      'epoch %d | validation | val/auc_macro %s | val/loss %s',
      trainer.current_epoch,
      auc,
      loss,
    )

  def on_train_epoch_end(
    self, trainer: pl.Trainer, pl_module: pl.LightningModule
  ) -> None:
    """Close the epoch with duration.

    Args:
        trainer: Active trainer.
        pl_module: Active module.
    """
    del pl_module
    if not trainer.is_global_zero:
      return
    minutes = (time.time() - self._epoch_start) / 60.0
    _LOGGER.info(
      'epoch %d | train epoch complete in %.1f min | gpu %s',
      trainer.current_epoch,
      minutes,
      self._gpu_mem_gib() if self.log_gpu_mem else 'off',
    )


__all__ = ['ProgressLogCallback']
