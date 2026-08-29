#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Lightning callbacks: session budget enforcement and artifact pushing."""

from __future__ import annotations

import os
import time

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

from knee.helpers.kaggle_io import KaggleDatasetClient
from knee.helpers.utils import get_logger

_LOGGER = get_logger(__name__)


class TimeBudgetCallback(Callback):
  """Stop fitting before the kernel's wall-clock session limit.

  The check runs at epoch boundaries (plus optional step granularity) so a
  checkpoint is always persisted in a consistent state by the framework.
  """

  def __init__(
    self,
    session_time_budget_h: float,
    time_margin_min: float = 30.0,
  ) -> None:
    """Configure the budget.

    Args:
        session_time_budget_h: Total allowed wall-clock hours.
        time_margin_min: Reserve stopped for save + dataset push.
    """
    super().__init__()
    self.budget_seconds = session_time_budget_h * 3600.0
    self.margin_seconds = time_margin_min * 60.0
    self._start = 0.0

  def on_fit_start(
    self, trainer: pl.Trainer, pl_module: pl.LightningModule
  ) -> None:
    """Record the fit start time.

    Args:
        trainer: Active trainer.
        pl_module: Active module.
    """
    self._start = time.time()

  def on_train_epoch_end(
    self,
    trainer: pl.Trainer,
    pl_module: pl.LightningModule,
  ) -> None:
    """Request graceful stop once the safe budget is exhausted.

    Args:
        trainer: Active trainer.
        pl_module: Active module.
    """
    elapsed = time.time() - self._start
    remaining = self.budget_seconds - self.margin_seconds - elapsed
    if remaining <= 0:
      _LOGGER.warning(
        'Time budget reached (%.1f min elapsed); stopping after this epoch',
        elapsed / 60.0,
      )
      trainer.should_stop = True


class PeriodicPushCallback(Callback):
  """Persist fold checkpoints and push them to Kaggle periodically."""

  def __init__(
    self,
    checkpoint_dir: str,
    fold_id: int,
    push_every_n_epochs: int,
    client: KaggleDatasetClient | None,
    push_slug: str | None,
    mark_done_on_train_end: bool = True,
  ) -> None:
    """Compose the callback.

    Args:
        checkpoint_dir: Root folder holding ``fold{k}/last.ckpt`` files.
        fold_id: Fold handled by the active training run.
        push_every_n_epochs: Epochs between remote pushes (0 disables).
        client: Kaggle client; None keeps checkpoints purely local.
        push_slug: Dataset slug receiving pushed versions.
        mark_done_on_train_end: Write the ``done`` marker when fit ends.
    """
    super().__init__()
    self.fold_dir = os.path.join(checkpoint_dir, f'fold{fold_id}')
    self.push_every_n_epochs = push_every_n_epochs
    self.client = client
    self.push_slug = push_slug
    self.mark_done_on_train_end = mark_done_on_train_end
    os.makedirs(self.fold_dir, exist_ok=True)

  @property
  def last_ckpt_path(self) -> str:
    """Return the canonical last-checkpoint path for this fold.

    Returns:
        Absolute path of ``fold{k}/last.ckpt``.
    """
    return os.path.join(self.fold_dir, 'last.ckpt')

  def on_validation_epoch_end(
    self,
    trainer: pl.Trainer,
    pl_module: pl.LightningModule,
  ) -> None:
    """Save the rolling last checkpoint every epoch.

    Args:
        trainer: Active trainer.
        pl_module: Active module.
    """
    trainer.save_checkpoint(self.last_ckpt_path)

  def _push(self) -> None:
    """Push the checkpoint directory when configured.

    Raises nothing: push failures are logged and left to the next session
    so a flaky network cannot destroy finished work.
    """
    if self.client is None or not self.push_slug:
      return
    try:
      self.client.push_version(self.push_slug, os.path.dirname(self.fold_dir))
      _LOGGER.info('Pushed checkpoint version to %s', self.push_slug)
    except RuntimeError as exc:
      _LOGGER.error('Checkpoint push failed (%s); local copy retained', exc)

  def on_train_epoch_end(
    self,
    trainer: pl.Trainer,
    pl_module: pl.LightningModule,
  ) -> None:
    """Trigger periodic pushes.

    Args:
        trainer: Active trainer.
        pl_module: Active module.
    """
    if (
      self.push_every_n_epochs > 0
      and trainer.current_epoch > 0
      and (trainer.current_epoch + 1) % self.push_every_n_epochs == 0
    ):
      self._push()

  def on_train_end(
    self, trainer: pl.Trainer, pl_module: pl.LightningModule
  ) -> None:
    """Final save, done-marker write (completed runs only), and push.

    The ``done`` marker must only exist when the fold trained through
    its LAST planned epoch. A session-budget stop raises
    ``trainer.should_stop`` mid-schedule; marking the fold done there
    would make the next session SKIP the remainder of its training and
    let a partially trained model slip into the inference ensemble.

    Args:
        trainer: Active trainer.
        pl_module: Active module.
    """
    trainer.save_checkpoint(self.last_ckpt_path)
    # PL counts COMPLETED epochs: a finished max_epochs=4 run reports
    # current_epoch == 4 at on_train_end; a budget stop after epoch 0
    # of 2 reports 1. "Reached the schedule" therefore means
    # current_epoch >= max_epochs, NOT >= max_epochs - 1.
    max_epochs = int(getattr(trainer, 'max_epochs', 0) or 0)
    reached_last_epoch = (
      trainer.current_epoch >= max_epochs if max_epochs > 0 else True
    )
    if self.mark_done_on_train_end and reached_last_epoch:
      with open(
        os.path.join(self.fold_dir, 'done'), 'w', encoding='utf-8'
      ) as handle:
        handle.write(str(trainer.global_step))
    else:
      _LOGGER.info(
        'fold incomplete (epoch %d/%s); no done marker - the next '
        'session will resume this fold',
        trainer.current_epoch,
        max_epochs,
      )
    self._push()
