#!/usr/bin/env python
"""Lightning callbacks: EMA weight averaging + Discord run updates."""

from __future__ import annotations

import copy
import time
import traceback
from typing import Any

import lightning.pytorch as pl
import torch
from torch import nn

from knee.loggers.discord_notifier import DiscordNotifier


class EMACallback(pl.Callback):
  """Exponential moving average of model weights.

  Shadow weights swap in for validation (so checkpoints and logged
  metrics reflect the EMA) and are persisted in checkpoint state for
  clean resume. Guarded against PL's pre-training sanity check.
  """

  def __init__(
    self, decay: float = 0.999, swap_at_validation: bool = True
  ) -> None:
    super().__init__()
    if not 0.0 <= decay < 1.0:
      raise ValueError('decay must be in [0, 1)')
    self.decay = decay
    self.swap_at_validation = swap_at_validation
    self._shadow: dict[str, torch.Tensor] | None = None
    self._live_backup: dict[str, torch.Tensor] | None = None

  # ------------------------------------------------------------------ #
  @staticmethod
  def _module(pl_module: pl.LightningModule) -> nn.Module:
    return pl_module

  def on_train_batch_end(
    self,
    unused_trainer,
    pl_module,
    unused_outputs,
    unused_batch,
    unused_batch_idx,
  ) -> None:
    live = self._module(pl_module).state_dict()
    if self._shadow is None:
      self._shadow = {k: v.detach().clone() for k, v in live.items()}
      return
    with torch.no_grad():
      for k, v in live.items():
        if v.dtype.is_floating_point:
          self._shadow[k].mul_(self.decay).add_(
            v.detach(), alpha=1.0 - self.decay
          )
        else:
          self._shadow[k] = v.detach().clone()  # buffers (BN stats etc.)

  # ------------------------- validation swap ------------------------ #
  def on_validation_epoch_start(self, unused_trainer, pl_module) -> None:
    if not self.swap_at_validation or self._shadow is None:
      return
    module = self._module(pl_module)
    self._live_backup = copy.deepcopy(module.state_dict())
    module.load_state_dict(self._shadow)

  def on_validation_epoch_end(self, unused_trainer, pl_module) -> None:
    if self._live_backup is None:
      return
    self._module(pl_module).load_state_dict(self._live_backup)
    self._live_backup = None

  # --------------------------- persistence -------------------------- #
  def on_save_checkpoint(
    self, unused_trainer, unused_pl_module, checkpoint
  ) -> None:
    checkpoint['ema_shadow'] = self._shadow

  def on_load_checkpoint(self, unused_trainer, pl_module, checkpoint) -> None:
    shadow = checkpoint.get('ema_shadow')
    if shadow is not None:
      self._shadow = {
        k: v.to(next(pl_module.parameters()).device) for k, v in shadow.items()
      }


class DiscordCallback(pl.Callback):
  """Posts fit lifecycle + metric updates to a Discord webhook.

  Reads DISCORD_WEBHOOK_URL from .env / env vars / Kaggle Secrets via
  DiscordNotifier; silently no-ops when unset. All sends are
  fire-and-forget -- training continues regardless of network state.
  """

  def __init__(
    self,
    notify_every_epochs: int = 1,
    report_best: bool = True,
    include_train_loss: bool = False,
  ) -> None:
    super().__init__()
    if notify_every_epochs < 1:
      raise ValueError('notify_every_epochs must be >= 1')
    self.notify_every_epochs = notify_every_epochs
    self.report_best = report_best
    self.include_train_loss = include_train_loss
    self.notifier = DiscordNotifier()
    self._best_score: float | None = None
    self._t_start: float | None = None

  # ----------------------------- helpers ---------------------------- #
  def _run_label(self, trainer: pl.Trainer) -> str:
    name = type(trainer.logger).__name__ if trainer.logger else 'run'
    version = getattr(trainer.logger, 'version', '')
    experiment = trainer.default_root_dir.rstrip('/').split('/')[-1]
    return f'{experiment} [{name}:{version}]'

  def _collect_metrics(self, trainer: pl.Trainer) -> dict[str, Any]:
    m = {k: v for k, v in trainer.callback_metrics.items()}
    if not self.include_train_loss:
      m.pop('train_loss', None)
    return m

  # ----------------------------- hooks ------------------------------ #
  def on_fit_start(self, trainer, pl_module) -> None:
    if not trainer.is_global_zero:
      return  # DDP: notify once, from rank 0 only
    self._t_start = time.monotonic()
    n_params = sum(p.numel() for p in pl_module.parameters()) / 1e6
    fold = getattr(trainer.datamodule, 'fold', 'n/a')
    self.notifier.send(
      f'[START] {self._run_label(trainer)}\n'
      f'fold(s): {fold} | max_epochs: {trainer.max_epochs} '
      f'| params: {n_params:.1f}M'
    )

  def on_validation_epoch_end(self, trainer, unused_pl_module) -> None:
    if not trainer.is_global_zero:
      return  # DDP: notify once, from rank 0 only
    epoch = trainer.current_epoch
    improved = False
    score = trainer.callback_metrics.get('val_macro_auc')
    if self.report_best and score is not None:
      val = float(score)
      if self._best_score is None or val > self._best_score:
        self._best_score = val
        improved = True
    if epoch % self.notify_every_epochs != 0 and not improved:
      return
    tag = '[BEST]' if improved else '[EPOCH]'
    body = self.notifier.fmt_metrics(self._collect_metrics(trainer))
    self.notifier.send(
      f'{tag} {self._run_label(trainer)} epoch {epoch}\n{body}'
    )

  def on_exception(
    self, trainer, unused_pl_module, exception: BaseException
  ) -> None:
    tb_tail = ''.join(traceback.format_exception(exception)[-600:])
    self.notifier.send(
      f'[CRASH] {self._run_label(trainer)} at step {trainer.global_step}\n'
      f'{type(exception).__name__}: {exception}\n```{tb_tail}```',
      force=True,
    )

  def on_fit_end(self, trainer, unused_pl_module) -> None:
    if not trainer.is_global_zero:
      return  # DDP: notify once, from rank 0 only
    mins = (time.monotonic() - (self._t_start or time.monotonic())) / 60
    best = (
      f'\nbest val_macro_auc: {self._best_score:.4f}'
      if self._best_score is not None
      else ''
    )
    self.notifier.send(
      f'[DONE] {self._run_label(trainer)} after {mins:.1f} min{best}'
    )
