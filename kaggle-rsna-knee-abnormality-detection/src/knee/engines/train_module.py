#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Lightning training module for KneeNet."""

from __future__ import annotations

import importlib
import math
import os

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch import Tensor, nn

from knee.metrics.auc import MultilabelAUC
from knee.models.knee_net import KneeNet


def build_step_schedules(
  optimizer,
  scheduler_cfg: dict | None,
  warmup_epochs: float,
  total_epochs: int,
  steps_per_epoch: int,
  accumulate_grad_batches: int = 1,
):
  """Step-based warmup + cosine schedule from epoch-denominated config.

  Historical bug: LinearLR/SequentialLR ran with ``interval='epoch'``,
  so ``warmup_epochs: 1`` held LR at 5% of base for the ENTIRE first
  epoch (~235 steps) - the model silently failed to learn while the
  progress line displayed ``lr 0.0000``.

  Args:
      optimizer: Configured optimizer instance.
      scheduler_cfg: ``optimizer.scheduler`` section (may be None).
      warmup_epochs: Warmup length in EPOCHS from config.
      total_epochs: Trainer max epochs.
      steps_per_epoch: Optimizer steps per epoch (post-accumulation).
      accumulate_grad_batches: Trainer accumulation (divides batches).

  Returns:
      Lightning scheduler dict, or None when no scheduler configured.
  """
  if scheduler_cfg is None:
    return None
  accum = max(1, int(accumulate_grad_batches))
  opt_steps_per_epoch = max(1, int(steps_per_epoch) // accum)
  total_steps = max(1, int(total_epochs) * opt_steps_per_epoch)
  warmup_steps = int(float(warmup_epochs) * opt_steps_per_epoch)
  sched_init = dict(scheduler_cfg.get('init_params', {}))
  if 'T_max' in sched_init:
    # Config denominates T_max in epochs; the step regime needs steps.
    sched_init['T_max'] = max(1, total_steps - warmup_steps)
  sched_cls = _resolve_class(scheduler_cfg['class_path'])
  cosine = sched_cls(optimizer, **sched_init)
  if warmup_steps <= 0:
    return {'scheduler': cosine, 'interval': 'step'}
  linear = torch.optim.lr_scheduler.LinearLR(
    optimizer, start_factor=0.05, total_iters=max(1, warmup_steps)
  )
  return {
    'scheduler': torch.optim.lr_scheduler.SequentialLR(
      optimizer, [linear, cosine], milestones=[max(1, warmup_steps)]
    ),
    'interval': 'step',
  }


class KneeModule(pl.LightningModule):
  """Wrap KneeNet with loss, optimizer/schedule, metrics, and OOF dumps."""

  def __init__(
    self,
    model: KneeNet,
    criterion: nn.Module,
    optimizer_cfg: dict,
    scheduler_cfg: dict | None,
    warmup_epochs: int,
    backbone_lr_scale: float,
    total_epochs: int,
    target_columns: list[str],
    oof_dir: str,
    fold_id: int,
    steps_per_epoch_hint: int = 0,
  ) -> None:
    """Store all training-time collaborators.

    Args:
        model: Instantiated KneeNet.
        criterion: Masked multi-target loss instance.
        optimizer_cfg: Spec dict with class_path/init_params.
        scheduler_cfg: Optional scheduler spec dict.
        warmup_epochs: Linear warmup length before cosine decay.
        backbone_lr_scale: Multiplier for the backbone param group lr.
        total_epochs: Max epochs used to size the cosine schedule.
        target_columns: Canonical 12-target order.
        oof_dir: Directory receiving ``oof_fold{k}.csv``.
        fold_id: Fold handled by this module instance.
    """
    super().__init__()
    self.model = model
    self.criterion = criterion
    self.optimizer_cfg = optimizer_cfg
    self.scheduler_cfg = scheduler_cfg
    self.warmup_epochs = warmup_epochs
    self.backbone_lr_scale = backbone_lr_scale
    self.total_epochs = max(total_epochs, warmup_epochs + 1)
    self.target_columns = target_columns
    self.oof_dir = oof_dir
    self.steps_per_epoch_hint = int(steps_per_epoch_hint)
    self.fold_id = fold_id
    self.metric = MultilabelAUC(target_columns)

  def forward(self, batch: dict[str, Tensor]) -> Tensor:
    """Delegate to KneeNet.

    Args:
        batch: Collated study batch.

    Returns:
        Raw logits ``(batch, n_targets)``.
    """
    return self.model(batch)

  def training_step(self, batch: dict, batch_idx: int) -> Tensor:
    """Compute optimization loss for one step.

    Args:
        batch: Collated study batch with labels.
        batch_idx: Index of the step within the epoch.

    Returns:
        Scalar loss tensor.
    """
    del batch_idx  # hook signature parity with Lightning
    logits = self.model(batch)
    loss = self.criterion(logits, batch['label'])
    self.log('train/loss', loss, on_step=True, prog_bar=True)
    return loss

  @torch.no_grad()
  def validation_step(self, batch: dict, batch_idx: int) -> None:
    """Accumulate validation predictions and targets.

    Args:
        batch: Collated study batch with labels.
        batch_idx: Index of the step within the epoch.
    """
    del batch_idx  # hook signature parity with Lightning
    probs = torch.sigmoid(self.model(batch))
    self.metric.update(
      probs.float().cpu().numpy(),
      batch['label'].cpu().numpy(),
      list(batch['study_uid']),
    )

  def on_validation_epoch_end(self) -> None:
    """Log per-class AUCs and persist the OOF table.

    The OOF csv is rewritten each epoch (small; thousands of rows) so an
    interrupted session still leaves usable predictions behind.
    """
    summary = self.metric.summary()
    self.log('val/auc_macro', summary['auc/macro'], prog_bar=True)
    for name, value in summary.items():
      if name != 'auc/macro' and not np.isnan(value):
        self.log(f'val/{name}', value)
    probs, targets = self.metric.stacked()
    frame = pd.DataFrame(
      probs, columns=[f'{c}_prob' for c in self.target_columns]
    )
    for col, name in enumerate(self.target_columns):
      frame[name] = targets[:, col]
    frame.insert(0, 'StudyInstanceUID', self.metric.study_uids)
    frame.insert(1, 'fold', self.fold_id)
    os.makedirs(self.oof_dir, exist_ok=True)
    frame.to_csv(
      os.path.join(self.oof_dir, f'oof_fold{self.fold_id}.csv'), index=False
    )

  def configure_optimizers(self):
    """Build differential-LR AdamW-style optimizer and warmup+cosine schedule.

    Returns:
        Lightning-compatible optimizer/scheduler configuration.
    """
    init_params = dict(self.optimizer_cfg['init_params'])
    base_lr = float(init_params.pop('lr'))
    groups = [
      {'params': self.model.head.parameters(), 'lr': base_lr},
      {
        'params': self.model.series_encoder.backbone.parameters(),
        'lr': base_lr * self.backbone_lr_scale,
      },
    ]
    remaining = [
      p
      for n, p in self.model.named_parameters()
      if not n.startswith(('head', 'series_encoder.backbone'))
    ]
    if remaining:
      groups.append({'params': remaining, 'lr': base_lr})
    opt_cls = _resolve_class(self.optimizer_cfg['class_path'])
    optimizer = opt_cls(groups, **init_params)
    trainer = getattr(self, 'trainer', None)
    batches = getattr(trainer, 'num_training_batches', 0)
    if not (isinstance(batches, (int, float)) and math.isfinite(batches)):
      batches = 0
    steps_per_epoch = int(self.steps_per_epoch_hint) or int(batches or 0)
    accumulate = int(getattr(trainer, 'accumulate_grad_batches', 1) or 1)
    total_epochs = int(getattr(trainer, 'max_epochs', 1) or 1)
    schedulers = []
    schedule = build_step_schedules(
      optimizer,
      self.scheduler_cfg,
      self.warmup_epochs,
      total_epochs,
      steps_per_epoch,
      accumulate,
    )
    if schedule is not None:
      schedulers.append(schedule)
    return (
      {'optimizer': optimizer, 'lr_scheduler': schedulers[0]}
      if schedulers
      else {'optimizer': optimizer}
    )


def _resolve_class(class_path: str) -> type:
  """Import a class by dotted path.

  Args:
      class_path: Dotted path of the optimizer or scheduler class.

  Returns:
      The imported class object.
  """
  module_path, _, attr = class_path.rpartition('.')
  return getattr(importlib.import_module(module_path), attr)
