#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Lightning training module for KneeNet."""

from __future__ import annotations

import importlib
import os

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch import Tensor, nn

from knee.metrics.auc import MultilabelAUC
from knee.models.knee_net import KneeNet


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
    schedulers = []
    if self.scheduler_cfg is not None:
      sched_init = dict(self.scheduler_cfg.get('init_params', {}))
      sched_cls = _resolve_class(self.scheduler_cfg['class_path'])
      cosine = sched_cls(optimizer, **sched_init)
      if self.warmup_epochs > 0:
        linear = torch.optim.lr_scheduler.LinearLR(
          optimizer, start_factor=0.05, total_iters=self.warmup_epochs
        )
        schedulers.append(
          {
            'scheduler': torch.optim.lr_scheduler.SequentialLR(
              optimizer,
              [linear, cosine],
              milestones=[self.warmup_epochs],
            ),
            'interval': 'epoch',
          }
        )
      else:
        schedulers.append({'scheduler': cosine, 'interval': 'epoch'})
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
