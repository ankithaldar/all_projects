#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Lightning training module for KneeNet."""

from __future__ import annotations

import glob
import importlib
import math
import os

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from torch import Tensor, nn

from knee.helpers.utils import get_logger
from knee.metrics.auc import MultilabelAUC
from knee.models.knee_net import KneeNet

OOF_SHARD_DIR = '.shards'
_LOGGER = get_logger(__name__)


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

  def on_validation_epoch_start(self) -> None:
    """Flush the metric accumulator before a new epoch collects.

    Without the reset, Lightning's sanity check and EVERY previous
    validation epoch leak into the accumulator, so the logged
    ``val/auc_macro`` blends epochs and the OOF csv becomes the union
    of all epochs instead of the current epoch's predictions (which
    the noise-floor harness and OOF analyses rely on).
    """
    self.metric.reset()

  def on_validation_epoch_end(self) -> None:
    """Log per-class AUCs and persist the OOF table.

    DDP safety: each rank writes a per-rank shard to
    ``<oof_dir>/.shards/oof_fold{k}_rank{r}.csv`` and a barrier waits
    for the other ranks. Rank 0 then concatenates the shards into the
    canonical ``<oof_dir>/oof_fold{k}.csv`` consumed by downstream
    analysis (noise floor, OOF inspection). Without this split every
    rank's ``on_validation_epoch_end`` would overwrite the canonical
    from its own val-shard alone and the OOF would contain only one
    rank's studies; under 2xT4 the file would be half the val set.

    The canonical gets deduped by ``StudyInstanceUID`` and sorted so
    downstream readers see a stable row order across epochs.

    Single-process: ``world_size == 1`` so the barrier is skipped and
    rank 0 (the only rank) writes the canonical from its own shard -
    identical to the legacy behavior.
    """
    summary = self.metric.summary()
    try:
      # ``self.log`` is informational here; the canonical OOF (assembled
      # below) is what feeds downstream analysis. No ``sync_dist``:
      # there is no DDP-aware model selection (no ModelCheckpoint /
      # EarlyStopping) that would need the globally reduced metric.
      self.log('val/auc_macro', summary['auc/macro'], prog_bar=True)
      for name, value in summary.items():
        if name != 'auc/macro' and not np.isnan(value):
          self.log(f'val/{name}', value)
    except (
      AttributeError,
      RuntimeError,
      MisconfigurationException,
    ) as exc:
      # Logging must NEVER block the OOF write (same degradation
      # contract as DiscordCallback._current_lr): trainer-shaped
      # surprises degrade to 'metric not logged'.
      _LOGGER.warning('metric logging skipped (%s)', exc)
    self._write_oof_canonical()

  def _write_oof_canonical(self) -> None:
    """Per-rank OOF shard + rank-0 canonical assembly (DDP-safe)."""
    pl_trainer = None
    try:
      pl_trainer = self.trainer
    except RuntimeError:
      pl_trainer = None  # hook reachable without a Trainer (tests)
    if getattr(pl_trainer, 'sanity_checking', False):
      # Sanity predictions are throwaway (2 batches) and would
      # transiently publish a bogus canonical OOF.
      return
    probs, targets = self.metric.stacked()
    uids = list(self.metric.study_uids)
    frame = pd.DataFrame(
      probs, columns=[f'{c}_prob' for c in self.target_columns]
    )
    for col, name in enumerate(self.target_columns):
      frame[name] = targets[:, col]
    frame.insert(0, 'StudyInstanceUID', uids)
    frame.insert(1, 'fold', self.fold_id)
    os.makedirs(self.oof_dir, exist_ok=True)

    rank = int(getattr(pl_trainer, 'global_rank', 0) or 0)
    world_size = int(getattr(pl_trainer, 'world_size', 1) or 1)
    shard_dir = os.path.join(self.oof_dir, OOF_SHARD_DIR)
    os.makedirs(shard_dir, exist_ok=True)
    shard_path = os.path.join(
      shard_dir, f'oof_fold{self.fold_id}_rank{rank}.csv'
    )
    frame.to_csv(shard_path, index=False)

    # DDP sync: every rank's barrier call is the rendezvous point; PL's
    # Strategy.barrier is a collective barrier under DDP and a no-op
    # under single-process strategies.
    if world_size > 1:
      strategy = getattr(pl_trainer, 'strategy', None)
      if strategy is not None and hasattr(strategy, 'barrier'):
        strategy.barrier()

    if rank == 0:
      canonical = assemble_oof_canonical(self.oof_dir, self.fold_id)
      if canonical is None:
        # No shards found at all: fall back to this rank's frame so the
        # canonical always reflects the CURRENT epoch (a stale file from
        # a previous epoch/session must never survive).
        canonical = frame
      # Write UNCONDITIONALLY, even for an empty val epoch: skipping
      # would leave a stale canonical from a previous epoch in place.
      canonical.to_csv(
        os.path.join(self.oof_dir, f'oof_fold{self.fold_id}.csv'),
        index=False,
      )
      # Shards served their purpose; remove exactly the files we read
      # (never the directory - a fast rank could already be writing the
      # next epoch's shard into it).
      for shard_file in sorted(
        glob.glob(os.path.join(shard_dir, f'oof_fold{self.fold_id}_rank*.csv'))
      ):
        try:
          os.remove(shard_file)
        except OSError:
          pass

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
  """Import a class from its dotted path.

  Args:
      class_path: Dotted path of the optimizer or scheduler class.

  Returns:
      The imported class object.
  """
  module_path, _, attr = class_path.rpartition('.')
  return getattr(importlib.import_module(module_path), attr)


def assemble_oof_canonical(oof_dir: str, fold_id: int) -> pd.DataFrame | None:
  """Concatenate per-rank OOF shards into the canonical table.

  Each rank of a DDP fit writes a shard to
  ``<oof_dir>/.shards/oof_fold{f}_rank{r}.csv`` during
  :meth:`KneeModule.on_validation_epoch_end`; rank 0 then calls this
  function after a collective barrier to build the canonical
  ``<oof_dir>/oof_fold{k}.csv`` consumed by downstream analysis.

  The concatenation:
    1. preserves the union of StudyInstanceUIDs across ranks
       (defensive dedupe: the last duplicate wins);
    2. sorts by StudyInstanceUID for a stable row order across epochs.

  Args:
      oof_dir: Directory holding ``.shards/``; the canonical is
          written next to it.
      fold_id: Fold identifier to assemble.

  Returns:
      Assembled DataFrame, or None when no shards exist.
  """
  shard_dir = os.path.join(oof_dir, OOF_SHARD_DIR)
  if not os.path.isdir(shard_dir):
    return None
  pattern = os.path.join(shard_dir, f'oof_fold{int(fold_id)}_rank*.csv')
  shard_files = sorted(glob.glob(pattern))
  if not shard_files:
    return None
  frames = [pd.read_csv(path) for path in shard_files]
  merged = pd.concat(frames, ignore_index=True)
  if 'StudyInstanceUID' in merged.columns and not merged.empty:
    merged = merged.drop_duplicates(subset='StudyInstanceUID', keep='last')
    merged = merged.sort_values('StudyInstanceUID').reset_index(drop=True)
  return merged
