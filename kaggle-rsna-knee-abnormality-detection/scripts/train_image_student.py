#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Kernel 5: train the image student per fold with Lightning + config specs.

Kaggle budgets shape this kernel (BLUEPRINT section 9): 12 h per
notebook, ~30 GPU-hours total, no volumes cache. Consequently:

* **Fold sharding** -- ``--folds 0,1`` trains only those folds; spread
  folds over kernels to stay inside each budget.
* **Resume** -- a fold whose best checkpoint AND per-fold OOF parquet
  both exist is skipped; finished folds are loaded from disk, so simply
  re-running the same command continues the experiment. A fold cut off
  mid-training resumes epoch-level from ``fold<N>/last.ckpt`` when
  ``train.resume: true`` (kaggle_run.sh sets it by default).
* **Time budget** -- ``train.time_budget_hours`` (or ``--time-budget``)
  becomes Lightning ``max_time`` per fold, sized to the *remaining*
  kernel time, so the run always ends with valid checkpoints instead of
  being killed mid-write.

Usage:
    python scripts/train_image_student.py \
        --config configs/experiment/student_2p5d_effnetv2.yaml \
        --folds 0,1 [--time-budget 11] \
        [--set train.label_source=weak seed=7]

Outputs:
    <output_dir>/checkpoints/fold{N}-*.ckpt
    <output_dir>/predictions/<name>_oof_fold{N}.parquet   (per fold)
    <output_dir>/predictions/<name>_oof.parquet           (merged)
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from knee.config_params.loader import load_config
from knee.config_params.schema import TARGETS, ExperimentConfig
from knee.datamodules.knee_datamodule import KneeDataModule
from knee.engines.predictor import run_predict
from knee.engines.study_lit_module import KneeStudyLitModule
from knee.engines.trainer_factory import build_trainer
from knee.helpers.logging_utils import get_logger
from knee.helpers.seeding import seed_everything

#: Never start a fresh fold with less wall-clock left than this.
MIN_FOLD_SECONDS = 20 * 60


def parse_args() -> argparse.Namespace:
  """Parse CLI arguments.

  Returns:
      Namespace with config, overrides, fold filter and time budget.
  """
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--config', required=True)
  parser.add_argument(
    '--folds',
    default=None,
    help='Comma-separated fold ids overriding train.train_folds.',
  )
  parser.add_argument(
    '--time-budget',
    type=float,
    default=None,
    help='Wall-clock hours for this invocation (overrides '
    'train.time_budget_hours).',
  )
  parser.add_argument(
    '--set', nargs='*', default=[], dest='overrides', help='omegaconf dotlist'
  )
  return parser.parse_args()


def resolve_folds(args: argparse.Namespace, cfg: ExperimentConfig) -> list[int]:
  """Determine which folds this invocation owns.

  Args:
      args: Parsed CLI namespace.
      cfg: Loaded experiment configuration.

  Returns:
      Deduplicated fold ids in requested order.
  """
  if not args.folds:
    return list(cfg.train.train_folds)
  return sorted({int(f) for f in args.folds.split(',') if f.strip()})


class _ValOnlyDataModule:
  """Expose only the validation loader to the predictor.

  Args:
      dm: The fitted KneeDataModule.
  """

  def __init__(self, dm) -> None:
    self.dm = dm

  def predict_dataloader(self):
    """Return the underlying validation DataLoader.

    Returns:
        Validation DataLoader from the wrapped module.
    """
    return self.dm.val_dataloader()


def _best_checkpoint(cfg: ExperimentConfig, fold: int) -> Path | None:
  """Newest checkpoint for a fold, across new and legacy layouts.

  Current layout scopes each fold into ``checkpoints/fold<N>/`` so
  ``last.ckpt`` files cannot collide across folds; older kernels kept
  everything flat under ``checkpoints/`` -- both are searched.

  Args:
      cfg: Loaded experiment configuration.
      fold: Zero-based fold id.

  Returns:
      Most recently modified matching checkpoint path, or None.
  """
  roots = [
    Path(cfg.paths.output_dir) / cfg.paths.checkpoints_dir / f'fold{fold}',
    Path(cfg.paths.output_dir) / cfg.paths.checkpoints_dir,
  ]
  ordered = sorted(
    (
      p
      for root in roots
      for p in root.glob(f'fold{fold}-*.ckpt')
      if 'last' not in p.name
    ),
    key=lambda p: p.stat().st_mtime,
  )
  return ordered[-1] if ordered else None


def _resume_checkpoint(cfg: ExperimentConfig, fold: int) -> str | None:
  """Epoch-level resume target for a partially trained fold.

  Only the fold-scoped ``last.ckpt`` (exact trainer state written by
  ``save_last``) is trusted; resuming from a *best* checkpoint would
  reset optimizer/scheduler state mid-schedule.

  Args:
      cfg: Loaded experiment configuration.
      fold: Zero-based fold id.

  Returns:
      Checkpoint path for ``trainer.fit(ckpt_path=...)`` when resuming
      is enabled and a last checkpoint exists, else None (fresh start).
  """
  if not cfg.train.resume:
    return None
  last = (
    Path(cfg.paths.output_dir)
    / cfg.paths.checkpoints_dir
    / f'fold{fold}'
    / 'last.ckpt'
  )
  return str(last) if last.exists() else None


def _fold_oof_path(cfg: ExperimentConfig, fold: int) -> Path:
  """Per-fold OOF parquet location.

  Args:
      cfg: Loaded experiment configuration.
      fold: Zero-based fold id.

  Returns:
      Path under the predictions directory.
  """
  return (
    Path(cfg.paths.output_dir)
    / cfg.paths.predictions_dir
    / f'{cfg.name}_oof_fold{fold}.parquet'
  )


def _predict_val_fold(
  cfg: ExperimentConfig, fold: int, checkpoint: Path
) -> pd.DataFrame:
  """Score the held-out fold with its best checkpoint.

  Args:
      cfg: Loaded experiment configuration.
      fold: Zero-based fold id.
      checkpoint: Fold checkpoint to predict with.

  Returns:
      Frame with columns ``[StudyInstanceUID, *TARGETS]``.
  """
  dm = KneeDataModule(
    paths=cfg.paths,
    data_cfg=cfg.data,
    dm_cfg=cfg.datamodule,
    augment=cfg.augment,
    train_cfg=cfg.train,
    sampler_cfg=cfg.sampler,
    fold=fold,
  )
  dm.setup('validate')  # materializes only the held-out fold split
  return run_predict([checkpoint], _ValOnlyDataModule(dm), cfg.tta)


def main() -> None:
  """Run the resumable fold loop with per-fold OOF persistence."""

  args = parse_args()
  log = get_logger('train_image_student')
  cfg = load_config(args.config, overrides=args.overrides)
  seed_everything(cfg.seed)

  out_dir = Path(cfg.paths.output_dir)
  (out_dir / cfg.paths.predictions_dir).mkdir(parents=True, exist_ok=True)
  folds = resolve_folds(args, cfg)
  budget_hours = (
    args.time_budget
    if args.time_budget is not None
    else cfg.train.time_budget_hours
  )
  started = time.monotonic()

  oof_parts: list[pd.DataFrame] = []
  for fold in folds:
    oof_path = _fold_oof_path(cfg, fold)
    checkpoint = _best_checkpoint(cfg, fold)
    if checkpoint and oof_path.exists():
      log.info('fold %d already complete; reusing %s', fold, oof_path.name)
      oof_parts.append(pd.read_parquet(oof_path))
      continue

    elapsed = time.monotonic() - started
    if budget_hours is not None:
      remaining_h = float(budget_hours) - elapsed / 3600.0
      if remaining_h * 3600 < MIN_FOLD_SECONDS:
        log.warning(
          'stopping before fold %d: %.2f h left < minimum %.0f min '
          '(re-run the same command in a fresh kernel to continue)',
          fold,
          remaining_h,
          MIN_FOLD_SECONDS / 60,
        )
        break
      # Each fold gets exactly the time still remaining in the budget.
      cfg.train.time_budget_hours = max(remaining_h, MIN_FOLD_SECONDS / 3600)

    dm = KneeDataModule(
      paths=cfg.paths,
      data_cfg=cfg.data,
      dm_cfg=cfg.datamodule,
      augment=cfg.augment,
      train_cfg=cfg.train,
      sampler_cfg=cfg.sampler,
      fold=fold,
    )
    dm.setup('fit')
    module = KneeStudyLitModule(
      model_cfg=cfg.model,
      loss_cfg=cfg.loss,
      optimizer_cfg=cfg.optimizer,
      train_cfg=cfg.train,
    )
    trainer = build_trainer(cfg.train, cfg.paths, fold=fold, run_name=cfg.name)
    trainer.fit(
      module,
      datamodule=dm,
      ckpt_path=_resume_checkpoint(cfg, fold),
    )

    best = _best_checkpoint(cfg, fold)
    assert best, f'no checkpoint produced for fold {fold}'
    probs_frame = _predict_val_fold(cfg, fold, best)
    probs_frame.insert(0, 'fold', fold)
    probs_frame.to_parquet(oof_path)
    oof_parts.append(probs_frame)
    log.info(
      'fold %d complete (ckpt=%s, oof=%s)', fold, best.name, oof_path.name
    )

  _merge_and_score(cfg, oof_parts, log)


def _merge_and_score(
  cfg: ExperimentConfig,
  oof_parts: list[pd.DataFrame],
  log,
) -> None:
  """Merge available per-fold OOF parts and report gold-OOF macro AUC.

  Args:
      cfg: Loaded experiment configuration.
      oof_parts: In-memory OOF frames produced or reloaded above.
      log: Logger for progress and the summary line.
  """
  # Include parts persisted by previous kernels, even folds not listed.
  seen: set[str] = set()
  merged_parts: list[pd.DataFrame] = []
  for frame in oof_parts:
    merged_parts.append(frame)
    seen.update(frame['StudyInstanceUID'].astype(str))
  for path in sorted(
    (Path(cfg.paths.output_dir) / cfg.paths.predictions_dir).glob(
      f'{cfg.name}_oof_fold*.parquet'
    )
  ):
    frame = pd.read_parquet(path)
    if seen.isdisjoint(frame['StudyInstanceUID'].astype(str)):
      merged_parts.append(frame)
      seen.update(frame['StudyInstanceUID'].astype(str))

  if not merged_parts:
    log.warning('no OOF predictions available; skipping merge')
    return
  targets = list(TARGETS)
  oof = (
    pd.concat(merged_parts, ignore_index=True)
    .groupby('StudyInstanceUID', as_index=False)[targets]
    .mean()
  )
  gold = pd.read_csv(str(Path(cfg.paths.data_root) / cfg.paths.train_csv))
  if all(t in gold.columns for t in targets):
    scored = oof.merge(
      gold[['StudyInstanceUID', *targets]],
      on='StudyInstanceUID',
      suffixes=('_pred', ''),
    )
    aucs = []
    for t in targets:
      y_true_t = scored[t]
      pred_t = scored[f'{t}_pred']
      if pred_t.isna().any() or y_true_t.isna().any():
        continue
      if y_true_t.nunique() > 1:
        aucs.append(roc_auc_score(y_true_t, pred_t))
    log.info(
      'gold-OOF macro AUC %.4f (%d/%d classes defined, %d studies)',
      float(np.mean(aucs)) if aucs else float('nan'),
      len(aucs),
      len(targets),
      len(scored),
    )
  out = (
    Path(cfg.paths.output_dir)
    / cfg.paths.predictions_dir
    / f'{cfg.name}_oof.parquet'
  )
  out.parent.mkdir(parents=True, exist_ok=True)
  oof.to_parquet(out)
  log.info('OOF -> %s', out)


if __name__ == '__main__':
  main()
