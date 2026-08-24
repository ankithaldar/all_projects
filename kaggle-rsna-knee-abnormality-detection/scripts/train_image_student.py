#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Kernel 5: train the image student per fold with Lightning + config specs.

Usage:
    python scripts/train_image_student.py \
        --config configs/experiment/student_2p5d_effnetv2.yaml \
        [--set train.label_source=weak seed=7]
Outputs:
    <output_dir>/checkpoints/fold{N}-*.ckpt, oof_probs.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from knee.config_params.loader import load_config
from knee.config_params.schema import TARGETS
from knee.datamodules.knee_datamodule import KneeDataModule
from knee.engines.predictor import run_predict
from knee.engines.study_lit_module import KneeStudyLitModule
from knee.engines.trainer_factory import build_trainer
from knee.helpers.logging_utils import get_logger
from knee.helpers.seeding import seed_everything


def parse_args() -> argparse.Namespace:
  """Parse CLI arguments.

  Returns:
      Namespace with config, overrides, and fold filter.
  """
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--config', required=True)
  parser.add_argument(
    '--fold', type=int, default=None, help='train a single fold'
  )
  parser.add_argument(
    '--set', nargs='*', default=[], dest='overrides', help='omegaconf dotlist'
  )
  return parser.parse_args()


def main() -> None:
  """Run the fold loop: fit -> best checkpoint -> OOF probabilities."""

  args = parse_args()
  log = get_logger('train_image_student')
  cfg = load_config(args.config, overrides=args.overrides)
  seed_everything(cfg.seed)

  out_dir = Path(cfg.paths.output_dir)
  folds = cfg.train.train_folds if args.fold is None else [args.fold]
  oof_rows = []
  for fold in folds:
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
    trainer.fit(module, datamodule=dm)

    best = sorted(
      (Path(cfg.paths.output_dir) / cfg.paths.checkpoints_dir).glob(
        f'fold{fold}-*.ckpt'
      )
    )
    assert best, f'no checkpoint produced for fold {fold}'
    probs_frame = run_predict([best[-1]], _val_only(dm), cfg.tta)
    probs_frame.insert(0, 'fold', fold)
    oof_rows.append(probs_frame)
    log.info('fold %d complete (ckpt=%s)', fold, best[-1].name)

  oof = (
    pd.concat(oof_rows)
    .groupby('StudyInstanceUID', as_index=False)[list(TARGETS)]
    .mean()
  )
  gold = pd.read_csv(str(Path(cfg.paths.data_root) / cfg.paths.train_csv))
  if all(t in gold.columns for t in TARGETS):
    scored = oof.merge(
      gold[['StudyInstanceUID', *TARGETS]],
      on='StudyInstanceUID',
      suffixes=('_pred', ''),
    )
    aucs = []
    for t in TARGETS:
      if scored[f'{t}_pred'].isna().any():
        continue
      y_true_t = scored[t]
      if y_true_t.nunique() > 1:
        aucs.append(roc_auc_score(y_true_t, scored[f'{t}_pred']))
    log.info(
      'gold-OOF macro AUC %.4f (%d classes defined, %d studies)',
      float(np.mean(aucs)) if aucs else float('nan'),
      len(aucs),
      len(scored),
    )
  out = out_dir / cfg.paths.predictions_dir / f'{cfg.name}_oof.parquet'
  out.parent.mkdir(parents=True, exist_ok=True)
  oof.to_parquet(out)
  log.info('OOF -> %s', out)


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


def _val_only(dm):
  """Wrap a datamodule so run_predict can iterate its val loader.

  Args:
      dm: Fitted KneeDataModule.

  Returns:
        Predictor-compatible adapter.
  """
  return _ValOnlyDataModule(dm)


if __name__ == '__main__':
  main()
