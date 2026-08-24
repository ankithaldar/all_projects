#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Kernel 4: train the multilingual text teacher and emit OOF probabilities.

Kaggle budgets: the teacher trains 5 XLM-R folds; a hard kernel kill at
12 h would lose everything, so this kernel is *resumable* (finished
fold checkpoints are reloaded, not retrained) and *time-budgeted*
(``train.time_budget_hours`` feeds Lightning ``max_time`` with the
remaining wall clock per fold).

Usage:
    python scripts/train_text_teacher.py \
        --config configs/labeling/text_teacher.yaml
Outputs (under cfg.data.output_dir):
    ckpts/fold{N}.ckpt, oof_probs.parquet
"""

from __future__ import annotations

import argparse
import time
from datetime import timedelta
from pathlib import Path

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from knee.config_params.schema import TARGETS
from knee.engines.text_teacher_lit import (
  ReportDataset,
  TextTeacherConfig,
  TextTeacherLitModule,
  predict_probs,
)
from knee.helpers.logging_utils import get_logger
from knee.helpers.seeding import seed_everything

#: Never start a fresh fold with less wall-clock left than this.
MIN_FOLD_SECONDS = 20 * 60


def _resolve_precision(amp: str) -> str:
  """Fall back to fp16-mixed when the GPU predates bf16 tensor cores.

  Kaggle T4/P100/K80 GPUs (compute capability < 8.0) cannot run bf16
  autocast; Lightning would fail or crawl. CPU-only runs keep bf16
  untouched (PyTorch emulates it fine there).

  Args:
      amp: Configured precision string ('bf16', '16-mixed', ...).

  Returns:
      The precision actually usable on this machine.
  """
  if 'bf16' not in amp or not torch.cuda.is_available():
    return amp
  major, _ = torch.cuda.get_device_capability(0)
  if major >= 8:
    return amp
  get_logger('train_text_teacher').warning(
    'pre-Ampere GPU detected; bf16 -> 16-mixed'
  )
  return '16-mixed'


def parse_args() -> argparse.Namespace:
  """Parse CLI arguments.

  Returns:
      Namespace with config path.
  """
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--config', default='configs/labeling/text_teacher.yaml')
  return parser.parse_args()


def main() -> None:
  """Train one teacher per fold; pool out-of-fold predictions."""
  args = parse_args()
  log = get_logger('train_text_teacher')

  cfg = TextTeacherConfig.model_validate(
    OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
  )
  seed_everything(cfg.seed)
  out_dir = Path(cfg.data.output_dir)
  ckpt_dir = out_dir / 'ckpts'
  ckpt_dir.mkdir(parents=True, exist_ok=True)

  df = pd.read_csv(cfg.data.train_csv)
  # Merge the frozen fold assignment when train.csv lacks it; without a
  # fold column every fold would train on ALL gold studies and the OOF
  # parquet would stay empty.
  if 'fold' not in df.columns and cfg.data.folds_csv:
    folds_path = Path(cfg.data.folds_csv)
    if not folds_path.is_absolute():
      folds_path = Path(cfg.data.train_csv).parent / folds_path
    if folds_path.exists():
      df = df.merge(
        pd.read_csv(folds_path)[['StudyInstanceUID', 'fold']],
        on='StudyInstanceUID',
        how='left',
      )
      log.info('merged fold column from %s', folds_path)
  gold = df.dropna(subset=[c for c in TARGETS if c in df.columns])

  tokenizer = AutoTokenizer.from_pretrained(cfg.model.backbone)
  oof = np.zeros((len(df), len(TARGETS)), dtype=np.float32)
  uid_to_idx = {u: i for i, u in enumerate(df['StudyInstanceUID'])}

  budget_hours = cfg.train.time_budget_hours
  started = time.monotonic()
  for fold in range(cfg.train.n_folds_to_train):
    ckpt_path = ckpt_dir / f'fold{fold}.ckpt'
    tr = gold[gold['fold'] != fold] if 'fold' in gold.columns else gold
    va = (
      gold[gold['fold'] == fold] if 'fold' in gold.columns else gold.iloc[0:0]
    )

    module = TextTeacherLitModule(
      backbone=cfg.model.backbone,
      lr=cfg.train.lr,
      weight_decay=cfg.train.weight_decay,
    )
    trainer_kwargs: dict = {
      'max_epochs': cfg.train.epochs,
      'precision': _resolve_precision(cfg.train.amp),
      'accelerator': 'auto',
      'devices': 1,
      'accumulate_grad_batches': cfg.train.grad_accum,
      'default_root_dir': str(out_dir),
      'enable_checkpointing': False,
      'logger': False,
    }
    if ckpt_path.exists():
      # Resume: reuse the finished fold's weights for OOF prediction.
      log.info('fold %d checkpoint found; skipping training', fold)
    else:
      if budget_hours is not None:
        remaining_h = float(budget_hours) - (time.monotonic() - started) / 3600
        if remaining_h * 3600 < MIN_FOLD_SECONDS:
          log.warning(
            'stopping before teacher fold %d: %.2f h left < minimum '
            '%.0f min (re-run to continue)',
            fold,
            remaining_h,
            MIN_FOLD_SECONDS / 60,
          )
          break
        trainer_kwargs['max_time'] = timedelta(
          hours=max(remaining_h, MIN_FOLD_SECONDS / 3600)
        )
      train_ds = ReportDataset(
        tr['Report'].fillna(''), tokenizer, cfg.model.max_length
      )

      class _TorchDs(torch.utils.data.Dataset):
        """Wrap report windows with their targets.

        Args:
            base: Underlying ReportDataset.
            targets: Aligned target matrix.
        """

        def __init__(self, base: ReportDataset, targets: np.ndarray) -> None:
          self.base = base
          self.targets = targets

        def __len__(self) -> int:
          """Number of reports.

          Returns:
              Dataset length.
          """
          return len(self.base)

        def __getitem__(self, i: int) -> dict:
          """Tokenize one report and attach its targets.

          Args:
              i: Report position.

          Returns:
              Dict with input_ids/attention_mask plus target vector.
          """
          item = self.base[i]
          item['targets'] = torch.from_numpy(self.targets[i])
          return item

      targets_tr = (
        tr[list(TARGETS)].to_numpy(np.float32)
        if len(tr)
        else np.zeros((0, 12), np.float32)
      )
      loader = DataLoader(
        _TorchDs(train_ds, targets_tr),
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=2,
      )
      trainer = pl.Trainer(**trainer_kwargs)
      trainer.fit(module, loader)
      trainer.save_checkpoint(ckpt_path)
      log.info('fold %d trained -> %s', fold, ckpt_path)

    module.load_state_dict(
      torch.load(ckpt_path, map_location='cpu')['state_dict']
    )
    module.to('cuda' if torch.cuda.is_available() else 'cpu')
    if len(va):
      va_probs = predict_probs(
        module,
        tokenizer,
        va['Report'].fillna('').tolist(),
        max_length=cfg.model.max_length,
        stride=cfg.model.stride,
      )
      for j, u in enumerate(va['StudyInstanceUID']):
        oof[uid_to_idx[u]] = va_probs[j]

  oof_frame = df[['StudyInstanceUID']].copy()
  for j, t in enumerate(TARGETS):
    oof_frame[t] = oof[:, j]
  oof_frame.to_parquet(out_dir / 'oof_probs.parquet')
  log.info('OOF probs -> %s', out_dir / 'oof_probs.parquet')


if __name__ == '__main__':
  main()
