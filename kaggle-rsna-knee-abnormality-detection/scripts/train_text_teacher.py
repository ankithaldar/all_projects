#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Kernel 4: train the multilingual text teacher and emit OOF probabilities.

Usage:
    python scripts/train_text_teacher.py \
        --config configs/labeling/text_teacher.yaml
Outputs (under cfg.data.output_dir):
    ckpts/fold{N}.ckpt, oof_probs.parquet
"""

from __future__ import annotations

import argparse
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
  gold = df.dropna(subset=[c for c in TARGETS if c in df.columns])

  tokenizer = AutoTokenizer.from_pretrained(cfg.model.backbone)
  oof = np.zeros((len(df), len(TARGETS)), dtype=np.float32)
  uid_to_idx = {u: i for i, u in enumerate(df['StudyInstanceUID'])}

  for fold in range(cfg.train.n_folds_to_train):
    tr = gold[gold['fold'] != fold] if 'fold' in gold.columns else gold
    va = (
      gold[gold['fold'] == fold] if 'fold' in gold.columns else gold.iloc[0:0]
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
        return len(self.base)

      def __getitem__(self, i: int) -> dict:
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
    module = TextTeacherLitModule(
      backbone=cfg.model.backbone,
      lr=cfg.train.lr,
      weight_decay=cfg.train.weight_decay,
    )
    trainer = pl.Trainer(
      max_epochs=cfg.train.epochs,
      precision=cfg.train.amp,
      accelerator='auto',
      devices=1,
      accumulate_grad_batches=cfg.train.grad_accum,
      default_root_dir=str(out_dir),
      enable_checkpointing=False,
      logger=False,
    )
    trainer.fit(module, loader)
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
    ckpt_path = ckpt_dir / f'fold{fold}.ckpt'
    trainer.save_checkpoint(ckpt_path)
    log.info('fold %d done -> %s', fold, ckpt_path)

  oof_frame = df[['StudyInstanceUID']].copy()
  for j, t in enumerate(TARGETS):
    oof_frame[t] = oof[:, j]
  oof_frame.to_parquet(out_dir / 'oof_probs.parquet')
  log.info('OOF probs -> %s', out_dir / 'oof_probs.parquet')


if __name__ == '__main__':
  main()
