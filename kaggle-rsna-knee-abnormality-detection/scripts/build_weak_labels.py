#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Kernel 3: build weak_labels.parquet from rules + (optionally) the teacher.

Modes:
  1. rules-only: fast, high precision, moderate recall.
  2. --teacher-dir with fold checkpoints from train_text_teacher.py:
     full cascade; teacher OOF probabilities fill rule gaps.

Usage:
    python scripts/build_weak_labels.py \
        --config configs/labeling/text_teacher.yaml \
        [--teacher-dir text_teacher/ckpts]
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from knee.config_params.schema import TARGETS
from knee.engines.rule_labeler import RuleBasedLabeler
from knee.engines.text_teacher_lit import TextTeacherConfig
from knee.engines.weak_label_builder import WeakLabelBuilder
from knee.helpers.logging_utils import get_logger


def parse_args() -> argparse.Namespace:
  """Parse CLI arguments.

  Returns:
      Namespace with config, teacher_dir, out_parquet.
  """
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--config', default='configs/labeling/text_teacher.yaml')
  parser.add_argument(
    '--teacher-dir', default=None, help='dir of fold ckpts + oof parquet'
  )
  parser.add_argument('--out-parquet', default=None)
  return parser.parse_args()


def measure_rule_precision(
  gold: pd.DataFrame, probs: pd.DataFrame, mask: pd.DataFrame
) -> pd.Series:
  """Per-target precision of affirmative rule seeds on gold-labeled studies.

  Args:
      gold: Frame with StudyInstanceUID + binary TARGETS.
      probs: Rule seed probabilities aligned by StudyInstanceUID.
      mask: Rule validity mask aligned by StudyInstanceUID.

  Returns:
      Series mapping target -> precision (NaN where no seeds exist).
  """
  joined = gold.merge(
    probs, on='StudyInstanceUID', suffixes=('_gold', '_rule')
  ).merge(mask, on='StudyInstanceUID', suffixes=('', '_m'))
  precisions = {}
  for t in TARGETS:
    sel = (
      (joined[f'{t}_m'] > 0.5) & (joined[f'{t}_gold'] >= 0)
      if f'{t}_gold' in joined
      else None
    )
    if sel is None:
      continue
    seeded = joined[sel]
    positive_seeds = seeded[seeded[f'{t}_rule'] > 0.5]
    precisions[t] = (
      float((positive_seeds[f'{t}_gold'] > 0.5).mean())
      if len(positive_seeds)
      else np.nan
    )
  return pd.Series(precisions)


def main() -> None:
  """Run the Tier-1 (+ optional Tier-2) label cascade and save parquet."""

  args = parse_args()
  log = get_logger('build_weak_labels')
  cfg = TextTeacherConfig.model_validate(
    OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
  )

  df = pd.read_csv(cfg.data.train_csv)
  labeler = RuleBasedLabeler()
  seed_matrix = [labeler.seed_labels(r) for r in df['Report'].fillna('')]
  probs = pd.DataFrame(
    np.stack([s[0] for s in seed_matrix]), columns=list(TARGETS)
  )
  mask = pd.DataFrame(
    np.stack([s[1] for s in seed_matrix]), columns=list(TARGETS)
  )
  probs.insert(0, 'StudyInstanceUID', df['StudyInstanceUID'])
  mask.insert(0, 'StudyInstanceUID', df['StudyInstanceUID'])

  gold_cols = ['StudyInstanceUID'] + [c for c in TARGETS if c in df.columns]
  precision = (
    measure_rule_precision(df[gold_cols], probs, mask)
    if len(gold_cols) > 1
    else None
  )
  log.info('rule precision on gold subset:\n%s', precision)

  teacher_oof = None
  if args.teacher_dir:
    candidate = pd.read_parquet(f'{args.teacher_dir}/oof_probs.parquet')
    teacher_oof = candidate[['StudyInstanceUID', *TARGETS]]
    log.info('loaded teacher OOF for %d studies', len(teacher_oof))

  builder = WeakLabelBuilder(
    rule_confidence_floor=cfg.fusion.rule_confidence_floor,
    teacher_weight=cfg.fusion.teacher_weight,
    min_positive_prob=cfg.fusion.min_positive_prob,
  )
  weak, stats = builder.build(
    train_csv=cfg.data.train_csv,
    rule_probs=probs,
    rule_mask=mask,
    rule_precision=precision,
    teacher_oof=teacher_oof,
  )
  out = args.out_parquet or f'{cfg.data.output_dir}/{cfg.fusion.output_parquet}'
  weak.to_parquet(out)
  log.info('weak labels -> %s | %s', out, stats)


if __name__ == '__main__':
  main()
