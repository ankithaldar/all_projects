#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Kernel 5b: one noisy-student round - student OOF becomes new teacher.

Usage:
    python scripts/self_train.py \
        --config configs/experiment/student_2p5d_effnetv2.yaml \
        --student-oof predictions/student_2p5d_effnetv2_distilled_oof.parquet \
        --round 2
Writes:
    weak_labels_round{N}.parquet  (point --set paths.weak_labels_parquet at it)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from knee.config_params.loader import load_config
from knee.config_params.schema import TARGETS
from knee.engines.rule_labeler import RuleBasedLabeler
from knee.engines.weak_label_builder import WeakLabelBuilder
from knee.helpers.logging_utils import get_logger


def parse_args() -> argparse.Namespace:
  """Parse CLI arguments.

  Returns:
      Namespace with config, student_oof, round number.
  """
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--config', required=True)
  parser.add_argument('--student-oof', required=True)
  parser.add_argument('--round', type=int, default=2, dest='round_number')
  return parser.parse_args()


def main() -> None:
  """Re-fuse labels with the student's own OOF as teacher signal."""

  args = parse_args()
  log = get_logger('self_train')
  cfg = load_config(args.config)

  train_csv = str(Path(cfg.paths.data_root) / cfg.paths.train_csv)
  df = pd.read_csv(train_csv)
  labeler = RuleBasedLabeler()
  seeds = [labeler.seed_labels(r) for r in df['Report'].fillna('')]
  rule_probs = pd.DataFrame([s[0] for s in seeds], columns=list(TARGETS))
  rule_mask = pd.DataFrame([s[1] for s in seeds], columns=list(TARGETS))
  rule_probs.insert(0, 'StudyInstanceUID', df['StudyInstanceUID'])
  rule_mask.insert(0, 'StudyInstanceUID', df['StudyInstanceUID'])

  builder = WeakLabelBuilder(
    rule_confidence_floor=0.9,
    teacher_weight=1.0,
    min_positive_prob=0.35,
  )
  # Cap self-training at two rounds (BLUEPRINT risk list): keep round-0
  # model as an ensemble anchor to prevent drift.
  if args.round_number > 2:
    raise SystemExit(
      'refusing to run more than 2 self-training rounds (drift risk)'
    )
  weak, stats = builder.build(
    train_csv=train_csv,
    rule_probs=rule_probs,
    rule_mask=rule_mask,
    rule_precision=None,
    teacher_oof=pd.read_parquet(args.student_oof)[
      ['StudyInstanceUID', *TARGETS]
    ],
  )
  out = (
    Path(cfg.paths.output_dir) / f'weak_labels_round{args.round_number}.parquet'
  )
  weak.to_parquet(out)
  log.info('round %d labels -> %s | %s', args.round_number, out, stats)


if __name__ == '__main__':
  main()
