#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Kernel 7: greedy-blend OOF members and apply weights to test submissions.

Usage:
    python scripts/blend_submissions.py \
        --oof 'predictions/*_oof.parquet' \
        --gold train_folds.csv \
        --subs 'submissions/*.csv' \
        --out submission_blended.csv [--rank-average]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from knee.config_params.schema import TARGETS
from knee.engines.ensemble import GreedyBlender, save_submission
from knee.helpers.logging_utils import get_logger


def parse_args() -> argparse.Namespace:
  """Parse CLI arguments.

  Returns:
      Namespace with oof glob, gold csv, subs glob, out path, rank flag.
  """
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    '--oof', required=True, help='glob of *_oof.parquet frames'
  )
  parser.add_argument(
    '--gold', required=True, help='csv with StudyInstanceUID + binary targets'
  )
  parser.add_argument(
    '--subs',
    required=True,
    help='glob of test submission csvs (same member order)',
  )
  parser.add_argument('--out', default='submission_blended.csv')
  parser.add_argument('--rank-average', action='store_true')
  return parser.parse_args()


def main() -> None:
  """Select ensemble members on OOF, then blend the matching test files."""

  args = parse_args()
  log = get_logger('blend')
  oof_paths = sorted(Path().glob(args.oof))
  # Never let a previous blended output join the member pool: re-running
  # the stage would otherwise feed the blend its own submission.
  out_name = Path(args.out).name
  sub_paths = [p for p in sorted(Path().glob(args.subs)) if p.name != out_name]
  assert len(oof_paths) == len(sub_paths) >= 1, (
    'member count mismatch between --oof and --subs '
    f'(oof={len(oof_paths)}, subs={len(sub_paths)} after '
    f'excluding {out_name})'
  )

  names = [p.stem.replace('_oof', '') for p in oof_paths]
  oof_frames = {
    n: pd.read_parquet(p) for n, p in zip(names, oof_paths, strict=True)
  }
  gold = pd.read_csv(args.gold)

  blender = GreedyBlender(rank_average=args.rank_average).fit(oof_frames, gold)
  log.info(
    'selected %s (gold-OOF macro AUC %.4f)',
    blender.selected_,
    blender.best_score_,
  )

  sub_by_name = dict(zip(names, sub_paths, strict=True))
  total_weight = sum(blender.weights_.values())
  blended = pd.read_csv(sub_by_name[blender.selected_[0]])[
    ['StudyInstanceUID']
  ].copy()
  for t in TARGETS:
    blended[t] = 0.0
  for name in blender.selected_:
    member = pd.read_csv(sub_by_name[name])
    w = blender.weights_[name] / total_weight
    for t in TARGETS:
      blended[t] += w * member[t]
  save_submission(blended, args.out)
  log.info('blended submission -> %s', args.out)


if __name__ == '__main__':
  main()
