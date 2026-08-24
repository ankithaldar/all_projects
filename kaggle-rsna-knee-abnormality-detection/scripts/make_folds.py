#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Kernel 2: create the frozen multi-label stratified fold assignment.

Usage:
    python scripts/make_folds.py --train-csv train.csv --out-csv train_folds.csv
"""

from __future__ import annotations

import argparse

from knee.datasets.folds import fold_summary, make_iterative_multilabel_folds
from knee.helpers.logging_utils import get_logger


def parse_args() -> argparse.Namespace:
  """Parse CLI arguments.

  Returns:
      Namespace with train_csv, out_csv, n_folds, seed.
  """
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--train-csv', required=True)
  parser.add_argument('--out-csv', default='train_folds.csv')
  parser.add_argument('--n-folds', type=int, default=5)
  parser.add_argument('--seed', type=int, default=42)
  return parser.parse_args()


def main() -> None:
  """Write train_folds.csv and print the per-fold balance summary."""

  args = parse_args()
  folds = make_iterative_multilabel_folds(
    args.train_csv, n_folds=args.n_folds, seed=args.seed
  )
  folds.to_csv(args.out_csv, index=False)
  get_logger('make_folds').info(
    'wrote %s\n%s', args.out_csv, fold_summary(folds).to_string()
  )


if __name__ == '__main__':
  main()
