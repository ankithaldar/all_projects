#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Kernel 6: inference over the test set -> submission.csv.

Usage:
    python scripts/infer.py \
        --config configs/experiment/student_2p5d_effnetv2.yaml \
        --test-csv /kaggle/input/.../test.csv \
        --ckpt-dir checkpoints
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from knee.config_params.loader import load_config
from knee.config_params.schema import TARGETS
from knee.datamodules.knee_datamodule import KneeDataModule
from knee.engines.ensemble import save_submission
from knee.engines.predictor import run_predict
from knee.helpers.logging_utils import get_logger
from knee.helpers.seeding import seed_everything


def parse_args() -> argparse.Namespace:
  """Parse CLI arguments.

  Returns:
      Namespace with config, test_csv, ckpt_dir, out path.
  """
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--config', required=True)
  parser.add_argument('--test-csv', required=True)
  parser.add_argument('--ckpt-dir', default='checkpoints')
  parser.add_argument('--out', default='submission.csv')
  return parser.parse_args()


def main() -> None:
  """Predict with all fold checkpoints and write a submission file."""

  args = parse_args()
  log = get_logger('infer')
  cfg = load_config(args.config)
  seed_everything(cfg.seed)

  dm = KneeDataModule(
    paths=cfg.paths,
    data_cfg=cfg.data,
    dm_cfg=cfg.datamodule,
    augment=cfg.augment,
    train_cfg=cfg.train,
    fold=-1,
    test_studies_csv=args.test_csv,
  )
  dm.setup('predict')
  # Fold-scoped layout (<ckpt-dir>/fold<N>/fold<N>-*.ckpt) with fallback
  # to the legacy flat layout from earlier kernels. ``last.ckpt`` files
  # never match the ``fold*`` glob, so only best checkpoints are used.
  ckpts = sorted(Path(args.ckpt_dir).rglob('fold*.ckpt'))
  assert ckpts, f'no fold checkpoints under {args.ckpt_dir}'
  frame = run_predict(ckpts, dm, cfg.tta)
  missing = set(pd_read_uids(args.test_csv)) - set(frame['StudyInstanceUID'])
  if missing:
    for uid in missing:
      row = {'StudyInstanceUID': uid}
      row.update({t: 0.5 for t in TARGETS})
      frame = pd_concat_row(frame, row)
    log.warning('%d studies had no prediction; filled 0.5', len(missing))
  save_submission(frame, args.out)
  log.info('submission -> %s (%d rows)', args.out, len(frame))


def pd_read_uids(csv_path: str):
  """Read StudyInstanceUID column from a test CSV.

  Args:
      csv_path: Path to test.csv.

  Returns:
      Series of study UIDs.
  """

  return pd.read_csv(csv_path)['StudyInstanceUID']


def pd_concat_row(frame, row: dict):
  """Append one fallback row to a submission frame.

  Args:
      frame: Existing submission-shaped DataFrame.
      row: Column->value dict.

  Returns:
      Extended DataFrame.
  """

  return pd.concat([frame, pd.DataFrame([row])], ignore_index=True)


if __name__ == '__main__':
  main()
