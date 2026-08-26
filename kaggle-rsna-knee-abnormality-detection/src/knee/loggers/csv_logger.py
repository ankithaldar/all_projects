#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""CSV logging helpers for fold-level metric history."""

from __future__ import annotations

import os

from pytorch_lightning.loggers import CSVLogger


def build_csv_logger(
  save_dir: str, experiment_name: str, fold_id: int
) -> CSVLogger:
  """Create a Lightning CSV logger namespaced per fold.

  Args:
      save_dir: Base directory for run artifacts.
      experiment_name: Experiment identifier used as the log name.
      fold_id: Fold number appended to the version string.

  Returns:
      Configured ``CSVLogger`` instance.
  """
  return CSVLogger(
    save_dir=os.path.join(save_dir, 'logs'),
    name=experiment_name,
    version=f'fold{fold_id}',
  )
