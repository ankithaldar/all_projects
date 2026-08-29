#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Regression tests for per-epoch metric isolation in KneeModule.

Historical bug: MultilabelAUC was never reset between validation
epochs, so Lightning's sanity-check batches and every prior epoch's
predictions accumulated into the OOF csv and the logged val/auc_macro.
"""

# Only the metric lifecycle hooks are exercised; model/loss are inert
# placeholders because validation hooks never call them.
# pylint: disable=invalid-name

import os

import numpy as np
import pandas as pd

from knee.engines.train_module import KneeModule

TARGETS = ['ACL', 'Fracture']


def _module(tmp_path, fold=3):
  """KneeModule wired to a tmp oof dir with inert collaborators.

  Args:
      tmp_path: pytest temporary directory.
      fold: Fold id stamped into the OOF filename/rows.

  Returns:
      KneeModule instance.
  """
  return KneeModule(
    model=object(),
    criterion=object(),
    optimizer_cfg={'class_path': 'torch.optim.Adam', 'init_params': {}},
    scheduler_cfg=None,
    warmup_epochs=0,
    backbone_lr_scale=0.1,
    total_epochs=1,
    target_columns=TARGETS,
    oof_dir=str(tmp_path),
    fold_id=fold,
  )


def _update(module, uids):
  """Push a fake two-study validation batch into the accumulator.

  Args:
      module: KneeModule under test.
      uids: Study uids for the fake batch.
  """
  probs = np.arange(2 * len(TARGETS), dtype=np.float64).reshape(2, -1) / 10
  targets = np.tile([1.0, 0.0], (2, 1))
  module.metric.update(probs, targets, uids)


def _oof_rows(tmp_path, fold=3):
  """Read the persisted OOF csv.

  Args:
      tmp_path: Directory holding oof_fold{k}.csv.
      fold: Fold id.

  Returns:
      List of StudyInstanceUID strings in file order.
  """
  path = os.path.join(str(tmp_path), f'oof_fold{fold}.csv')
  assert os.path.exists(path)
  return pd.read_csv(path)['StudyInstanceUID'].tolist()


def test_sanity_check_does_not_leak_into_epoch(tmp_path):
  """Simulated sanity batches are flushed before the real epoch."""
  module = _module(tmp_path)
  _update(module, ['sanity_1', 'sanity_2'])
  module.on_validation_epoch_start()
  _update(module, ['real_1', 'real_2'])
  module.on_validation_epoch_end()
  assert _oof_rows(tmp_path) == ['real_1', 'real_2']


def test_epochs_do_not_accumulate(tmp_path):
  """Each epoch rewrites the OOF file with its own rows only."""
  module = _module(tmp_path)
  for uids in (['e0_a', 'e0_b'], ['e1_a', 'e1_b'], ['e2_a', 'e2_b']):
    module.on_validation_epoch_start()
    _update(module, uids)
    module.on_validation_epoch_end()
    assert _oof_rows(tmp_path) == uids
    assert len(_oof_rows(tmp_path)) == 2


def test_oof_row_count_matches_val_set(tmp_path):
  """OOF rows equal the validation study count, never a multiple."""
  module = _module(tmp_path)
  uids = [f'val_{i}' for i in range(5)]
  module.on_validation_epoch_start()
  for pos in range(0, 5, 1):
    module.metric.update(
      np.full((1, len(TARGETS)), 0.5),
      np.zeros((1, len(TARGETS))),
      [uids[pos]],
    )
  module.on_validation_epoch_end()
  rows = _oof_rows(tmp_path)
  assert rows == uids
  assert len(rows) == 5
