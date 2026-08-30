#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Regression tests for per-epoch metric isolation in KneeModule.

Three historical bugs are guarded here:

1. MultilabelAUC was never reset between validation epochs, so
   Lightning's sanity-check batches and every prior epoch's predictions
   accumulated into the OOF csv and the logged val/auc_macro.
2. The OOF csv was overwritten by every DDP rank, so under 2xT4 the
   file held only one rank's val-shard instead of the union.
3. (Ensured by the assemble helper) the canonical OOF needs stable
   ordering across epochs for downstream analysis.
"""

# Only the metric lifecycle hooks are exercised; model/loss are inert
# placeholders because validation hooks never call them.
# pylint: disable=invalid-name

import os

import numpy as np
import pandas as pd

from knee.engines.train_module import (
  OOF_SHARD_DIR,
  KneeModule,
  assemble_oof_canonical,
)

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


def test_shards_are_cleaned_after_canonical_write(tmp_path):
  """No shard files leak into the pushed checkpoint directory."""
  module = _module(tmp_path)
  _update(module, ['a', 'b'])
  module.on_validation_epoch_end()
  shard_dir = os.path.join(str(tmp_path), OOF_SHARD_DIR)
  leftovers = (
    [name for _, _, names in os.walk(shard_dir) for name in names]
    if os.path.isdir(shard_dir)
    else []
  )
  assert leftovers == []


class _FakeStrategy:
  """Records barrier calls (stand-in for PL's Strategy)."""

  def __init__(self):
    self.barriers = 0

  def barrier(self):
    self.barriers += 1


class _FakeTrainer:
  """Minimal trainer exposing rank/world_size/strategy/sanity flag."""

  def __init__(self, rank, world_size, sanity_checking=False):
    self.global_rank = rank
    self.world_size = world_size
    self.sanity_checking = sanity_checking
    self.barebones = False
    self.strategy = _FakeStrategy()


def test_ddp_ranks_merge_into_one_canonical(tmp_path):
  """Both ranks' shards must appear in the canonical OOF exactly once.

  Regression: every rank used to overwrite the same canonical file from
  its own val-shard, so under 2xT4 the OOF held only half the val set.
  """
  rank0 = _module(tmp_path, fold=0)
  rank0._trainer = _FakeTrainer(rank=0, world_size=2)
  rank1 = KneeModule(
    model=object(),
    criterion=object(),
    optimizer_cfg={'class_path': 'torch.optim.Adam', 'init_params': {}},
    scheduler_cfg=None,
    warmup_epochs=0,
    backbone_lr_scale=0.1,
    total_epochs=1,
    target_columns=TARGETS,
    oof_dir=str(tmp_path),
    fold_id=0,
  )
  rank1._trainer = _FakeTrainer(rank=1, world_size=2)

  rank0.on_validation_epoch_start()
  rank1.on_validation_epoch_start()
  _update(rank0, ['val_a', 'val_b'])  # rank-0 shard
  _update(rank1, ['val_c', 'val_d'])  # rank-1 shard
  # Real DDP order: the barrier releases all ranks after every shard
  # is on disk; rank 0's assembly runs LAST. Simulate by finishing
  # rank 1's hook before rank 0's.
  rank1.on_validation_epoch_end()
  assert rank1._trainer.strategy.barriers == 1
  rank0.on_validation_epoch_end()
  assert rank0._trainer.strategy.barriers == 1

  rows = _oof_rows(tmp_path, fold=0)
  assert rows == ['val_a', 'val_b', 'val_c', 'val_d']
  # Shards are consumed by the canonical write; nothing left behind.
  assert not os.path.isdir(os.path.join(str(tmp_path), OOF_SHARD_DIR)) or (
    not os.listdir(os.path.join(str(tmp_path), OOF_SHARD_DIR))
  )


def test_sanity_checking_skips_oof_write(tmp_path):
  """Throwaway sanity predictions never publish a canonical OOF."""
  module = _module(tmp_path)
  module._trainer = _FakeTrainer(rank=0, world_size=1, sanity_checking=True)
  _update(module, ['sanity_only'])
  module.on_validation_epoch_end()
  assert not os.path.exists(os.path.join(str(tmp_path), 'oof_fold3.csv'))


def test_empty_validation_overwrites_stale_canonical(tmp_path):
  """A zero-row val epoch must not leave the previous OOF in place."""
  module = _module(tmp_path)
  _update(module, ['stale_a', 'stale_b'])
  module.on_validation_epoch_end()
  assert len(_oof_rows(tmp_path)) == 2
  module.on_validation_epoch_start()  # reset -> metric is empty
  module.on_validation_epoch_end()
  assert len(_oof_rows(tmp_path)) == 0


class TestAssembleOofCanonical:
  """Unit tests for the shard concatenation helper."""

  def _write_shard(self, tmp_path, rank, uids):
    frame = pd.DataFrame(
      {
        'StudyInstanceUID': uids,
        'fold': [0] * len(uids),
        'ACL_prob': [0.1] * len(uids),
        'ACL': [1.0] * len(uids),
        'Fracture_prob': [0.2] * len(uids),
        'Fracture': [0.0] * len(uids),
      }
    )
    shard_dir = os.path.join(str(tmp_path), OOF_SHARD_DIR)
    os.makedirs(shard_dir, exist_ok=True)
    frame.to_csv(
      os.path.join(shard_dir, f'oof_fold0_rank{rank}.csv'), index=False
    )

  def test_union_across_ranks(self, tmp_path):
    self._write_shard(tmp_path, 0, ['s1', 's2'])
    self._write_shard(tmp_path, 1, ['s3'])
    merged = assemble_oof_canonical(str(tmp_path), 0)
    assert merged['StudyInstanceUID'].tolist() == ['s1', 's2', 's3']

  def test_duplicates_deduped_last_rank_wins(self, tmp_path):
    # DistributedSampler pads uneven shards by REPEATING samples, so
    # the same study can appear on multiple ranks with identical
    # predictions; the OOF must keep one row per study.
    self._write_shard(tmp_path, 0, ['s1', 's1'])
    self._write_shard(tmp_path, 1, ['s1'])
    merged = assemble_oof_canonical(str(tmp_path), 0)
    assert merged['StudyInstanceUID'].tolist() == ['s1']

  def test_no_shards_returns_none(self, tmp_path):
    assert assemble_oof_canonical(str(tmp_path), 0) is None

  def test_other_fold_shards_ignored(self, tmp_path):
    self._write_shard(tmp_path, 0, ['s1'])
    merged = assemble_oof_canonical(str(tmp_path), 7)
    assert merged is None
