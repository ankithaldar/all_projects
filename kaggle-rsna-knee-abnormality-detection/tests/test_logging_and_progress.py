#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for the production logging bootstrap + progress callback.

Covers: tee mirroring, idempotent re-setup, handler dedup, progress
line cadence/format, and GPU-off degradation.
"""

# pylint: disable=protected-access,redefined-outer-name

import logging
import os
import traceback

import pytest

from knee.helpers.logging_setup import (
  current_log_path,
  reset_logging,
  setup_logging,
)
from knee.loggers.progress import ProgressLogCallback


class FakeModule:
  """Module surface the progress callback's LR probe reads."""

  def optimizers(self):
    return [type('O', (), {'param_groups': [{'lr': 2.5e-4}]})()]


@pytest.fixture()
def log_env(tmp_path):
  """Fresh log dir + cleanup of global logging state per test."""
  log_dir = tmp_path / 'logs'
  yield log_dir
  reset_logging()


def test_setup_logging_writes_file_and_mirrors_streams(log_env, capsys):
  path = setup_logging(str(log_env), stage='train', experiment_name='exp')
  assert os.path.exists(path)
  assert current_log_path() == path
  logging.getLogger('probe').info('file-line-%d', 1)
  print('stream-line-2')
  content = open(path, encoding='utf-8').read()
  assert 'file-line-1' in content
  assert 'stream-line-2' in content
  assert '=== knee stage=train' in content
  captured = capsys.readouterr()
  assert 'stream-line-2' in captured.out  # original stream still works


def test_double_setup_rebinds_without_duplicating_handlers(log_env):
  first = setup_logging(str(log_env), 'train', 'exp')
  before = len(logging.getLogger().handlers)
  second = setup_logging(str(log_env), 'infer', 'exp')
  assert first != second
  after = len(logging.getLogger().handlers)
  # One file handler per setup call, never duplicated per call.
  assert after == before + 1
  print('second-stage-line')
  assert 'second-stage-line' in open(second, encoding='utf-8').read()


def test_stderr_mirror_captures_tracebacks(log_env):
  setup_logging(str(log_env), 'cache', 'exp')
  try:
    raise ValueError('boom-traceback')
  except ValueError:
    traceback.print_exc()
  content = open(current_log_path(), encoding='utf-8').read()
  assert 'boom-traceback' in content


class FakeProgressTrainer:
  """Trainer surface the progress callback reads."""

  def __init__(self, step, epoch=0, batches=441):
    self.global_step = step
    self.current_epoch = epoch
    self.num_training_batches = batches
    self.callback_metrics = {
      'train/loss': 0.31,
      'val/auc_macro': 0.8,
      'val/loss': 0.4,
    }
    self.is_global_zero = True
    self.sanity_checking = False


class FakeOptModule(FakeModule):
  def optimizers(self):
    return [type('O', (), {'param_groups': [{'lr': 2.5e-4}]})()]


def test_progress_line_every_n_steps(caplog):
  cb = ProgressLogCallback(log_every_n_steps=2, log_gpu_mem=False)
  trainer = FakeProgressTrainer(step=0)
  module = FakeOptModule()
  trainer.global_step = 2
  with caplog.at_level(logging.INFO, logger='progress'):
    cb.on_train_batch_end(trainer, module, None, None, 1)
    trainer.global_step = 3
    cb.on_train_batch_end(trainer, module, None, None, 2)  # not multiple
    trainer.global_step = 4
    cb.on_train_batch_end(trainer, module, None, None, 3)
  lines = [r.message for r in caplog.records if r.name == 'progress']
  assert len(lines) == 2
  expected = (
      'epoch 0 | batch 2/441 | step 2 | train_loss 0.3100 | lr 0.0003'
  )
  assert expected in lines[0]
  assert 'gpu off' in lines[0]


def test_progress_epoch_and_validation_lines(caplog):
  cb = ProgressLogCallback(log_every_n_steps=1, log_gpu_mem=False)
  trainer = FakeProgressTrainer(step=1, epoch=3)
  module = FakeOptModule()
  with caplog.at_level(logging.INFO, logger='progress'):
    cb.on_validation_epoch_end(trainer, module)
    cb.on_train_epoch_end(trainer, module)
  msgs = [r.message for r in caplog.records if r.name == 'progress']
  assert any('val/auc_macro 0.8000' in m for m in msgs)
  assert any('train epoch complete' in m for m in msgs)


def test_progress_silent_on_non_zero_rank(caplog):
  cb = ProgressLogCallback(log_every_n_steps=1, log_gpu_mem=False)
  trainer = FakeProgressTrainer(step=1)
  trainer.is_global_zero = False
  with caplog.at_level(logging.INFO, logger='progress'):
    cb.on_train_batch_end(trainer, FakeOptModule(), None, None, 0)
  assert not [r for r in caplog.records if r.name == 'progress']


if __name__ == '__main__':
  pytest.main([__file__])
