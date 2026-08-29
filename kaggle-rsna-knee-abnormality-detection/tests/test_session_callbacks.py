#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for the session-budget / done-marker contract.

Regression: PeriodicPushCallback used to write the ``done`` marker on
EVERY on_train_end, including graceful stops raised by
TimeBudgetCallback mid-schedule. An incomplete fold then looked
finished: the next session skipped it and its partially trained
weights entered the fold ensemble.
"""

# The hooks only need a tiny trainer stand-in.
# pylint: disable=invalid-name

import os

from knee.callbacks.session import PeriodicPushCallback, TimeBudgetCallback


class FakeTrainer:
  """Minimal trainer stand-in for callback hook tests."""

  def __init__(self, current_epoch, max_epochs, global_step=17):
    """Configure the epoch state the callbacks inspect.

    Args:
        current_epoch: Epoch the fit stopped at.
        max_epochs: Planned epoch count (0 = unknown/infinite).
        global_step: Optimizer steps taken so far.
    """
    self.current_epoch = current_epoch
    self.max_epochs = max_epochs
    self.global_step = global_step
    self.saved: list[str] = []

  def save_checkpoint(self, path):
    """Record checkpoint writes instead of serializing.

    Args:
        path: Requested checkpoint path.
    """
    self.saved.append(path)


class FakeModule:
  """Unused placeholder matching the hook signature."""


def _callback(tmp_path):
  return PeriodicPushCallback(
    checkpoint_dir=str(tmp_path),
    fold_id=0,
    push_every_n_epochs=0,
    client=None,
    push_slug=None,
  )


def _done_path(tmp_path):
  return os.path.join(str(tmp_path), 'fold0', 'done')


def test_completed_run_writes_done_marker(tmp_path):
  # PL counts COMPLETED epochs at on_train_end: a finished 4-epoch run
  # reports current_epoch == 4.
  trainer = FakeTrainer(current_epoch=4, max_epochs=4)
  _callback(tmp_path).on_train_end(trainer, FakeModule())
  assert os.path.exists(_done_path(tmp_path))
  with open(_done_path(tmp_path), encoding='utf-8') as handle:
    assert handle.read() == '17'


def test_budget_stopped_run_has_no_done_marker(tmp_path):
  # Stopped after epoch 0 of 4 -> 1 completed epoch -> no done marker.
  trainer = FakeTrainer(current_epoch=1, max_epochs=4)
  _callback(tmp_path).on_train_end(trainer, FakeModule())
  assert not os.path.exists(_done_path(tmp_path))
  # The checkpoint itself is still saved so the fold stays resumable.
  assert trainer.saved


def test_final_epoch_budget_stop_writes_done_marker(tmp_path):
  # Budget stop at the END of the last planned epoch: the schedule is
  # complete (3 of 4 epochs ran... no - current_epoch counts completed,
  # so a full 4-epoch run with should_stop still reports 4) -> done.
  trainer = FakeTrainer(current_epoch=4, max_epochs=4)
  trainer.should_stop = True
  _callback(tmp_path).on_train_end(trainer, FakeModule())
  assert os.path.exists(_done_path(tmp_path))


def test_unknown_max_epochs_keeps_done_marker(tmp_path):
  # max_epochs <= 0 cannot prove incompleteness; keep old behavior.
  trainer = FakeTrainer(current_epoch=7, max_epochs=0)
  _callback(tmp_path).on_train_end(trainer, FakeModule())
  assert os.path.exists(_done_path(tmp_path))


def test_disabled_marker_flag_wins(tmp_path):
  callback = PeriodicPushCallback(
    checkpoint_dir=str(tmp_path),
    fold_id=0,
    push_every_n_epochs=0,
    client=None,
    push_slug=None,
    mark_done_on_train_end=False,
  )
  trainer = FakeTrainer(current_epoch=4, max_epochs=4)
  callback.on_train_end(trainer, FakeModule())
  assert not os.path.exists(_done_path(tmp_path))
  assert trainer.saved


def test_time_budget_stops_without_writing_marker(tmp_path):
  """Both callbacks together: budget stop leaves the fold resumable."""
  budget = TimeBudgetCallback(session_time_budget_h=0.0, time_margin_min=1e6)
  trainer = FakeTrainer(current_epoch=0, max_epochs=4)
  budget.on_fit_start(trainer, FakeModule())
  budget.on_train_epoch_end(trainer, FakeModule())
  assert trainer.should_stop is True
  # PL completes the epoch bookkeeping: 1 epoch completed of 4.
  trainer.current_epoch = 1
  _callback(tmp_path).on_train_end(trainer, FakeModule())
  assert not os.path.exists(_done_path(tmp_path))
