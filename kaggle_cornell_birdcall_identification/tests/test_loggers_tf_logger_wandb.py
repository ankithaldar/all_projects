#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''The tests of the wandb tracker.

A wandb failure must be a warning, never a fatal error, and the
tracker must work with ``WANDB_MODE=offline`` and with
``WANDB_DISABLED=true``.
'''

import pytest

from birdcall.loggers.tf_logger_wandb import WandbLogger


class FakeRun:
  '''A wandb run that only records what it is given.'''

  def __init__(self, fail: bool = False) -> None:
    self.payloads = []
    self.finished = False
    self.fail = fail

  def log(self, payload, step=None):
    '''Records one payload.

    Args:
      payload: The series of the payload.
      step: The step the payload belongs to.

    Raises:
      RuntimeError: If the run was built to fail.
    '''
    if self.fail:
      raise RuntimeError('wandb is down')
    self.payloads.append((dict(payload), step))

  def finish(self) -> None:
    '''Closes the run.'''
    self.finished = True


class FakeModel:
  '''A model with an optimiser and the last loss components.'''

  class Optimizer:
    '''An optimiser with a fixed learning rate.'''

    learning_rate = 0.001

  def __init__(self) -> None:
    self.optimizer = self.Optimizer()
    self.last_components = {'bce_loss': 0.5}


def logger(run=None, log_every: int = 2) -> WandbLogger:
  '''Builds the tracker of the test.

  Args:
    run: The run to log to.
    log_every: The number of batches between two logs.

  Returns:
    The tracker.
  '''
  return WandbLogger(project_name='birdcall', name='run',
                     prefix_name='group', tags=['tag'], params={'lr': 0.001},
                     log_every=log_every, run=run)


def test_the_run_is_not_started_when_one_is_given() -> None:
  '''An injected run is used as it is.'''
  run = FakeRun()
  tracker = logger(run)
  tracker.on_train_begin({})
  assert tracker.run is run


def test_the_training_series_are_logged_every_n_batches() -> None:
  '''The per-iteration series are the original's.'''
  run = FakeRun()
  tracker = logger(run, log_every=2)
  tracker.set_model(FakeModel())
  for _ in range(4):
    tracker.on_batch_end(0, {'loss': 0.25})
  assert len(run.payloads) == 2
  payload, step = run.payloads[0]
  assert payload['train/loss'] == pytest.approx(0.25)
  assert payload['train/bce_loss'] == pytest.approx(0.5)
  assert payload['train/lr'] == pytest.approx(0.001)
  assert step == 2


def test_the_epoch_metrics_are_logged() -> None:
  '''The per-epoch series are the original's.'''
  run = FakeRun()
  tracker = logger(run)
  tracker.on_epoch_end(4, {'f1score_clip': 0.75})
  payload, step = run.payloads[-1]
  assert payload == {'train/f1score_clip': 0.75}
  assert step == 4


def test_the_validation_metrics_are_logged() -> None:
  '''The validation series are the original's.'''
  run = FakeRun()
  tracker = logger(run)
  tracker.on_test_end({'lraps': 0.25})
  payload, _ = run.payloads[-1]
  assert payload == {'valid/lraps': 0.25}


def test_a_failing_wandb_is_only_a_warning() -> None:
  '''A tracker that is down must not kill the run.'''
  run = FakeRun(fail=True)
  tracker = logger(run)
  tracker.set_model(FakeModel())
  tracker.on_batch_end(0, {'loss': 0.25})
  tracker.on_epoch_end(0, {'loss': 0.25})
  tracker.on_test_end({'lraps': 0.25})
  tracker.on_train_end({})


def test_the_run_is_closed_at_the_end_of_the_training() -> None:
  '''The wandb run is finished when the training ends.'''
  run = FakeRun()
  tracker = logger(run)
  tracker.on_train_end({})
  assert run.finished is True


def test_the_tracker_works_without_a_run() -> None:
  '''A tracker that could not start does not raise.'''
  tracker = WandbLogger(run=None)
  tracker.set_model(FakeModel())
  tracker.on_batch_end(0, {'loss': 0.25})
  tracker.on_epoch_end(0, {'loss': 0.25})
  tracker.on_test_end({'lraps': 0.25})
  tracker.on_train_end({})


def test_the_tracker_works_when_wandb_is_disabled(monkeypatch) -> None:
  '''WANDB_DISABLED=true is honoured, and the run is a no-op.'''
  monkeypatch.setenv('WANDB_DISABLED', 'true')
  tracker = WandbLogger(run=None)
  tracker.set_model(FakeModel())
  tracker.on_train_begin({})
  tracker.on_batch_end(0, {'loss': 0.25})
  tracker.on_train_end({})


def test_the_tracker_works_offline(monkeypatch, tmp_path) -> None:
  '''WANDB_MODE=offline is honoured.'''
  monkeypatch.setenv('WANDB_MODE', 'offline')
  monkeypatch.setenv('WANDB_DIR', str(tmp_path))
  tracker = WandbLogger(run=None, name='offline-run')
  tracker.set_model(FakeModel())
  tracker.on_train_begin({})
  tracker.on_batch_end(0, {'loss': 0.25})
  tracker.on_train_end({})
