#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Tests of the every-N-steps logging callback.

The original filtered the event per handler, so the series were written
every N iterations and not every iteration.  ``Trainer(log_every_n_steps)``
is not that: it is a global knob, so the counter is hand rolled and
tested here.
'''

import pytest
import torch

from birdcall.callbacks.pl_callback_log_every import LogEveryN


class FakeLogger:
  '''A logger that remembers what it was asked to write.

  Attributes:
    written: The series of every call.
  '''

  def __init__(self) -> None:
    '''Builds the logger.'''
    self.written: list[tuple[dict, int]] = []

  def log_metrics(self, metrics: dict, step=None) -> None:
    '''Records the series.

    Args:
      metrics: The series.
      step: The step they belong to.
    '''
    self.written.append((metrics, step))


class FakeTrainer:
  '''A trainer with a logger and a step counter.

  Attributes:
    global_step: The number of steps that were run.
    logger: The logger of the run.
  '''

  def __init__(self, logger=None) -> None:
    '''Builds the trainer.

    Args:
      logger: The logger of the run.
    '''
    self.global_step = 0
    self.logger = logger


class FakeModule:
  '''A module with one optimiser.

  Attributes:
    optimizer: The optimiser of the run.
  '''

  def __init__(self, optimizer) -> None:
    '''Builds the module.

    Args:
      optimizer: The optimiser of the run.
    '''
    self.optimizer = optimizer

  def optimizers(self):
    '''Returns the optimisers of the run.'''
    return [self.optimizer]


def _module() -> FakeModule:
  '''Builds a module with an optimiser at a known learning rate.

  Returns:
    The module.
  '''
  parameter = torch.nn.Parameter(torch.zeros(1))
  optimizer = torch.optim.AdamW([parameter], lr=0.001)
  return FakeModule(optimizer)


def test_the_callback_fires_every_n_steps_and_not_every_step():
  '''The counter is the original's per-handler filter.'''
  callback = LogEveryN(log_every=3)
  trainer = FakeTrainer()
  module = _module()
  for index in range(7):
    callback.on_train_batch_end(trainer, module, torch.tensor(1.0), None,
                                index)
  assert callback.steps == 7
  assert callback.logged_steps == [3, 6]


def test_the_callback_does_not_fire_before_the_first_multiple():
  '''Nothing is written before the cadence is reached.'''
  callback = LogEveryN(log_every=5)
  trainer = FakeTrainer()
  module = _module()
  for index in range(4):
    callback.on_train_batch_end(trainer, module, torch.tensor(1.0), None,
                                index)
  assert callback.logged_steps == []
  assert callback.last_loss is None


def test_the_callback_writes_the_original_series():
  '''The two series are the original's per-iteration series.'''
  logger = FakeLogger()
  callback = LogEveryN(log_every=2)
  trainer = FakeTrainer(logger)
  module = _module()
  for index in range(2):
    trainer.global_step = index
    callback.on_train_batch_end(trainer, module, torch.tensor(0.5), None,
                                index)
  assert len(logger.written) == 1
  metrics, step = logger.written[0]
  assert metrics['train/loss'] == pytest.approx(0.5)
  assert metrics['train/lr'] == pytest.approx(0.001)
  assert step == 1


def test_the_callback_reads_a_loss_out_of_a_dict():
  '''A step that returns a dict is understood.'''
  callback = LogEveryN(log_every=1)
  callback.on_train_batch_end(FakeTrainer(), _module(),
                              {'loss': torch.tensor(2.0)}, None, 0)
  assert callback.last_loss == pytest.approx(2.0)


def test_the_callback_survives_a_run_without_a_logger():
  '''A run without a tracker still counts its steps.'''
  callback = LogEveryN(log_every=1)
  callback.on_train_batch_end(FakeTrainer(logger=None), _module(),
                              torch.tensor(1.0), None, 0)
  assert callback.logged_steps == [1]


def test_a_cadence_below_one_is_a_clear_error():
  '''The cadence has to be a positive number.'''
  with pytest.raises(ValueError, match='log_every'):
    LogEveryN(log_every=0)
