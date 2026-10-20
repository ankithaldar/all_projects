#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''The tests of the console tracker.

The series are the original's, and the format is the original's: the
name, the step and the value, one series per line.
'''

import io

import pytest

from birdcall.loggers.tf_logger_print import PrintLogger


class FakeModel:
  '''A model with an optimiser and the last loss components.'''

  class Optimizer:
    '''An optimiser with a fixed learning rate.'''

    learning_rate = 0.002

  def __init__(self) -> None:
    self.optimizer = self.Optimizer()
    self.last_components = {'bce_loss': 0.5, 'F_loss': 0.5}


def tracker(log_every: int = 2) -> tuple[PrintLogger, io.StringIO]:
  '''Builds the tracker of the test.

  Args:
    log_every: The number of batches between two logs.

  Returns:
    A tuple ``(tracker, stream)``.
  '''
  stream = io.StringIO()
  return PrintLogger(log_every=log_every, stream=stream), stream


def test_the_training_series_are_printed_every_n_batches() -> None:
  '''The per-iteration series are the original's.'''
  logger, stream = tracker(log_every=2)
  logger.set_model(FakeModel())
  for _ in range(4):
    logger.on_batch_end(0, {'loss': 0.25})
  lines = stream.getvalue().splitlines()
  # Two logging steps, and four series each.
  assert len(lines) == 8
  assert lines[0].startswith('train/loss 2 0.25')
  assert any(line.startswith('train/bce_loss') for line in lines)
  assert any(line.startswith('train/lr') for line in lines)


def test_the_epoch_metrics_are_printed() -> None:
  '''The per-epoch series are the original's.'''
  logger, stream = tracker()
  logger.on_epoch_end(2, {'f1score_clip': 0.75})
  assert stream.getvalue().strip() == 'train/f1score_clip 2 0.75'


def test_the_validation_metrics_are_printed() -> None:
  '''The validation series are the original's.'''
  logger, stream = tracker()
  logger.on_test_end({'lraps': 0.25})
  assert stream.getvalue().strip() == 'valid/lraps None 0.25'


def test_nothing_is_printed_between_the_logging_steps() -> None:
  '''The cadence is the every-N filter of the original.'''
  logger, stream = tracker(log_every=5)
  logger.set_model(FakeModel())
  for _ in range(4):
    logger.on_batch_end(0, {'loss': 0.25})
  assert stream.getvalue() == ''


def test_the_tracker_works_without_a_model() -> None:
  '''A tracker without a model does not raise.'''
  logger, stream = tracker(log_every=1)
  logger.on_batch_end(0, {'loss': 0.25})
  assert 'train/loss' in stream.getvalue()


def test_the_learning_rate_is_printed() -> None:
  '''The learning rate is part of the training series.'''
  logger, stream = tracker(log_every=1)
  logger.set_model(FakeModel())
  logger.on_batch_end(0, {'loss': 0.25})
  assert 'train/lr 1 0.002' in stream.getvalue()


def test_the_default_stream_is_stdout() -> None:
  '''The tracker writes to stdout by default.'''
  import sys  # pylint: disable=import-outside-toplevel
  assert PrintLogger().stream is sys.stdout
