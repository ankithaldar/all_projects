#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''The tests of the every-N logging callback.

The original's ``Events.ITERATION_COMPLETED(every=N)`` filter has no
Keras equivalent, so the counter is hand rolled and tested as such.
'''

import pytest

from birdcall.callbacks.tf_callback_log_every import LogEveryCallback


def test_the_counter_starts_at_zero() -> None:
  '''A fresh callback has not seen a batch.'''
  callback = LogEveryCallback(log_every=3)
  assert callback.step == 0
  assert callback.should_log() is False


def test_the_callback_fires_every_n_batches() -> None:
  '''The fire happens on every N-th batch, and not before.'''
  callback = LogEveryCallback(log_every=3)
  fired = []
  for _ in range(7):
    callback.on_batch_end(0, {'loss': 1.0})
    if callback.should_log():
      fired.append(callback.step)
  assert fired == [3, 6]


def test_a_single_batch_does_not_fire() -> None:
  '''The very first batch is not a logging step.'''
  callback = LogEveryCallback(log_every=1)
  callback.on_batch_end(0, {'loss': 1.0})
  assert callback.should_log() is True


def test_the_counter_resets_at_the_start_of_the_training() -> None:
  '''A second run starts from zero again.'''
  callback = LogEveryCallback(log_every=2)
  callback.on_batch_end(0, {'loss': 1.0})
  callback.on_train_begin({'loss': 0.0})
  assert callback.step == 0


def test_a_non_positive_interval_is_refused() -> None:
  '''An interval of zero would never fire.'''
  with pytest.raises(ValueError, match='log_every'):
    LogEveryCallback(log_every=0)
