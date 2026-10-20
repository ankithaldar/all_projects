#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''The ``every=N`` event filter, as a Keras callback.

The original registered its handlers on ``Events.ITERATION_COMPLETED
(every=log_every)``.  That filter is **per handler**, and neither
Lightning nor Keras has an equivalent: a callback sees every event and
has to keep its own counter.  This module is that counter.

The counter is deliberately separate from the loggers: the original's
print logger and its Neptune logger had different cadences, and the
counter is the only part they share.
'''

import tensorflow as tf


class LogEveryCallback(tf.keras.callbacks.Callback):
  '''Counts the batches and fires every N of them.

  Attributes:
    log_every: The number of batches between two firings.
    step: The number of batches seen so far.
  '''

  def __init__(self, log_every: int = 10) -> None:
    '''Builds the counter.

    Args:
      log_every: The number of batches between two firings.

    Raises:
      ValueError: If the interval is not positive.
    '''
    super().__init__()
    if log_every <= 0:
      raise ValueError('log_every must be positive')
    self.log_every = int(log_every)
    self.step = 0

  def on_train_begin(self, logs=None) -> None:
    '''Resets the counter.

    Args:
      logs: The training logs, unused.
    '''
    del logs
    self.step = 0

  def on_batch_end(self, batch, logs=None) -> None:
    '''Counts the batch.

    Args:
      batch: The index of the batch.
      logs: The batch logs, unused.
    '''
    del batch, logs
    self.step += 1

  def should_log(self) -> bool:
    '''Whether the current step is a logging step.

    Returns:
      True on every N-th batch.  The zeroth batch is not one: nothing
      has happened yet.
    '''
    return self.step > 0 and self.step % self.log_every == 0
