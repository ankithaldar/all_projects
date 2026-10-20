#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''The console tracker, as a Keras callback.

The original's print logger wrote the same series as its Neptune
logger, to stdout, at the same cadence:

============================  ==========================================
when                          names
============================  ==========================================
every ``log_every`` batches   ``train/loss``, ``train/<loss_key>``,
                              ``train/lr``
end of a training epoch       ``train/<metric>``
end of a validation           ``valid/<metric>``
============================  ==========================================

The format is the original's: the name, the step and the value, one
series per line.
'''

import sys

from birdcall.callbacks.tf_callback_log_every import LogEveryCallback


class PrintLogger(LogEveryCallback):
  '''The console tracker.

  Attributes:
    stream: The stream the series are written to.  The default is
      stdout.
  '''

  def __init__(self, log_every: int = 10, stream=None) -> None:
    '''Builds the tracker.

    Args:
      log_every: The number of batches between two logs.
      stream: The stream the series are written to.
    '''
    super().__init__(log_every=log_every)
    self.stream = stream if stream is not None else sys.stdout

  def _write(self, name: str, step, value) -> None:
    '''Writes one series.

    Args:
      name: The name of the series.
      step: The step the value belongs to.
      value: The value.
    '''
    print(name, step, value, file=self.stream)

  @staticmethod
  def _scalars(logs: dict) -> dict:
    '''Keeps the entries of a log dict that are numbers.

    The evaluation logs carry the predictions and the file names as
    well, and they are not numbers.

    Args:
      logs: The logs of a batch or of an epoch.

    Returns:
      The numeric entries of the logs.
    '''
    values = {}
    for key, value in (logs or {}).items():
      try:
        values[key] = float(value)
      except (TypeError, ValueError):
        continue
    return values

  def _learning_rate(self):
    '''Returns the learning rate of the optimiser.

    Returns:
      The learning rate, or ``None`` when there is no optimiser.
    '''
    optimizer = getattr(self.model, 'optimizer', None)
    if optimizer is None:
      return None
    try:
      return float(optimizer.learning_rate)
    except (TypeError, ValueError):
      return None

  def on_batch_end(self, batch, logs=None) -> None:
    '''Logs the training series, every ``log_every`` batches.

    Args:
      batch: The index of the batch.
      logs: The batch logs.
    '''
    super().on_batch_end(batch, logs)
    if not self.should_log():
      return
    logs = logs or {}
    if 'loss' in logs:
      self._write('train/loss', self.step, float(logs['loss']))
    components = getattr(self.model, 'last_components', {}) or {}
    for key, value in components.items():
      self._write(f'train/{key}', self.step, float(value))
    learning_rate = self._learning_rate()
    if learning_rate is not None:
      self._write('train/lr', self.step, learning_rate)

  def on_epoch_end(self, epoch, logs=None) -> None:
    '''Logs the training epoch metrics.

    Args:
      epoch: The index of the epoch.
      logs: The epoch logs.
    '''
    for key, value in self._scalars(logs).items():
      if key.startswith('val_'):
        # The validation series are logged by `on_test_end`.
        continue
      self._write(f'train/{key}', epoch, value)

  def on_test_end(self, logs=None) -> None:
    '''Logs the validation metrics.

    Args:
      logs: The validation logs.
    '''
    for key, value in self._scalars(logs).items():
      self._write(f'valid/{key}', None, value)
