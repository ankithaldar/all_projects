#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''The wandb tracker, as a Keras callback.

The original logged to Neptune, through a client that spoke the 0.4.x
API.  The Keras port logs to **wandb**, and it does so by calling
``wandb.log`` directly rather than through ``wandb.keras.WandbCallback``,
whose internals change between versions.

The series are the original's, unchanged:

============================  ==========================================
when                          names
============================  ==========================================
every ``log_every`` batches   ``train/loss``, ``train/<loss_key>``,
                              ``train/lr``
end of a training epoch       ``train/<metric>``
end of a validation           ``valid/<metric>``
============================  ==========================================

**A wandb failure is a warning, never a fatal error.**  A tracker that
is down, throttled or misconfigured must not kill a fifty epoch run, so
every call into wandb is guarded.

The tracker works with ``WANDB_MODE=offline`` and with
``WANDB_DISABLED=true``; in the latter case wandb's own no-op run is
used and the series are dropped.
'''

import logging

from birdcall.callbacks.tf_callback_log_every import LogEveryCallback

LOGGER = logging.getLogger(__name__)


class WandbLogger(LogEveryCallback):
  '''The wandb tracker.

  Attributes:
    project_name: The wandb project.
    name: The name of the run.
    prefix_name: The wandb group of the run.
    tags: The tags of the run.
    params: The hyper-parameter payload of the run.
    run: The wandb run.  It is created when the training starts, unless
      one is given.
  '''

  def __init__(self, project_name: str = 'birdcall', name: str = 'run',
               prefix_name: str = 'birdcall', tags=None, params=None,
               log_every: int = 10, run=None) -> None:
    '''Builds the tracker.

    Args:
      project_name: The wandb project.
      name: The name of the run.
      prefix_name: The wandb group of the run.
      tags: The tags of the run.
      params: The hyper-parameter payload of the run.
      log_every: The number of batches between two logs.
      run: An existing wandb run to log to.  The tests inject one.
    '''
    super().__init__(log_every=log_every)
    self.project_name = project_name
    self.name = name
    self.prefix_name = prefix_name
    self.tags = list(tags) if tags else []
    self.params = dict(params) if params else {}
    self.run = run

  def _start(self) -> None:
    '''Creates the wandb run, if there is not one already.'''
    if self.run is not None:
      return
    try:
      import wandb  # pylint: disable=import-outside-toplevel
      self.run = wandb.init(project=self.project_name, name=self.name,
                            group=self.prefix_name, tags=self.tags,
                            config=self.params)
    except Exception as error:  # pylint: disable=broad-except
      LOGGER.warning('wandb could not be started: %s', error)
      self.run = None

  def _log(self, payload: dict, step=None) -> None:
    '''Writes one payload to wandb.

    Args:
      payload: The series of the payload.
      step: The step the payload belongs to.
    '''
    if self.run is None:
      return
    try:
      if step is None:
        self.run.log(payload)
      else:
        self.run.log(payload, step=int(step))
    except Exception as error:  # pylint: disable=broad-except
      LOGGER.warning('wandb could not log %s: %s', sorted(payload), error)

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

  def on_train_begin(self, logs=None) -> None:
    '''Starts the wandb run.

    Args:
      logs: The training logs, unused.
    '''
    super().on_train_begin(logs)
    self._start()

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
    payload = {}
    if 'loss' in logs:
      payload['train/loss'] = float(logs['loss'])
    components = getattr(self.model, 'last_components', {}) or {}
    for key, value in components.items():
      payload[f'train/{key}'] = float(value)
    learning_rate = self._learning_rate()
    if learning_rate is not None:
      payload['train/lr'] = learning_rate
    if payload:
      self._log(payload, step=self.step)

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
      self._log({f'train/{key}': value}, step=epoch)

  def on_test_end(self, logs=None) -> None:
    '''Logs the validation metrics.

    Args:
      logs: The validation logs.
    '''
    for key, value in self._scalars(logs).items():
      self._log({f'valid/{key}': value})

  def on_train_end(self, logs=None) -> None:
    '''Closes the wandb run.

    Args:
      logs: The training logs, unused.
    '''
    del logs
    if self.run is None:
      return
    try:
      finish = getattr(self.run, 'finish', None)
      if finish is not None:
        finish()
    except Exception as error:  # pylint: disable=broad-except
      LOGGER.warning('wandb could not be closed: %s', error)
