#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''The every-N-steps logging callback.

The original registered ``Events.ITERATION_COMPLETED(every=log_every)``,
which is a **per-handler** filter: every handler of the event saw only
every N-th iteration.  Lightning has no equivalent -- ``log_every_n_steps``
is a global knob of the trainer, not a filter of one handler -- so the
counter is hand rolled.

The callback writes the two series the original wrote every N iterations:
``train/loss`` and ``train/lr``.
'''

import logging

from lightning.pytorch import Callback

LOGGER = logging.getLogger(__name__)

LOSS_SERIES = 'train/loss'
LR_SERIES = 'train/lr'


class LogEveryN(Callback):
  '''Logs the loss and the learning rate every N training steps.

  Attributes:
    log_every: The number of steps between two logs.
    steps: The number of training steps seen so far.
    logged_steps: The step numbers that were logged.
    last_loss: The loss of the last logged step.
    last_lr: The learning rate of the last logged step.
  '''

  def __init__(self, log_every: int = 10) -> None:
    '''Builds the callback.

    Args:
      log_every: The number of steps between two logs.

    Raises:
      ValueError: If the cadence is not a positive number.
    '''
    super().__init__()
    if int(log_every) < 1:
      raise ValueError(
          f'log_every must be a positive number, got {log_every!r}.')
    self.log_every = int(log_every)
    self.steps = 0
    self.logged_steps: list[int] = []
    self.last_loss = None
    self.last_lr = None

  def on_train_batch_end(self, trainer, pl_module, outputs, batch,
                         batch_idx) -> None:
    '''Logs the loss and the learning rate, every N steps.

    Args:
      trainer: The trainer.
      pl_module: The module of the run.
      outputs: What the training step returned.
      batch: The batch of the step.
      batch_idx: The index of the batch.
    '''
    self.steps += 1
    if self.steps % self.log_every:
      return
    loss = self._loss_of(outputs)
    lr = self._learning_rate(pl_module)
    self.last_loss = loss
    self.last_lr = lr
    self.logged_steps.append(self.steps)
    metrics = {}
    if loss is not None:
      metrics[LOSS_SERIES] = loss
    if lr is not None:
      metrics[LR_SERIES] = lr
    if metrics:
      self._write(trainer, metrics)

  def _write(self, trainer, metrics: dict) -> None:
    '''Writes the series to the logger of the run.

    Args:
      trainer: The trainer.
      metrics: The series to write.
    '''
    logger = getattr(trainer, 'logger', None)
    if logger is None:
      return
    try:
      logger.log_metrics(metrics, step=trainer.global_step)
    except Exception as error:  # pylint: disable=broad-except
      LOGGER.warning('Could not write the step metrics: %s', error)

  def _loss_of(self, outputs):
    '''Returns the loss of a training step.

    Args:
      outputs: What the training step returned.

    Returns:
      The loss as a float, or ``None`` when the step returned none.
    '''
    if outputs is None:
      return None
    if isinstance(outputs, dict):
      outputs = outputs.get('loss')
    if outputs is None:
      return None
    if hasattr(outputs, 'detach'):
      outputs = outputs.detach()
    if hasattr(outputs, 'item'):
      outputs = outputs.item()
    return float(outputs)

  def _learning_rate(self, pl_module):
    '''Returns the learning rate of the run.

    Args:
      pl_module: The module of the run.

    Returns:
      The learning rate of the first optimiser, or ``None`` when there is
      none.
    '''
    try:
      optimizers = pl_module.optimizers()
    except (RuntimeError, AttributeError):
      return None
    if not optimizers:
      return None
    return float(optimizers[0].param_groups[0]['lr'])
