#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''The AdamW optimiser and the batch stepped learning rate schedule.

The optimiser is the original's: ``AdamW(lr, betas=(0.9, 0.999),
eps=1e-08, weight_decay=weight_decay, amsgrad=True)``.  Keras spells
``eps`` as ``epsilon`` and ``betas`` as ``beta_1`` and ``beta_2``.

The schedule is the one place where a near miss changes the run
quietly, so the formulas are **not** re-implemented here: the shared
``birdcall.engines.schedules.schedule_multiplier`` is called, which is
the transcription of Ignite's two schedulers and the home of the golden
fixture.

Two details are load bearing:

* the schedule steps per **batch**, which is what the original's
  Ignite scheduler did (``ITERATION_STARTED``).  ``keras.optimizers``
  tf.keras.optimizers.schedules step per batch by default, but
  ``keras.optimizers.schedules.CosineDecay`` is documented per *epoch*
  and must not be used here;
* the two phase schedule is a concatenation, and the boundary is at
  exactly ``epoch_length`` steps.
'''


import tensorflow as tf

from birdcall.engines.schedules import SCHEDULE_NAMES
from birdcall.engines.schedules import schedule_multiplier


class BatchSteppedSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  '''A learning rate schedule that steps once per batch.

  Attributes:
    schedule_name: The name of the shared schedule.
    base_lr: The learning rate the multipliers are applied to.
    epoch_length: The number of batches in one epoch.
    num_epochs: The number of epochs of the run.
    scale_factor: The factor the learning rate is scaled by at both
      ends of the schedule.
  '''

  def __init__(self, schedule_name: str, base_lr: float, epoch_length: int,
               num_epochs: int, scale_factor: float,
               name: str | None = None) -> None:
    '''Builds the schedule.

    Args:
      schedule_name: One of ``none``, ``warmup_with_cosine`` or
        ``one_cycle_cosine``.
      base_lr: The learning rate the multipliers are applied to.
      epoch_length: The number of batches in one epoch.
      num_epochs: The number of epochs of the run.
      scale_factor: The factor the learning rate is scaled by.
      name: The schedule name.

    Raises:
      ValueError: If the schedule name is not known, or the epoch length
        is not positive.
    '''
    super().__init__()
    if schedule_name not in SCHEDULE_NAMES:
      raise ValueError(
          f'Unknown schedule {schedule_name!r}. '
          f'Valid names: {sorted(SCHEDULE_NAMES)}')
    if epoch_length <= 0:
      raise ValueError('epoch_length must be positive')
    self.schedule_name = schedule_name
    self.base_lr = float(base_lr)
    self.epoch_length = int(epoch_length)
    self.num_epochs = int(num_epochs)
    self.scale_factor = float(scale_factor)
    self.name = name or schedule_name

  def _multiplier(self, index):
    '''Returns the multiplier of one step.

    Args:
      index: The global step index.

    Returns:
      The learning rate multiplier.
    '''
    return schedule_multiplier(self.schedule_name, int(index),
                                self.epoch_length, self.num_epochs,
                                self.scale_factor)

  def __call__(self, step):
    '''Returns the learning rate of one step.

    Args:
      step: The global step index.

    Returns:
      The learning rate, as a float32 scalar.
    '''
    if isinstance(step, (int, float)):
      return self.base_lr * self._multiplier(int(step))
    index = tf.cast(step, tf.int32)
    # The shared implementation is the single source of truth, and it
    # is python, so it is called through `py_function`.  The call is
    # once per batch and the result is a scalar.
    multiplier = tf.py_function(self._multiplier, [index], tf.float32)
    return tf.cast(self.base_lr, tf.float32) * multiplier

  def get_config(self) -> dict:
    '''Returns the serialisable configuration of the schedule.

    Returns:
      The configuration dictionary.
    '''
    return {
        'schedule_name': self.schedule_name,
        'base_lr': self.base_lr,
        'epoch_length': self.epoch_length,
        'num_epochs': self.num_epochs,
        'scale_factor': self.scale_factor,
        'name': self.name,
    }

def build_scheduler(parameters: dict, steps_per_epoch: int
                    ) -> BatchSteppedSchedule:
  '''Builds the batch stepped learning rate schedule of the run.

  Args:
    parameters: The run parameters.  The keys ``scheduler_name``, ``lr``,
      ``lr_scale_factor`` and ``num_epochs`` are read.
    steps_per_epoch: The number of batches in one epoch.

  Returns:
    The schedule.
  '''
  return BatchSteppedSchedule(
      schedule_name=parameters.get('scheduler_name', 'warmup_with_cosine'),
      base_lr=float(parameters.get('lr', 0.001)),
      epoch_length=int(steps_per_epoch),
      num_epochs=int(parameters.get('num_epochs', 50)),
      scale_factor=float(parameters.get('lr_scale_factor', 0.01)))

def build_optimizer(parameters: dict, model=None) -> tf.keras.optimizers.AdamW:
  '''Builds the AdamW optimiser of the run.

  The Keras optimiser does not need the model's variables: the
  optimiser collects them when it minimises.  The model is therefore
  accepted for symmetry with the torch backend and is not used.

  Args:
    parameters: The run parameters.  The keys ``lr``, ``weight_decay``,
      ``amsgrad``, ``betas``, ``epsilon``, ``clipnorm`` and
      ``gradient_accumulation_steps`` are read.
    model: The model, unused.

  Returns:
    The optimiser.
  '''
  betas = parameters.get('betas', (0.9, 0.999))
  accumulation = int(parameters.get('gradient_accumulation_steps', 1))
  optimizer = tf.keras.optimizers.AdamW(
      learning_rate=float(parameters.get('lr', 0.001)),
      beta_1=float(betas[0]),
      beta_2=float(betas[1]),
      epsilon=float(parameters.get('epsilon', 1e-08)),
      weight_decay=float(parameters.get('weight_decay', 0.01)),
      amsgrad=bool(parameters.get('amsgrad', True)),
      clipnorm=parameters.get('clipnorm', None),
      # Keras only accepts an accumulation of two or more.
      gradient_accumulation_steps=accumulation if accumulation > 1 else None,
      name=parameters.get('optimizer_name', 'adamw'))
  return optimizer
