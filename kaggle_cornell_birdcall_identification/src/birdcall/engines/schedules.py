#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Learning rate schedules, transcribed from PyTorch Ignite.

The formulas are taken from ``pytorch-ignite 0.4.0.post1``,
``ignite/contrib/handlers/param_scheduler.py``.  They are reproduced here
because the schedule is the one place where a near-miss changes the run
quietly: the original's curve is not a curve you can eyeball.

Two schedules exist, matching the original's two:

``warmup_with_cosine``
    A linear phase for exactly ``epoch_length`` iterations, then a cosine
    phase whose own counter restarts at zero.  The linear phase runs over a
    two-epoch triangle, so only its rising half is traversed.

``one_cycle_cosine``
    The cosine phase alone, over the whole run.

Every function returns a **multiplier** relative to the base learning rate,
so that both backends can apply the same schedule to their own optimiser.

The golden fixture is published in
``docs/ports/pytorch-lightning-guide.md``: for ``lr=0.001``,
``lr_scale_factor=0.01``, ``epoch_length=7`` and ``num_epochs=5`` the
iterations 0, 1, 2, 33 and 34 give ``1e-5``, ``0.00015143``, ``0.00029286``,
``0.00016292`` and ``0.00013223``.
'''

import math

SCHEDULE_NONE = 'none'
SCHEDULE_WARMUP_COSINE = 'warmup_with_cosine'
SCHEDULE_ONE_CYCLE_COSINE = 'one_cycle_cosine'

SCHEDULE_NAMES = (
    SCHEDULE_NONE,
    SCHEDULE_WARMUP_COSINE,
    SCHEDULE_ONE_CYCLE_COSINE,
)


def linear_multiplier(index: int, cycle_size: int, start_multiplier: float,
                      end_multiplier: float) -> float:
  '''The linear cyclical schedule of ``LinearCyclicalScheduler``.

  Args:
    index: The event index within the phase.
    cycle_size: The length of the full triangle.
    start_multiplier: The value at the top of the triangle.
    end_multiplier: The value at the bottom of the triangle.

  Returns:
    The learning rate multiplier for this index.
  '''
  progress = index / cycle_size
  return (end_multiplier +
          (start_multiplier - end_multiplier) * abs(progress - 0.5) * 2)


def cosine_multiplier(index: int, cycle_size: int, start_multiplier: float,
                      end_multiplier: float) -> float:
  '''The cosine schedule of ``CosineAnnealingScheduler``.

  Args:
    index: The event index within the phase.
    cycle_size: The length of the full cosine.
    start_multiplier: The value at the start of the cosine.
    end_multiplier: The value at the end of the cosine.

  Returns:
    The learning rate multiplier for this index.
  '''
  progress = index / cycle_size
  return (start_multiplier +
          (end_multiplier - start_multiplier) / 2 *
          (1 - math.cos(math.pi * progress)))


def warmup_with_cosine_multiplier(index: int, epoch_length: int,
                                  num_epochs: int,
                                  scale_factor: float) -> float:
  '''The ``warmup_with_cosine`` multiplier for one step.

  The linear phase covers the first ``epoch_length`` steps with its own
  counter; the cosine phase follows with its counter restarting at zero.
  The boundary is deliberately not smoothed: the original's curve steps
  from the last linear value straight to the cosine phase's first value.

  Args:
    index: The global step index.
    epoch_length: The number of iterations in one epoch.
    num_epochs: The total number of epochs in the run.
    scale_factor: The factor the learning rate is scaled by at both ends.

  Returns:
    The learning rate multiplier for this step.
  '''
  if index < epoch_length:
    return linear_multiplier(index, 2 * epoch_length, scale_factor, 1.0)
  return cosine_multiplier(index - epoch_length, num_epochs * epoch_length,
                            1.0, scale_factor)


def one_cycle_cosine_multiplier(index: int, num_epochs: int,
                                epoch_length: int,
                                scale_factor: float) -> float:
  '''The ``one_cycle_cosine`` multiplier for one step.

  Args:
    index: The global step index.
    num_epochs: The total number of epochs in the run.
    epoch_length: The number of iterations in one epoch.
    scale_factor: The factor the learning rate ends at.

  Returns:
    The learning rate multiplier for this step.
  '''
  return cosine_multiplier(index, num_epochs * epoch_length, 1.0,
                            scale_factor)


def schedule_multiplier(name: str, index: int, epoch_length: int,
                        num_epochs: int, scale_factor: float) -> float:
  '''Returns the learning rate multiplier for a step of the named schedule.

  Args:
    name: One of ``none``, ``warmup_with_cosine`` or ``one_cycle_cosine``.
    index: The global step index.
    epoch_length: The number of iterations in one epoch.
    num_epochs: The total number of epochs in the run.
    scale_factor: The factor the learning rate is scaled by.

  Returns:
    The learning rate multiplier for this step.

  Raises:
    ValueError: If the schedule name is not known.
  '''
  if name == SCHEDULE_NONE:
    return 1.0
  if name == SCHEDULE_WARMUP_COSINE:
    return warmup_with_cosine_multiplier(index, epoch_length, num_epochs,
                                          scale_factor)
  if name == SCHEDULE_ONE_CYCLE_COSINE:
    return one_cycle_cosine_multiplier(index, num_epochs, epoch_length,
                                        scale_factor)
  raise ValueError(
      f'Unknown schedule {name!r}. Valid names: {sorted(SCHEDULE_NAMES)}')
