#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''The AdamW optimiser and the learning rate schedule of the torch backend.

The optimiser is the original's, argument for argument: ``AdamW`` with
``betas=(0.9, 0.999)``, ``eps=1e-08``, the configured weight decay and
``amsgrad=True``.

The schedule is a ``LambdaLR`` whose lambda is
:func:`birdcall.engines.schedules.schedule_multiplier`, the transcription of
the original's Ignite schedulers.  It is driven with ``interval='step'``,
so the learning rate of batch ``t`` is the multiplier of step ``t``, which
is the index Ignite used for iteration ``t``.

The schedule name ``none`` yields a constant multiplier of one, which is
the Lightning spelling of "no scheduler".
'''

import torch

from birdcall.config_params.parameters import Parameters
from birdcall.engines.schedules import schedule_multiplier

OPTIMIZER_ADAMW = 'adamw'
'''The only optimiser the configuration can select.'''


def build_optimizer(parameters: Parameters,
                    model: torch.nn.Module) -> torch.optim.AdamW:
  '''Builds the optimiser of a run.

  Args:
    parameters: The validated configuration.
    model: The model whose parameters are optimised.

  Returns:
    The AdamW optimiser, with ``amsgrad`` on.

  Raises:
    ValueError: If the configuration selects an unknown optimiser.
  '''
  if parameters.optimizer_name != OPTIMIZER_ADAMW:
    raise ValueError(
        f'Unknown optimizer {parameters.optimizer_name!r}. '
        f'Valid names: {[OPTIMIZER_ADAMW]}.')
  return torch.optim.AdamW(model.parameters(),
                            lr=parameters.lr,
                            betas=(0.9, 0.999),
                            eps=1e-08,
                            weight_decay=parameters.weight_decay,
                            amsgrad=True)


def build_scheduler(parameters: Parameters,
                    optimizer: torch.optim.Optimizer,
                    epoch_length: int) -> torch.optim.lr_scheduler.LambdaLR:
  '''Builds the learning rate schedule of a run.

  Args:
    parameters: The validated configuration.
    optimizer: The optimiser the schedule drives.
    epoch_length: The number of iterations in one epoch.  It is the
      configured ``run_params['epoch_length']`` or the length of the
      training dataloader.

  Returns:
    A ``LambdaLR`` whose multiplier is the schedule multiplier of the
    step.  The schedule is stepped with ``interval='step'``.
  '''
  name = parameters.scheduler_name
  num_epochs = parameters.run_params.max_epochs
  scale_factor = parameters.lr_scale_factor
  length = max(1, int(epoch_length))

  def multiplier(step: int) -> float:
    '''Returns the learning rate multiplier of one step.

    Args:
      step: The global step index.

    Returns:
      The multiplier of the step, relative to the base learning rate.
    '''
    return schedule_multiplier(name, step, length, num_epochs, scale_factor)

  return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=multiplier)
