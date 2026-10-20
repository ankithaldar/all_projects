#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''The wandb logger of the torch backend.

The original tracked its runs with Neptune, through ignite's Neptune
logger, and the dashboards of the 13 model ensemble were built on the
series names that logger wrote.  This port tracks with **wandb** and
keeps the series names, so a dashboard built on the original's names
still reads the right numbers:

============================  ==================================
original                      wandb series
============================  ==================================
``train/loss``                ``train_loss``
``train/<loss_key>``          ``train_<loss_key>``
``train/lr``                  ``lr-AdamW``
``train/<metric>``            ``train_<metric>``
``valid/<metric>``            ``valid_<metric>``
============================  ==================================

The ``logger_params`` keys keep their names and map onto wandb:
``project_name`` is the project, ``name`` is the run name, ``prefix_name``
is the **group**, ``tags`` are the tags, ``params`` is the run config and
``log_every`` is the cadence of the per-iteration series.

A wandb failure is never fatal.  The tracker is a convenience, not a
dependency of the experiment, so every call into wandb is caught and
downgraded to a warning.
'''

import logging

from lightning.pytorch.loggers import Logger

from birdcall.config_params.parameters import Parameters

LOGGER = logging.getLogger(__name__)

TRAIN_PREFIX = 'train_'
VALID_PREFIX = 'valid_'
LR_PREFIX = 'lr-'


class WandbLoggerAdapter(Logger):
  '''A Lightning logger that writes the original's series to wandb.

  Attributes:
    config: The validated configuration of the run.
    log_every: The cadence of the per-iteration series.
    failure: The exception of the first wandb failure, if any.
  '''

  def __init__(self, parameters: Parameters) -> None:
    '''Builds the logger.

    The wandb run itself is created lazily, so a broken wandb install
    cannot break the import of this module.

    Args:
      parameters: The validated configuration.
    '''
    super().__init__()
    self.config = parameters
    self.log_every = parameters.logger_params.log_every
    self.failure: Exception | None = None
    self._inner = None
    self._tried = False

  @property
  def inner(self):
    '''Returns the wandb logger, or ``None`` when wandb is unusable.

    Returns:
      The ``WandbLogger`` of the run.
    '''
    if self._inner is None and not self._tried:
      self._tried = True
      try:
        from lightning.pytorch.loggers import WandbLogger
        params = self.config.logger_params
        self._inner = WandbLogger(
            project=params.project_name,
            name=params.name,
            group=params.prefix_name,
            tags=list(params.tags),
            config=params.params,
            save_dir=str(self.config.checkpoint_params.save_dir))
      except Exception as error:  # pylint: disable=broad-except
        self.failure = error
        LOGGER.warning('wandb is unavailable (%s); the run continues '
                       'without a tracker.', error)
    return self._inner

  @staticmethod
  def series_name(name: str) -> str:
    '''Returns the wandb series name of a Lightning metric.

    Args:
      name: The name Lightning logged the metric under.

    Returns:
      The name the original's tracker used.
    '''
    if name.startswith(TRAIN_PREFIX):
      return f'train/{name[len(TRAIN_PREFIX):]}'
    if name.startswith(VALID_PREFIX):
      return f'valid/{name[len(VALID_PREFIX):]}'
    if name.startswith(LR_PREFIX):
      return 'train/lr'
    return name

  def log_metrics(self, metrics: dict, step=None) -> None:
    '''Writes the metrics under the original's series names.

    Args:
      metrics: The metrics Lightning logged.
      step: The step the metrics belong to.
    '''
    inner = self.inner
    if inner is None:
      return
    renamed = {self.series_name(name): value
               for name, value in metrics.items()}
    try:
      inner.log_metrics(renamed, step=step)
    except Exception as error:  # pylint: disable=broad-except
      LOGGER.warning('Could not write the metrics to wandb: %s', error)

  def log_hyperparams(self, params, *args, **kwargs) -> None:
    '''Writes the hyper-parameters of the run.

    Args:
      params: The hyper-parameters.
      *args: Ignored.
      **kwargs: Ignored.
    '''
    inner = self.inner
    if inner is None:
      return
    try:
      inner.log_hyperparams(params)
    except Exception as error:  # pylint: disable=broad-except
      LOGGER.warning('Could not write the hyper-parameters to wandb: %s',
                     error)

  def save(self) -> None:
    '''Flushes the run.'''
    inner = self.inner
    if inner is None:
      return
    try:
      inner.save()
    except Exception as error:  # pylint: disable=broad-except
      LOGGER.warning('Could not flush the wandb run: %s', error)

  def finalize(self, status: str) -> None:
    '''Closes the run.

    Args:
      status: The status the run finished with.
    '''
    inner = self.inner
    if inner is None:
      return
    try:
      inner.finalize(status)
    except Exception as error:  # pylint: disable=broad-except
      LOGGER.warning('Could not close the wandb run: %s', error)

  @property
  def name(self) -> str:
    '''Returns the name of the run.'''
    inner = self.inner
    if inner is None:
      return self.config.logger_params.name
    return inner.name or self.config.logger_params.name

  @property
  def version(self):
    '''Returns the version of the run.'''
    inner = self.inner
    if inner is None:
      return None
    return inner.version

  @property
  def save_dir(self):
    '''Returns the directory the run writes to.'''
    inner = self.inner
    if inner is None:
      return None
    return inner.save_dir

  @property
  def experiment(self):
    '''Returns the wandb run.'''
    inner = self.inner
    if inner is None:
      return None
    return inner.experiment
