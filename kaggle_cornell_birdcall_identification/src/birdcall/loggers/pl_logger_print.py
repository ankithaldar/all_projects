#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''The console logger of the torch backend.

The original's print logger wrote the same series as its Neptune logger
to stdout, every ``log_every`` iterations.  This adapter writes the same
series names, so a console transcript of the original and of the port
can be read side by side.

The output goes through the ``logging`` module with a handler on stdout,
rather than through ``print``, so a run can redirect or silence it.

The name mapping is the one of the wandb adapter, so the two loggers
cannot drift apart: the wandb module imports no wandb at module level,
so borrowing its mapping costs nothing.
'''

import logging
import sys

from lightning.pytorch.loggers import Logger

from birdcall.loggers.pl_logger_wandb import WandbLoggerAdapter

LOGGER_NAME = 'birdcall.print_logger'


class StdoutHandler(logging.StreamHandler):
  '''A handler that writes to the **current** stdout.

  A stream handler that holds on to the stream it was built with keeps
  writing to a stdout that has been replaced since, which is what a
  test runner does.  This one looks the stream up every time.
  '''

  def __init__(self) -> None:
    '''Builds the handler.'''
    super().__init__(stream=None)

  @property
  def stream(self):
    '''Returns the current stdout.

    Returns:
      The stream of the process.
    '''
    return sys.stdout

  @stream.setter
  def stream(self, value) -> None:
    '''Ignores the stream the handler is built with.

    Args:
      value: The stream that was given.
    '''


def stdout_logger() -> logging.Logger:
  '''Returns the logger that writes to stdout.

  Returns:
    A logger with one handler on stdout, which never propagates to the
    root logger, so a series is not written twice.
  '''
  logger = logging.getLogger(LOGGER_NAME)
  if not logger.handlers:
    handler = StdoutHandler()
    handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
  return logger


class PrintLoggerAdapter(Logger):
  '''A Lightning logger that writes the original's series to stdout.

  Attributes:
    config: The validated configuration of the run.
    log_every: The cadence of the per-iteration series.
    lines: Every line that was written.
  '''

  def __init__(self, parameters) -> None:
    '''Builds the logger.

    Args:
      parameters: The validated configuration.
    '''
    super().__init__()
    self.config = parameters
    self.log_every = parameters.logger_params.log_every
    self.lines: list[str] = []
    self._logger = stdout_logger()

  def log_metrics(self, metrics: dict, step=None) -> None:
    '''Writes the metrics under the original's series names.

    Args:
      metrics: The metrics Lightning logged.
      step: The step the metrics belong to.
    '''
    for name, value in metrics.items():
      series = WandbLoggerAdapter.series_name(name)
      line = f'{series} {step} {value}'
      self.lines.append(line)
      self._logger.info(line)

  def log_hyperparams(self, params, *args, **kwargs) -> None:
    '''Writes the hyper-parameters of the run.

    Args:
      params: The hyper-parameters.
      *args: Ignored.
      **kwargs: Ignored.
    '''
    for name, value in dict(params).items():
      line = f'config/{name} {value}'
      self.lines.append(line)
      self._logger.info(line)

  @property
  def name(self) -> str:
    '''Returns the name of the run.'''
    return self.config.logger_params.name

  @property
  def version(self):
    '''Returns the version of the run.'''
    return None

  @property
  def save_dir(self):
    '''Returns the directory the run writes to.'''
    return None

  @property
  def experiment(self):
    '''Returns the experiment, which the console logger does not have.'''
    return None
