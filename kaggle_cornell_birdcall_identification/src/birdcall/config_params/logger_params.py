#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''The logger parameters.

The six keys are the original's ``logger_params``.  Their names are kept so
that the configuration surface is unchanged, and they map onto wandb as
follows: ``project_name`` is the project, ``name`` is the run name,
``prefix_name`` is the **group**, ``tags`` are the tags, ``params`` is the
run config, and ``log_every`` is the logging cadence.
'''

from typing import Any
from typing import Optional

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass


@dataclass(config=ConfigDict(extra='forbid', protected_namespaces=()))
class LoggerParams:
  '''The tracker parameters of one run.

  Attributes:
    project_name: The tracker project.
    log_every: How many iterations between two logs.
    name: The run name.
    prefix_name: The run group.  The original loggers accepted it and
      never read it; here it is the wandb group, which is what it was
      always meant to be.
    tags: The run tags.
    params: The flat hyper-parameter payload.  It is derived from the
      resolved configuration rather than written by hand, so it cannot
      drift from the run it describes.
  '''

  project_name: str = 'birdcall'
  log_every: int = 10
  name: str = 'run'
  prefix_name: str = 'birdcall'
  tags: list[str | int] = None  # type: ignore[assignment]
  params: Optional[dict[str, Any]] = None

  def __post_init__(self) -> None:
    '''Replaces an omitted tag list with an empty one.'''
    if self.tags is None:
      self.tags = []
