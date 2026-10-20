#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Which backend to run.

The two backends are interchangeable: same configuration, same data, same
seeds, and they must produce the same submission.  This module says which
of them a run should exercise.

``backend``
    ``pl`` runs the PyTorch Lightning backend, ``tf`` runs the Keras
    backend, and ``both`` runs each of them in turn.  The command line
    ``--backend`` flag overrides this file.

``backends``
    The per-backend settings.  ``entry_point`` is the console script that
    runs that backend.
'''

from pathlib import Path
from typing import Literal

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from birdcall.config_params.loader import RUN_DIR

BackendName = Literal['pl', 'tf', 'both']
BackendSelectionName = Literal['pl', 'tf']


@dataclass(config=ConfigDict(extra='forbid', protected_namespaces=()))
class BackendSettings:
  '''The settings of one backend.

  Attributes:
    enabled: Whether the backend runs.
    entry_point: The console script that runs it.
  '''

  enabled: bool = True
  entry_point: str = ''


@dataclass(config=ConfigDict(extra='forbid', protected_namespaces=()))
class BackendSelection:
  '''Which backend or backends a run exercises.

  Attributes:
    backend: The selection.
    backends: The per-backend settings, keyed by backend name.
  '''

  backend: BackendName = 'both'
  backends: dict[str, BackendSettings] = None  # type: ignore[assignment]

  def __post_init__(self) -> None:
    '''Fills the per-backend defaults when the file omits them.'''
    if self.backends is None:
      self.backends = {
          'pl': BackendSettings(enabled=True,
                                entry_point='birdcall.train_pl'),
          'tf': BackendSettings(enabled=True,
                                entry_point='birdcall.train_tf'),
      }

  def selected(self) -> list[str]:
    '''Returns the backends a run should exercise.

    Returns:
      The backend names, in the order they should run.
    '''
    if self.backend == 'both':
      return ['pl', 'tf']
    return [self.backend]

  def is_enabled(self, name: str) -> bool:
    '''Whether a backend runs.

    Args:
      name: The backend name.

    Returns:
      Whether the backend is selected and enabled.
    '''
    if name not in self.selected():
      return False
    return self.backends.get(name, BackendSettings()).enabled


BACKEND_SELECTION_PATH = RUN_DIR / 'backend.yaml'
