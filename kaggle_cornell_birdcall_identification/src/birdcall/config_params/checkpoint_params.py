#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''The checkpoint parameters.

The three keys are the original's ``checkpoint_params``.  ``n_saved`` is the
number of checkpoints kept, ranked by the tracked metric.
'''

from pathlib import Path

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass


@dataclass(config=ConfigDict(extra='forbid', protected_namespaces=()))
class CheckpointParams:
  '''The checkpointing parameters of one run.

  Attributes:
    save_dir: The directory the checkpoints are written to.
    n_saved: How many best checkpoints are kept.
    prefix_name: The filename prefix of a checkpoint.
  '''

  save_dir: Path
  n_saved: int = 5
  prefix_name: str = 'checkpoint'
