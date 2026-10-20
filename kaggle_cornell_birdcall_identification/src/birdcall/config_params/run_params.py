#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''The run parameters.

The two keys are the original's ``run_params``.  ``epoch_length`` truncates
an epoch to a fixed iteration count; it is ``None`` in every shipped
config, in which case the epoch length is the length of the dataloader.
'''

from typing import Optional

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass


@dataclass(config=ConfigDict(extra='forbid', protected_namespaces=()))
class RunParams:
  '''The training loop parameters of one run.

  Attributes:
    max_epochs: The number of epochs to train for.
    epoch_length: The number of iterations in an epoch, or ``None`` to use
      the length of the dataloader.
  '''

  max_epochs: int = 50
  epoch_length: Optional[int] = None
