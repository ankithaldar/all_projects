#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''The dataset parameters.

These are the original's ``train_ds_params``, ``valid_ds_params`` and
``test_ds_params`` dicts.  The label space is not written in the YAML: it is
injected from the shared label space module, because the 264 species codes
belong in exactly one place.
'''

from pathlib import Path
from typing import Optional

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from birdcall.config_params.configs import get_bird_code
from birdcall.config_params.configs import get_inv_ebird_label


@dataclass(config=ConfigDict(extra='forbid', protected_namespaces=()))
class DatasetParams:
  '''The parameters of one dataset split.

  Attributes:
    root_dir: The root of the audio tree.
    csv_dir: The fold CSV listing the rows of this split.
    period: The snippet length in seconds.
    bird_code: The species code to class index map.
    inv_ebird_label: The class index to label map.
    isTraining: Whether the split is the training one.  It selects the
      random slicing mode.
    num_test_samples: The number of snippets per file.  One for training,
      two for validation in every shipped config.
    background_audio_dir: A directory of label-less audio appended as hard
      negatives.  Only the validation and test splits set it.
  '''

  root_dir: Path
  csv_dir: Path
  period: int
  isTraining: bool
  num_test_samples: int
  background_audio_dir: Optional[Path] = None
  bird_code: dict[str, int] = None  # type: ignore[assignment]
  inv_ebird_label: dict[str, str] = None  # type: ignore[assignment]

  def __post_init__(self) -> None:
    '''Fills the label space when the YAML did not supply it.'''
    if self.bird_code is None:
      self.bird_code = get_bird_code()
    if self.inv_ebird_label is None:
      self.inv_ebird_label = get_inv_ebird_label()
