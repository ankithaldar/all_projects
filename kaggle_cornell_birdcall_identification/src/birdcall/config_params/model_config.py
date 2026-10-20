#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''The model hyper-parameters.

The nine keys here are the original's ``model_config`` dict, unchanged.  The
field is called ``model_config`` rather than something else so that the
attribute name still matches the original's configuration surface.
'''

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass


@dataclass(config=ConfigDict(extra='forbid', protected_namespaces=()))
class ModelConfig:
  '''The nine model hyper-parameters of the original configuration.

  Attributes:
    sample_rate: The audio sample rate in Hz.
    window_size: The STFT window size in samples.
    hop_size: The STFT hop size in samples.
    mel_bins: The number of mel bands.
    fmin: The lowest frequency of the mel bank in Hz.
    fmax: The highest frequency of the mel bank in Hz.
    classes_num: The number of output classes.
    apply_aug: Whether the in-graph SpecAugment runs.  Note that disabling
      it while mixup is on used to halve the batch against unmixed data.
    top_db: The dB ceiling of the log-mel.  ``None`` in every shipped
      config, which makes the clamp dead code.
  '''

  sample_rate: int = 32000
  window_size: int = 1024
  hop_size: int = 320
  mel_bins: int = 64
  fmin: int = 50
  fmax: int = 14000
  classes_num: int = 264
  apply_aug: bool = True
  top_db: float | None = None
