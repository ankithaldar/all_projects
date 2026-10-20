#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Tests of the model hyper-parameters.

The nine keys are the original's ``model_config`` dict, unchanged.  The
field is called ``model_config`` rather than something else so that the
attribute name still matches the original's configuration surface.
'''

import pytest
from pydantic import ValidationError

from birdcall.config_params.model_config import ModelConfig


def test_the_defaults_are_the_shipped_ones():
  '''The default is the configuration of the shipped experiments.'''
  config = ModelConfig()
  assert config.sample_rate == 32000
  assert config.window_size == 1024
  assert config.hop_size == 320
  assert config.mel_bins == 64
  assert config.fmin == 50
  assert config.fmax == 14000
  assert config.classes_num == 264
  assert config.apply_aug is True
  assert config.top_db is None


def test_the_top_db_ceiling_is_off_by_default():
  '''`None` makes the clamp dead code, as in every shipped config.'''
  assert ModelConfig().top_db is None
  assert ModelConfig(top_db=80.0).top_db == 80.0


def test_the_augmentation_switch_is_on_by_default():
  '''The in-graph SpecAugment runs unless it is turned off.'''
  assert ModelConfig().apply_aug is True
  assert ModelConfig(apply_aug=False).apply_aug is False


def test_an_unknown_key_is_refused():
  '''The configuration surface is closed.'''
  with pytest.raises(ValidationError):
    ModelConfig(n_mels=128)
