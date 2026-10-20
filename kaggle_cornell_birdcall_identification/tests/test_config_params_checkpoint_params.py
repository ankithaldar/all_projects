#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Tests of the checkpoint parameters.

The three keys are the original's ``checkpoint_params``: the directory the
checkpoints are written to, how many best ones are kept, and the filename
prefix.
'''

from pathlib import Path

import pytest
from pydantic import ValidationError

from birdcall.config_params.checkpoint_params import CheckpointParams


def test_the_save_directory_is_required():
  '''A checkpoint configuration without a directory is not a recipe.'''
  with pytest.raises(ValidationError):
    CheckpointParams()


def test_the_defaults_are_the_originals():
  '''Five best checkpoints, named ``checkpoint``.'''
  params = CheckpointParams(save_dir=Path('saved_models'))
  assert params.n_saved == 5
  assert params.prefix_name == 'checkpoint'


def test_the_values_are_kept():
  '''The three keys are stored as they are given.'''
  params = CheckpointParams(save_dir=Path('/tmp/models'), n_saved=10,
                            prefix_name='example_config')
  assert params.save_dir == Path('/tmp/models')
  assert params.n_saved == 10
  assert params.prefix_name == 'example_config'


def test_an_unknown_key_is_refused():
  '''The configuration surface is closed.'''
  with pytest.raises(ValidationError):
    CheckpointParams(save_dir=Path('saved_models'), n_best=3)
