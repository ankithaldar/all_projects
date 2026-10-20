#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Tests of the run parameters.

The two keys are the original's ``run_params``.  ``epoch_length``
truncates an epoch to a fixed iteration count; it is ``None`` in every
shipped config, in which case the epoch length is the length of the
dataloader.
'''

import pytest
from pydantic import ValidationError

from birdcall.config_params.run_params import RunParams


def test_the_defaults_are_the_originals():
  '''Fifty epochs, and no truncation of the epoch length.'''
  params = RunParams()
  assert params.max_epochs == 50
  assert params.epoch_length is None


def test_the_epoch_length_is_kept():
  '''A fixed epoch length is stored as it is given.'''
  assert RunParams(epoch_length=7).epoch_length == 7


def test_an_unknown_key_is_refused():
  '''The configuration surface is closed.'''
  with pytest.raises(ValidationError):
    RunParams(num_epochs=10)
