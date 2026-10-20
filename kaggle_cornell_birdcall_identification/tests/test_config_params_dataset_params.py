#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Tests of the dataset parameters.

The label space is not written in the YAML: it is injected from the
shared label space module, because the 264 species codes belong in
exactly one place.
'''

from pathlib import Path

import pytest
from pydantic import ValidationError

from birdcall.config_params.dataset_params import DatasetParams


def _params(**overrides) -> DatasetParams:
  '''Builds a dataset parameter group with the required keys filled in.'''
  values = {
      'root_dir': Path('/data'),
      'csv_dir': Path('/folds/fold_0_train.csv'),
      'period': 30,
      'isTraining': True,
      'num_test_samples': 1,
  }
  values.update(overrides)
  return DatasetParams(**values)


def test_the_label_space_is_injected_when_it_is_omitted():
  '''The YAML does not carry the 264 codes.'''
  params = _params()
  assert len(params.bird_code) == 264
  assert len(params.inv_ebird_label) == 264


def test_the_injected_label_space_is_the_shared_one():
  '''The injected map is the committed label space.'''
  from birdcall.config_params.configs import get_bird_code
  assert _params().bird_code == get_bird_code()


def test_an_explicit_label_space_is_kept():
  '''A label space given in the YAML is not overwritten.'''
  params = _params(bird_code={'AMEBR': 0}, inv_ebird_label={'A': 'AMEBR'})
  assert params.bird_code == {'AMEBR': 0}
  assert params.inv_ebird_label == {'A': 'AMEBR'}


def test_the_background_directory_is_optional():
  '''Only the validation and the test splits set it.'''
  assert _params().background_audio_dir is None
  params = _params(background_audio_dir=Path('/background/data_ssw'))
  assert params.background_audio_dir == Path('/background/data_ssw')


def test_the_data_paths_are_required():
  '''A dataset without its paths is not a recipe.'''
  with pytest.raises(ValidationError):
    DatasetParams(period=30, isTraining=True, num_test_samples=1)


def test_an_unknown_key_is_refused():
  '''The configuration surface is closed.'''
  with pytest.raises(ValidationError):
    _params(num_workers=8)
