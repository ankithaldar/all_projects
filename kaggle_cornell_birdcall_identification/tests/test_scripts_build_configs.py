#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Tests of the configuration generator.

The original kept 21 experiment configurations as Python modules, each
one repeating all 39 attributes.  This script inverts the layout: it
works out which values are shared, writes those into one YAML file per
functional module, and writes one YAML file per experiment holding only
what that experiment changes.

The original sources are not in this repository, so the tests build a
tiny synthetic source tree instead.
'''

import textwrap

import pytest

from birdcall.scripts.build_configs import DATASET_KEYS
from birdcall.scripts.build_configs import LOGGER_NAMES
from birdcall.scripts.build_configs import MODULE_FILES
from birdcall.scripts.build_configs import PROJECT_DEFAULTS
from birdcall.scripts.build_configs import build_module_defaults
from birdcall.scripts.build_configs import common_value
from birdcall.scripts.build_configs import deep_equal
from birdcall.scripts.build_configs import diff_against
from birdcall.scripts.build_configs import load_original_module
from birdcall.scripts.build_configs import normalise
from birdcall.scripts.build_configs import validate_criteria


class _Parameters:
  '''A stand-in for the original's configuration class.

  Attributes:
    name: The experiment name.
    fold: The fold number.
    criterion_name: The criterion the experiment selects.
    criterion_params: The criterion hyper-parameters.
  '''

  def __init__(self, name: str, fold: int, criterion_name: str,
               criterion_params: dict) -> None:
    self.name = name
    self.fold = fold
    self.criterion_name = criterion_name
    self.criterion_params = criterion_params
    for keys in MODULE_FILES.values():
      for key in keys:
        if not hasattr(self, key):
          setattr(self, key, _default_for(key))
    self.logger_params = {'project_name': 'p', 'name': name,
                          'prefix_name': 'g', 'tags': [0], 'params': {}}
    self.checkpoint_params = {'save_dir': 'saved_models', 'n_saved': 5,
                              'prefix_name': 'checkpoint'}


def _default_for(key: str):
  '''Builds a plausible value for one configuration key.

  Args:
    key: The name of the key.

  Returns:
    A value the generator can work with.
  '''
  if key in DATASET_KEYS:
    return {'root_dir': 'data', 'csv_dir': 'fold.csv', 'period': 30,
            'isTraining': True, 'num_test_samples': 1,
            'bird_code': {}, 'inv_ebird_label': {}}
  if key == 'logger_params':
    return {'project_name': 'p'}
  if key == 'checkpoint_params':
    return {'save_dir': 'saved_models', 'n_saved': 5,
            'prefix_name': 'checkpoint'}
  if key == 'model_config':
    return {'sample_rate': 32000}
  if key == 'run_params':
    return {'max_epochs': 50, 'epoch_length': None}
  if key == 'dist_params':
    return {}
  if key == 'criterion_params':
    return {'gamma': 0.0}
  if key == 'metrics':
    return ['lraps']
  if key == 'logger_name':
    return 'neptune'
  if key == 'model_name':
    return 'sed_dense121att'
  if key == 'criterion_name':
    return 'focal_loss_standard'
  if key == 'aug_name':
    return 'default'
  if key == 'optimizer_name':
    return 'adamw'
  if key == 'scheduler_name':
    return 'warmup_with_cosine'
  if key == 'pretrained_path':
    return None
  if key == 'bckgrd_aug_dir':
    return None
  if key == 'secondary_bckgrd_aug_dir':
    return None
  if key == 'checkpoint_dir':
    return None
  if key == 'val_length':
    return None
  return 1


def _source_tree(tmp_path):
  '''Builds a synthetic tree of original configuration modules.

  Args:
    tmp_path: The pytest temporary directory.

  Returns:
    The root of the synthetic source tree.
  '''
  source = tmp_path / 'src'
  config_root = source / 'config_params'
  (config_root / 'final_sed').mkdir(parents=True)
  (config_root / 'example_config.py').write_text(textwrap.dedent('''
      class Parameters:
        def __init__(self):
          self.name = 'example_config'
          self.fold = 0
          self.criterion_name = 'focal_loss_standard'
          self.criterion_params = {'gamma': 0.0}
      '''), encoding='utf-8')
  (config_root / 'final_sed' / 'one.py').write_text(textwrap.dedent('''
      class Parameters:
        def __init__(self):
          self.name = 'one'
          self.fold = 1
          self.criterion_name = 'focal_loss_standard'
          self.criterion_params = {'gamma': 0.0}
      '''), encoding='utf-8')
  return source


def test_the_module_files_cover_every_configuration_key():
  '''The ten functional modules own the whole configuration surface.'''
  assert set(MODULE_FILES) == {
      'model', 'augmentation', 'loss', 'optimizer', 'scheduler', 'data',
      'metrics', 'logging', 'checkpoint', 'runtime',
  }
  assert len(MODULE_FILES) == 10


def test_the_project_defaults_are_seeded():
  '''The attributes the original had no notion of are seeded here.'''
  assert PROJECT_DEFAULTS['seed'] == 42
  assert PROJECT_DEFAULTS['mixup_seed'] == 1234
  assert PROJECT_DEFAULTS['deterministic'] is True


def test_the_neptune_tracker_becomes_wandb():
  '''The original tracked to Neptune; this implementation tracks to wandb.'''
  assert LOGGER_NAMES['neptune'] == 'wandb'


def test_a_path_becomes_a_string():
  '''YAML holds strings, not paths.'''
  from pathlib import Path
  assert normalise(Path('data')) == 'data'


def test_the_normalisation_recurses():
  '''The nested structures are normalised too.'''
  from pathlib import Path
  value = {'a': Path('data'), 'b': (1, 2), 'c': [Path('x')]}
  assert normalise(value) == {'a': 'data', 'b': [1, 2], 'c': ['x']}


def test_two_structures_are_equal_when_they_normalise_the_same():
  '''A path and its string are the same value.'''
  from pathlib import Path
  assert deep_equal({'a': Path('data')}, {'a': 'data'})
  assert not deep_equal({'a': 1}, {'a': 2})


def test_the_most_common_value_wins():
  '''The shared value is the one most configurations agree on.'''
  assert common_value([1, 1, 2]) == 1
  assert common_value([1, 2]) == 1


def test_the_tie_is_broken_by_first_appearance():
  '''The result is deterministic.'''
  assert common_value(['a', 'b', 'b', 'a']) == 'a'


def test_the_difference_against_a_base_is_the_changed_part():
  '''Only the differing parts are kept.'''
  assert diff_against({'a': 1, 'b': 2}, {'a': 1, 'b': 3}) == {'b': 3}
  assert diff_against({'a': 1}, {'a': 1}) is None


def test_the_difference_of_a_new_key_is_the_key_itself():
  '''A key the base does not have is a difference.'''
  assert diff_against({'a': 1}, {'a': 1, 'b': 2}) == {'b': 2}


def test_an_original_module_is_imported_by_path(tmp_path):
  '''The original sources are imported from their path.'''
  source = _source_tree(tmp_path)
  module = load_original_module(source, 'config_params/example_config.py')
  assert module.Parameters().name == 'example_config'


def test_the_module_defaults_are_built():
  '''The shared values become the module files.'''
  parameters = {'a': _Parameters('a', 0, 'focal_loss_standard',
                                 {'gamma': 0.0}),
                'b': _Parameters('b', 1, 'focal_loss_standard',
                                 {'gamma': 0.0})}
  defaults = build_module_defaults(parameters)
  assert set(defaults) == set(MODULE_FILES)
  assert defaults['loss']['criterion_name'] == 'focal_loss_standard'
  assert defaults['model']['model_name'] == 'sed_dense121att'
  # The tracker name is translated.
  assert defaults['logging']['logger_name'] == 'wandb'
  # The seeded attributes are part of the recipe.
  assert defaults['runtime']['seed'] == 42


def test_the_criteria_are_validated():
  '''A configuration whose parameters do not fit its criterion is refused.'''
  parameters = {'a': _Parameters('a', 0, 'focal_loss_standard',
                                 {'gamma': 0.0, 'alpha': 1.0})}
  validate_criteria(parameters)
  bad = {'a': _Parameters('a', 0, 'focal_loss_standard',
                          {'gamma': 0.0, 'beta': 0.5})}
  with pytest.raises(ValueError):
    validate_criteria(bad)
