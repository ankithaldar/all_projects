#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Tests of the configuration loader.

A configuration is composed from three layers: the module files, the
experiment file and the backend selection.  The loader merges the first
two and validates the result against the pydantic schema, and the error
report names the file the bad value came from.
'''

import pytest
import yaml
from pydantic import ValidationError

from birdcall.config_params.loader import CONFIG_DIR
from birdcall.config_params.loader import MODULES_DIR
from birdcall.config_params.loader import backend_selection_path
from birdcall.config_params.loader import deep_merge
from birdcall.config_params.loader import iter_experiment_paths
from birdcall.config_params.loader import load_backend_selection
from birdcall.config_params.loader import load_parameters
from birdcall.config_params.loader import module_names
from birdcall.config_params.loader import read_config_document
from birdcall.config_params.loader import read_module
from birdcall.config_params.loader import read_yaml


def test_the_deep_merge_lets_the_override_win():
  '''The higher layer wins, key by key.'''
  merged = deep_merge({'a': 1, 'b': 2}, {'b': 3, 'c': 4})
  assert merged == {'a': 1, 'b': 3, 'c': 4}


def test_the_deep_merge_recurses_into_the_mappings():
  '''A nested mapping is merged, not replaced.'''
  merged = deep_merge({'a': {'x': 1, 'y': 2}}, {'a': {'y': 3}})
  assert merged == {'a': {'x': 1, 'y': 3}}


def test_the_deep_merge_does_not_touch_the_inputs():
  '''The merge returns a new dictionary.'''
  base = {'a': {'x': 1}}
  merged = deep_merge(base, {'a': {'x': 2}})
  assert base == {'a': {'x': 1}}
  assert merged == {'a': {'x': 2}}


def test_a_missing_file_is_a_clear_error():
  '''A path that does not exist names itself.'''
  with pytest.raises(FileNotFoundError, match='not found'):
    read_yaml('/tmp/definitely-not-a-config.yaml')


def test_an_empty_document_is_an_empty_mapping():
  '''An empty YAML file is an empty document.'''
  import tempfile
  from pathlib import Path
  with tempfile.TemporaryDirectory() as directory:
    path = Path(directory) / 'empty.yaml'
    path.write_text('', encoding='utf-8')
    assert read_yaml(path) == {}


def test_a_document_that_is_not_a_mapping_is_refused():
  '''A YAML list is not a configuration.'''
  import tempfile
  from pathlib import Path
  with tempfile.TemporaryDirectory() as directory:
    path = Path(directory) / 'list.yaml'
    path.write_text('- a\n- b\n', encoding='utf-8')
    with pytest.raises(ValueError, match='not a mapping'):
      read_yaml(path)


def test_the_module_files_are_listed():
  '''The functional modules are the ten shipped ones.'''
  names = module_names()
  assert names == sorted(names)
  assert 'model' in names
  assert 'loss' in names
  assert 'runtime' in names


def test_an_unknown_module_is_a_clear_error():
  '''A module name that does not exist names the available ones.'''
  with pytest.raises(FileNotFoundError, match='not_a_module'):
    read_module('not_a_module')


def test_the_experiment_files_are_listed():
  '''Every shipped experiment file is found.'''
  paths = iter_experiment_paths()
  assert paths
  assert all(path.suffix == '.yaml' for path in paths)
  assert paths == sorted(paths)


def test_an_experiment_document_is_read_without_validation():
  '''The raw document is returned as it is.'''
  path = CONFIG_DIR / 'experiments' / 'example_config' / 'example_config.yaml'
  document = read_config_document(path)
  assert document['name'] == 'example_config'
  assert 'overrides' in document
  assert 'modules' in document


def test_the_example_experiment_loads_and_validates():
  '''The shipped example configuration is a valid recipe.'''
  path = CONFIG_DIR / 'experiments' / 'example_config' / 'example_config.yaml'
  parameters = load_parameters(path)
  assert parameters.name == 'example_config'
  assert parameters.model_name == 'sed_dense121att'
  assert parameters.criterion_name == 'sed_scaled_pos_neg_focal_loss'
  assert parameters.train_ds_params.period == 30
  assert parameters.checkpoint_params.n_saved == 10


def test_every_shipped_experiment_loads():
  '''The twenty-one shipped configurations all validate.'''
  paths = iter_experiment_paths()
  assert len(paths) >= 21
  for path in paths:
    parameters = load_parameters(path)
    assert parameters.name


def test_an_unknown_key_is_refused_and_the_file_is_named():
  '''The error report says which file the bad value came from.'''
  import tempfile
  from pathlib import Path
  with tempfile.TemporaryDirectory() as directory:
    path = Path(directory) / 'broken.yaml'
    path.write_text(yaml.safe_dump({
        'modules': ['model'],
        'not_a_key': 1,
    }), encoding='utf-8')
    with pytest.raises(ValidationError) as error:
      load_parameters(path)
    assert str(path) in str(error.value)


def test_the_backend_selection_file_is_the_packaged_one():
  '''The backend selection lives in the run directory.'''
  assert backend_selection_path().name == 'backend.yaml'
  document = load_backend_selection()
  assert document['backend'] in ('pl', 'tf', 'both')
  assert set(document['backends']) == {'pl', 'tf'}


def test_the_module_directory_holds_the_module_files():
  '''The module files are the ones the loader merges.'''
  assert MODULES_DIR.is_dir()
  assert len(list(MODULES_DIR.glob('*.yaml'))) == len(module_names())
