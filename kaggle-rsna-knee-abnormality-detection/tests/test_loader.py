#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for YAML loading, overrides, and recursive instantiation."""

import os
import sys

import pytest

from knee.config_params.loader import (
  deep_merge,
  instantiate,
  load_config,
  load_experiment,
)

CONFIG_DIR = os.path.join(
  os.path.dirname(__file__), '..', 'configs', 'experiments'
)


class TestLoadConfig:
  """load_config behavior on single files."""

  def test_loads_yaml_and_resolves_interpolation(self, tmp_path):
    path = tmp_path / 'c.yaml'
    path.write_text('a:\n  b: ${x}\nx: 5\n')
    config = load_config(str(path))
    assert config['a']['b'] == 5

  def test_resolves_composite_interpolation(self, tmp_path):
    # Regression: suffix after the reference must expand too
    # (failed on Kaggle as literal '${paths.competition_dir}/train_series').
    path = tmp_path / 'c.yaml'
    path.write_text(
      'root: /data\ndicom_dir: ${root}/train_series\nnested: ${dicom_dir}/sub\n'
    )
    config = load_config(str(path))
    assert config['dicom_dir'] == '/data/train_series'
    assert config['nested'] == '/data/train_series/sub'

  def test_missing_interpolation_reference_raises_keyerror(self, tmp_path):
    path = tmp_path / 'c.yaml'
    path.write_text('a: ${nope.here}\n')
    with pytest.raises(KeyError):
      load_config(str(path))

  def test_interpolation_cycle_raises_valueerror(self, tmp_path):
    path = tmp_path / 'c.yaml'
    path.write_text('a: ${b}\nb: ${a}\n')
    with pytest.raises(ValueError):
      load_config(str(path))

  def test_dot_path_override_with_scalar_parsing(self, tmp_path):
    path = tmp_path / 'c.yaml'
    path.write_text('model:\n  init_params:\n    dropout: 0.1\n')
    config = load_config(str(path), overrides=['model.init_params.dropout=0.3'])
    assert config['model']['init_params']['dropout'] == 0.3

  def test_bad_override_path_raises_keyerror(self, tmp_path):
    path = tmp_path / 'c.yaml'
    path.write_text('a: 1\n')
    with pytest.raises(KeyError):
      load_config(str(path), overrides=['missing.path=1'])


class TestDeepMerge:
  """deep_merge recursion and immutability."""

  def test_nested_leaf_wins(self):
    base = {'a': {'b': 1, 'c': 2}}
    patch = {'a': {'b': 9}}
    merged = deep_merge(base, patch)
    assert merged == {'a': {'b': 9, 'c': 2}}
    assert base['a']['b'] == 1  # inputs untouched

  def test_new_keys_added(self):
    assert deep_merge({'x': 1}, {'y': [1, 2]})['y'] == [1, 2]


class TestInstantiate:
  """Recursive class_path resolution."""

  def test_instantiates_nested_specs(self, tmp_path):
    module = tmp_path / 'fake_mod.py'
    module.write_text(
      'class Inner:\n'
      '    def __init__(self, v):\n'
      '        self.v = v\n\n\n'
      'class Outer:\n'
      '    def __init__(self, inner=None, k=0):\n'
      '        self.inner = inner\n'
      '        self.k = k\n'
    )
    sys.path.insert(0, str(tmp_path))
    spec = {
      'class_path': 'fake_mod.Outer',
      'init_params': {
        'inner': {'class_path': 'fake_mod.Inner', 'init_params': {'v': 3}},
        'k': 7,
      },
    }
    obj = instantiate(spec)
    assert obj.inner.v == 3 and obj.k == 7

  def test_passthrough_plain_values(self):
    assert instantiate([1, {'a': 2}]) == [1, {'a': 2}]


class TestExperiment:
  """Experiment composition from defaults + override."""

  def test_all_experiment_files_compose(self):
    for name in os.listdir(CONFIG_DIR):
      if name.endswith('.yaml'):
        config = load_experiment(os.path.join(CONFIG_DIR, name))
        for section in [
          'experiment',
          'data',
          'folds',
          'model',
          'loss',
          'optimizer',
          'train',
          'datamodule',
          'infer',
        ]:
          assert section in config, f'{name} missing section {section}'

  def test_override_section_wins(self):
    config = load_experiment(os.path.join(CONFIG_DIR, 'smoke_ci.yaml'))
    assert config['trainer']['init_params']['max_epochs'] == 1
    assert config['experiment']['name'] == 'smoke_ci'

  def test_cli_overrides_apply_after_composition(self):
    config = load_experiment(
      os.path.join(CONFIG_DIR, 'mvp_efnv2s_384_k24_5f.yaml'),
      overrides=['data.n_slices=12'],
    )
    assert config['data']['n_slices'] == 12
