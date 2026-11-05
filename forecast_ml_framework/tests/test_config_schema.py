#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Tests for pydantic config schema and OmegaConf loading.'''

from pathlib import Path

import pytest

pytest.importorskip('pydantic')
pytest.importorskip('omegaconf')

from omegaconf import OmegaConf  # noqa: E402

from forecast_ml_framework.core.config_loader import (  # noqa: E402
  ConfigLoader,
)
from forecast_ml_framework.core.config_schema import (  # noqa: E402
  ParametersConfig,
  VariablesConfig,
  load_config,
  validate_parameters,
  validate_variables,
)


def _root():
  '''Return project root.'''
  return Path(__file__).resolve().parents[1]


def test_variables_defaults_and_types():
  '''VariablesConfig defaults and shipped values parse.'''
  cfg = VariablesConfig()
  assert cfg.target_col == 'target'
  assert cfg.cat_cols == []
  assert cfg.business_relevant_features == []
  shipped = validate_variables({
    'target_col': 'target', 'id_cols': ['id'],
    'cat_cols': ['region'], 'num_cols': ['age'],
    'date_cols': [], 'seq_cols': ['seq'], 'ind_cols': [],
    'drop_cols': [], 'bucket_segment_col': 'region',
    'business_relevant_features': [], 'override_features': [],
    'exclude_features': []})
  assert shipped.cat_cols == ['region']
  assert shipped.target_col == 'target'


def test_variables_rejects_wrong_type():
  '''VariablesConfig rejects non-string target_col.'''
  with pytest.raises(Exception):
    VariablesConfig(target_col=123)


def test_parameters_shipped_conf_passes():
  '''ParametersConfig accepts shipped base parameters (parameters/).'''
  import yaml
  raw = {}
  for path in sorted((_root() / 'conf' / 'base' / 'parameters').glob('*.yml')):
    with path.open() as handle:
      raw.update(yaml.safe_load(handle))
  cfg = validate_parameters(raw)
  assert isinstance(cfg, ParametersConfig)
  assert 'modeling_params' in cfg.model_dump()
  assert 'feature_selection_params' in cfg.model_dump()
  assert 'validation_params' in cfg.model_dump()


def test_parameters_extra_allow():
  '''Unknown top-level keys are allowed (extra=allow).'''
  cfg = ParametersConfig(unknown_section={'x': 1})
  assert cfg.model_extra == {'unknown_section': {'x': 1}} or (
    cfg.unknown_section == {'x': 1})


def test_load_config_base_plain_dict():
  '''load_config returns resolved plain dicts.'''
  cfg = load_config('base')
  assert isinstance(cfg, dict)
  assert cfg['variables']['target_col'] == 'target'
  assert cfg['parameters']['global_params']['target_col'] == 'target'
  assert '${' not in str(
    {'variables': cfg['variables'],
     'parameters': cfg['parameters']})


def test_load_config_env_overlay_merges():
  '''load_config merges local overlay over base.'''
  cfg = load_config('local')
  assert cfg['parameters']['artifact_store_params']['type'] == 'local'
  # Base section not in overlay survives merge.
  assert 'modeling_params' in cfg['parameters']


def test_load_config_accepts_dir_path():
  '''load_config accepts an env directory path.'''
  cfg = load_config(_root() / 'conf' / 'base')
  assert cfg['variables']['target_col'] == 'target'


def test_load_config_merges_parameters_folder(tmp_path):
  '''load_config merges sorted parameters/*.yml, overlay wins.'''
  conf = tmp_path / 'conf'
  base = conf / 'base' / 'parameters'
  local = conf / 'local' / 'parameters'
  base.mkdir(parents=True)
  local.mkdir(parents=True)
  (conf / 'base' / 'variables.yml').write_text(
    "target_col: target\nreceipt_date: '2024-01-01'\n")
  (base / '01_global.yml').write_text(
    'global_params:\n  target_col: target\n'
    '  shared: from_01\n'
    "run_date: '2024-01-01'\n")
  (base / '02_model.yml').write_text(
    'global_params:\n  shared: from_02\n'
    'modeling_params:\n  task_type: regression\n')
  (base / 'notes.txt').write_text('ignored\n')
  (local / '01_overlay.yml').write_text(
    'global_params:\n  shared: from_local\n')

  cfg = load_config(conf / 'local')
  params = cfg['parameters']
  # Merged keys from both sorted base bucket files.
  assert params['global_params']['target_col'] == 'target'
  assert params['modeling_params']['task_type'] == 'regression'
  # Sorted order: 02_model.yml wins over 01_global.yml ...
  # ... and the local overlay wins over base.
  assert params['global_params']['shared'] == 'from_local'
  assert cfg['variables']['target_col'] == 'target'


def test_config_loader_accepts_dictconfig():
  '''ConfigLoader accepts OmegaConf DictConfig inputs.'''
  variables = OmegaConf.create({
    'target_col': 'target', 'business_features': ['b1'],
    'exclude_features': ['x']})
  loader = ConfigLoader(variables=variables, parameters={})
  assert loader.get('target_col') == 'target'
  assert loader.merged_feature_list(['a', 'x']) == ['a', 'b1']


def test_config_loader_accepts_pydantic_model():
  '''ConfigLoader accepts a pydantic VariablesConfig.'''
  variables = VariablesConfig(
    target_col='target', business_relevant_features=[])
  loader = ConfigLoader(variables=variables, parameters={})
  assert loader.get('target_col') == 'target'
  assert loader.task_type() == 'regression'


def test_config_loader_task_type_preserved():
  '''Existing task_type behavior still works with dicts.'''
  loader = ConfigLoader(
    parameters={'task_type': 'classification'})
  assert loader.task_type() == 'classification'
