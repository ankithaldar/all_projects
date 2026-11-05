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


def _shipped_parameters_raw():
  '''Return merged shipped base parameters/*.yml as a dict.'''
  import yaml
  raw = {}
  for path in sorted((_root() / 'conf' / 'base' / 'parameters').glob('*.yml')):
    with path.open() as handle:
      raw.update(yaml.safe_load(handle) or {})
  return raw


def test_phase3_sections_are_typed_nested_models():
  '''Phase3 sections validate as nested extra=allow models.'''
  from forecast_ml_framework.core.config_schema import (
    AbParams, CostParams, FeatureStoreParams, GovernanceParams,
    InfraParams, MonitoringParams, RegistryParams, RetrainingParams,
    ScoringParams)
  cfg = validate_parameters(_shipped_parameters_raw())
  for attr, cls in (
      ('registry_params', RegistryParams),
      ('scoring_params', ScoringParams),
      ('infra_params', InfraParams),
      ('monitoring_params', MonitoringParams),
      ('retraining_params', RetrainingParams),
      ('feature_store_params', FeatureStoreParams),
      ('ab_params', AbParams),
      ('cost_params', CostParams),
      ('governance_params', GovernanceParams)):
    assert isinstance(getattr(cfg, attr), cls), attr
  dumped = cfg.model_dump()
  for section in ('registry_params', 'scoring_params', 'infra_params',
                  'monitoring_params', 'retraining_params',
                  'feature_store_params', 'ab_params', 'cost_params',
                  'governance_params'):
    assert section in dumped, section
    assert isinstance(dumped[section], dict) and dumped[section], section
  assert dumped['scoring_params']['input_dataset'] == 'scoring_input'
  assert dumped['registry_params']['model_name'] == 'forecast_model'
  assert dumped['monitoring_params']['psi_threshold'] == 0.1
  assert dumped['retraining_params']['schedule_days'] == 7
  assert dumped['feature_store_params']['entity_key'] == 'id'
  assert dumped['ab_params']['mode'] == 'shadow'
  assert dumped['cost_params']['compute_price_per_hour'] == 0.5
  assert dumped['governance_params']['retention_days'] == 365
  assert dumped['infra_params']['gcp'] is not None


def test_pandas_conversion_max_rows_for_pandas():
  '''PandasConversionParams carries max_rows_for_pandas (5_000_000).'''
  from forecast_ml_framework.core.config_schema import (
    PandasConversionParams)
  assert PandasConversionParams().max_rows_for_pandas == 5000000
  cfg = validate_parameters(_shipped_parameters_raw())
  assert cfg.pandas_conversion_params.max_rows_for_pandas == 5000000


def test_round_trip_every_shipped_top_level_key():
  '''Every shipped parameters/*.yml top-level key round-trips.'''
  raw = _shipped_parameters_raw()
  cfg = validate_parameters(raw)
  dumped = cfg.model_dump()
  for key in raw:
    assert key in dumped, key
  # Known sections are typed fields, not silently-ignored extras.
  for key in ('global_params', 'modeling_params', 'registry_params',
              'scoring_params', 'infra_params', 'monitoring_params',
              'retraining_params', 'feature_store_params', 'ab_params',
              'cost_params', 'governance_params'):
    assert key not in (cfg.model_extra or {}), key


def test_global_params_task_type_default_classification():
  '''GlobalParams.task_type defaults to classification (documented).'''
  from forecast_ml_framework.core.config_schema import GlobalParams
  assert GlobalParams().task_type == 'classification'
  assert ParametersConfig().global_params.task_type == 'classification'
  assert ParametersConfig().modeling_params.task_type == 'classification'


def _write_conf(tmp_path, parameters_text):
  '''Write a minimal conf tree and return its root.'''
  conf = tmp_path / 'conf'
  base = conf / 'base'
  base.mkdir(parents=True)
  (base / 'variables.yml').write_text('target_col: target\n')
  (base / 'parameters.yml').write_text(parameters_text)
  (base / 'catalog.yml').write_text('{}\n')
  return conf


def test_load_config_blocks_oc_env_secret(tmp_path):
  '''load_config raises ValueError for ${oc.env:SECRET} (no default).'''
  conf = _write_conf(
    tmp_path, "x: '${oc.env:SECRET}'\n")
  with pytest.raises(ValueError, match='oc\\.'):
    load_config(conf / 'base')


def test_load_config_blocks_oc_create(tmp_path):
  '''load_config raises ValueError for ${oc.create:...}.'''
  conf = _write_conf(
    tmp_path, "x: '${oc.create:dict}'\n")
  with pytest.raises(ValueError, match='oc\\.'):
    load_config(conf / 'base')


def test_load_config_does_not_read_env_for_oc_env_default(
    tmp_path, monkeypatch):
  '''oc.env with a default resolves to the default, never the env.'''
  monkeypatch.setenv('MY_TRACKING', 'leaked-secret-value')
  conf = _write_conf(
    tmp_path, "registry_params:\n"
    "  tracking_uri: '${oc.env:MY_TRACKING,mlruns}'\n")
  cfg = load_config(conf / 'base')
  assert cfg['parameters']['registry_params']['tracking_uri'] == 'mlruns'
  assert 'leaked-secret-value' not in str(cfg)


def test_config_loader_rejects_oc_env_dictconfig():
  '''ConfigLoader rejects DictConfig containing ${oc.env:...}.'''
  variables = OmegaConf.create({'x': '${oc.env:SECRET}'})
  with pytest.raises(ValueError, match='oc\\.'):
    ConfigLoader(variables=variables, parameters={})


def test_config_loader_rejects_oc_create_dictconfig():
  '''ConfigLoader rejects DictConfig containing ${oc.create:...}.'''
  variables = OmegaConf.create({'x': '${oc.create:dict}'})
  with pytest.raises(ValueError, match='oc\\.'):
    ConfigLoader(variables=variables, parameters={})


def test_config_loader_reraises_pydantic_validation_error():
  '''ConfigLoader re-raises pydantic validation errors (no swallow).'''
  class BadConfig:
    '''Fake pydantic-like model whose validation always fails.'''

    def model_dump(self):
      return {'target_col': 123}

    @classmethod
    def model_validate(cls, dumped):
      raise ValueError('invalid target_col')

  with pytest.raises(ValueError, match='invalid target_col'):
    ConfigLoader(variables=BadConfig(), parameters={})


def test_config_loader_task_type_from_global_params():
  '''Explicit global_params.task_type wins in ConfigLoader.'''
  loader = ConfigLoader(
    parameters={'global_params': {'task_type': 'classification'}})
  assert loader.task_type() == 'classification'

# --- cycle2 oc whitespace / defaults (RED) ---
import os as _os

@pytest.mark.parametrize('expr', [
  '${ oc.env:SECRET}', '${oc.env:SECRET }', '${ OC.env:SECRET}',
  '${oc .env:SECRET}', '${ oc.env:SECRET,default}',
  '${ OC.ENV:SECRET,default}',
])
def test_load_config_rejects_spaced_oc(tmp_path, monkeypatch, expr):
  '''Spaced/case oc interpolations are rejected, never leak env.'''
  monkeypatch.setenv('SECRET', 'leaked-secret-value')
  conf = _write_conf(tmp_path, "x: '%s'\n" % expr)
  with pytest.raises(ValueError, match='oc\\.'):
    load_config(conf / 'base')
  values = load_config.__doc__ or ''
  assert 'leaked-secret-value' not in values

@pytest.mark.parametrize('expr', [
  '${ oc.env:SECRET}', '${ OC.env:SECRET}',
])
def test_config_loader_rejects_spaced_oc(expr):
  '''ConfigLoader rejects spaced/case oc DictConfig.'''
  variables = OmegaConf.create({'x': expr})
  with pytest.raises(ValueError, match='oc\\.'):
    ConfigLoader(variables=variables, parameters={})

def test_kedro_loader_rejects_spaced_oc(tmp_path, monkeypatch):
  '''Kedro CONFIG_LOADER_CLASS path rejects spaced oc.env.'''
  monkeypatch.setenv('SECRET', 'leaked-secret-value')
  pytest.importorskip('kedro')
  from forecast_ml_framework import settings
  loader_cls = settings.CONFIG_LOADER_CLASS
  assert loader_cls is not None
  conf = tmp_path / 'conf'
  base = conf / 'base'
  base.mkdir(parents=True)
  (base / 'parameters.yml').write_text(
    "x: '${ oc.env:SECRET}'\n")
  (base / 'catalog.yml').write_text('{}\n')
  try:
    loader = loader_cls(str(conf), env='base', base_env='base')
    with pytest.raises(ValueError, match='oc\\.'):
      loader['parameters']
  finally:
    # Kedro clears global oc.env resolver on init; restore it
    # so later OmegaConf tests are not polluted.
    from omegaconf import OmegaConf as _OC
    from omegaconf.resolvers import oc as _oc_resolvers
    if not _OC.has_resolver('oc.env'):
      _OC.register_new_resolver('oc.env', _oc_resolvers.env)

def test_scoring_params_max_rows_default():
  '''ScoringParams.max_rows defaults to 100000, not None.'''
  from forecast_ml_framework.core.config_schema import ScoringParams
  assert ScoringParams().max_rows == 100000

def test_cost_params_new_names_and_old_alias():
  '''CostParams uses new names; old names are deprecated aliases.'''
  from forecast_ml_framework.core.config_schema import (
    CostParams, validate_parameters)
  cost = CostParams()
  assert cost.data_scan_price_per_gb == 0.005
  assert cost.storage_price_per_gb_month == 0.02
  old = CostParams(storage_price_per_gb=0.99,
          scan_price_per_tb_scanned=10.0)
  # old scan_price_per_tb_scanned=10 USD/TB == 0.01 USD/GB
  assert old.storage_price_per_gb_month == 0.99
  assert old.data_scan_price_per_gb == pytest.approx(0.01)
  shipped = validate_parameters(_shipped_parameters_raw())
  assert shipped.cost_params.storage_price_per_gb_month == 0.02
  assert shipped.cost_params.data_scan_price_per_gb == 0.005
