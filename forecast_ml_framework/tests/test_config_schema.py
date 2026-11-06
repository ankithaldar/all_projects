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
  assert loader.task_type() == 'classification'


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
              'cost_params', 'governance_params', 'eda_params'):
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


# --- loop3: cross-key recombined oc.env (RED) ---

def test_load_config_blocks_cross_key_oc_env(tmp_path, monkeypatch):
  '''Cross-key ${a}${b} recombining into ${oc.env:...} is blocked.'''
  monkeypatch.setenv('SECRET_ENV', 'leaked-secret-value')
  conf = _write_conf(
    tmp_path,
    "a: '$'\n"
    "b: '{oc.env:SECRET_ENV}'\n"
    "x: '${parameters.a}${parameters.b}'\n")
  with pytest.raises(ValueError, match='oc\\.') as excinfo:
    load_config(conf / 'base')
  assert 'leaked-secret-value' not in str(excinfo.value)


def test_load_config_blocks_cross_key_oc_env_list(
    tmp_path, monkeypatch):
  '''Cross-key recombination inside a list is blocked.'''
  monkeypatch.setenv('SECRET_ENV', 'leaked-secret-value')
  conf = _write_conf(
    tmp_path,
    "a: '$'\n"
    "b: '{oc.env:SECRET_ENV}'\n"
    "items: ['${parameters.a}${parameters.b}']\n")
  with pytest.raises(ValueError, match='oc\\.') as excinfo:
    load_config(conf / 'base')
  assert 'leaked-secret-value' not in str(excinfo.value)


def test_load_config_blocks_cross_key_oc_env_nested(
    tmp_path, monkeypatch):
  '''Cross-key recombination in a nested mapping is blocked.'''
  monkeypatch.setenv('SECRET_ENV', 'leaked-secret-value')
  conf = _write_conf(
    tmp_path,
    "a: '$'\n"
    "b: '{oc.env:SECRET_ENV}'\n"
    "nested:\n"
    "  x: '${parameters.a}${parameters.b}'\n")
  with pytest.raises(ValueError, match='oc\\.') as excinfo:
    load_config(conf / 'base')
  assert 'leaked-secret-value' not in str(excinfo.value)


def test_load_config_blocks_cross_key_oc_env_variables(
    tmp_path, monkeypatch):
  '''Fragments in variables, interpolation in parameters, blocked.'''
  monkeypatch.setenv('SECRET_ENV', 'leaked-secret-value')
  conf = tmp_path / 'conf'
  base = conf / 'base'
  base.mkdir(parents=True)
  (base / 'variables.yml').write_text(
    "target_col: target\n"
    "a: '$'\n"
    "b: '{oc.env:SECRET_ENV}'\n")
  (base / 'parameters.yml').write_text(
    "x: '${variables.a}${variables.b}'\n")
  (base / 'catalog.yml').write_text('{}\n')
  with pytest.raises(ValueError, match='oc\\.') as excinfo:
    load_config(conf / 'base')
  assert 'leaked-secret-value' not in str(excinfo.value)


def test_config_loader_blocks_cross_key_oc_env(monkeypatch):
  '''ConfigLoader rejects cross-key recombined oc.env DictConfig.'''
  monkeypatch.setenv('SECRET_ENV', 'leaked-secret-value')
  variables = OmegaConf.create({
    'a': '$', 'b': '{oc.env:SECRET_ENV}', 'x': '${a}${b}'})
  with pytest.raises(ValueError, match='oc\\.') as excinfo:
    ConfigLoader(variables=variables, parameters={})
  assert 'leaked-secret-value' not in str(excinfo.value)


def test_config_loader_blocks_cross_key_oc_env_list(monkeypatch):
  '''ConfigLoader rejects cross-key recombination in a list.'''
  monkeypatch.setenv('SECRET_ENV', 'leaked-secret-value')
  variables = OmegaConf.create({
    'a': '$', 'b': '{oc.env:SECRET_ENV}',
    'items': ['${a}${b}']})
  with pytest.raises(ValueError, match='oc\\.') as excinfo:
    ConfigLoader(variables=variables, parameters={})
  assert 'leaked-secret-value' not in str(excinfo.value)


def test_kedro_loader_blocks_cross_key_oc_env(
    tmp_path, monkeypatch):
  '''Kedro loader rejects cross-key recombined oc.env.'''
  monkeypatch.setenv('SECRET_ENV', 'leaked-secret-value')
  pytest.importorskip('kedro')
  from forecast_ml_framework import settings
  loader_cls = settings.CONFIG_LOADER_CLASS
  assert loader_cls is not None
  conf = tmp_path / 'conf'
  base = conf / 'base'
  base.mkdir(parents=True)
  (base / 'parameters.yml').write_text(
    "a: '$'\n"
    "b: '{oc.env:SECRET_ENV}'\n"
    "x: '${a}${b}'\n")
  (base / 'catalog.yml').write_text('{}\n')
  try:
    loader = loader_cls(str(conf), env='base', base_env='base')
    with pytest.raises(ValueError, match='oc\\.') as excinfo:
      loader['parameters']
    assert 'leaked-secret-value' not in str(excinfo.value)
  finally:
    from omegaconf import OmegaConf as _OC
    from omegaconf.resolvers import oc as _oc_resolvers
    if not _OC.has_resolver('oc.env'):
      _OC.register_new_resolver('oc.env', _oc_resolvers.env)


def test_load_config_legit_oc_env_default_still_substitutes(
    tmp_path, monkeypatch):
  '''Canonical oc.env:VAR,default still substitutes, no leak.'''
  monkeypatch.setenv('MY_TRACKING', 'leaked-secret-value')
  conf = _write_conf(
    tmp_path,
    "registry_params:\n"
    "  tracking_uri: '${oc.env:MY_TRACKING,mlruns}'\n")
  cfg = load_config(conf / 'base')
  assert cfg['parameters']['registry_params']['tracking_uri'] == 'mlruns'
  assert 'leaked-secret-value' not in str(cfg)


# --- loop2: nested/recombined oc.env (RED) ---

@pytest.mark.parametrize('expr', [
  '${oc.env:FOO,$}{oc.env:SECRET}',
  '${oc.env:FOO,$}{oc.env:SECRET,fallback}',
  '${oc.env:FOO,${oc.env:SECRET}}',
  '${oc.env:FOO,${oc.env:SECRET,bar}}',
])
def test_load_config_blocks_recombined_oc_env(
    tmp_path, monkeypatch, expr):
  '''Recombined oc.env after substitution never leaks env.'''
  monkeypatch.setenv('SECRET', 'leaked-secret-value')
  monkeypatch.setenv('FOO', 'foo-value')
  conf = _write_conf(tmp_path, "x: '%s'\n" % expr)
  with pytest.raises(ValueError, match='oc\\.') as excinfo:
    load_config(conf / 'base')
  assert 'leaked-secret-value' not in str(excinfo.value)
  assert 'foo-value' not in str(excinfo.value)


# --- loop4: bracket-index cross-key recombined oc.env (RED) ---

def test_assert_no_cross_key_oc_bracket_index():
  '''Bracket indexing lst[0] recombining into oc.env is blocked.'''
  from forecast_ml_framework.core.config_schema import (
    assert_no_cross_key_oc)
  plain = {'parameters': {
    'lst': ['$','{oc.env:SECRET}'],
    'x': '${parameters.lst[0]}${parameters.lst[1]}'}}
  with pytest.raises(ValueError, match='oc\\.'):
    assert_no_cross_key_oc(plain)


def test_assert_no_cross_key_oc_bracket_nested_and_relative():
  '''Nested / relative bracket indexing is expanded and blocked.'''
  from forecast_ml_framework.core.config_schema import (
    assert_no_cross_key_oc)
  plain = {'parameters': {
    'a': {'lst': ['$','{oc.env:SECRET}']},
    'x': '${parameters.a.lst[0]}${parameters.a.lst[1]}'}}
  with pytest.raises(ValueError, match='oc\\.'):
    assert_no_cross_key_oc(plain)
  plain2 = {'lst': ['$','{oc.env:SECRET}'],
            'x': '${.lst[0]}${.lst[1]}'}
  with pytest.raises(ValueError, match='oc\\.'):
    assert_no_cross_key_oc(plain2)


def test_load_config_blocks_cross_key_oc_bracket(tmp_path, monkeypatch):
  '''load_config blocks bracket-index recombination, no env leak.'''
  monkeypatch.setenv('SECRET', 'leaked-secret-value')
  conf = _write_conf(
    tmp_path,
    "lst: ['$','{oc.env:SECRET}']\n"
    "x: '${parameters.lst[0]}${parameters.lst[1]}'\n")
  with pytest.raises(ValueError, match='oc\\.') as excinfo:
    load_config(conf / 'base')
  assert 'leaked-secret-value' not in str(excinfo.value)


def test_config_loader_blocks_cross_key_oc_bracket(monkeypatch):
  '''ConfigLoader rejects bracket-index recombined oc.env.'''
  monkeypatch.setenv('SECRET', 'leaked-secret-value')
  variables = OmegaConf.create({
    'lst': ['$','{oc.env:SECRET}'],
    'x': '${lst[0]}${lst[1]}'})
  with pytest.raises(ValueError, match='oc\\.') as excinfo:
    ConfigLoader(variables=variables, parameters={})
  assert 'leaked-secret-value' not in str(excinfo.value)


# --- loop4: settings.RestrictedOmegaConfigLoader branches ---


def test_settings_reject_oc_list_and_tuple_branches():
  '''_reject_oc recurses lists/tuples and ignores plain values.'''
  from forecast_ml_framework import settings
  cls = settings.CONFIG_LOADER_CLASS
  assert cls is not None
  assert Path(settings.CONF_SOURCE).is_absolute()
  assert (Path(settings.CONF_SOURCE) / 'base' / 'catalog.yml').exists()
  assert settings.HOOKS == ()
  assert settings.__all__ == [
    'CONFIG_LOADER_CLASS', 'CONF_SOURCE', 'HOOKS']
  cls._reject_oc({'a': [1, 2], 'b': ('x', 'y'), 'c': 'plain'})
  cls._reject_oc(['plain', ('also', 1)])
  with pytest.raises(ValueError, match='oc\\.'):
    cls._reject_oc(['ok', '${oc.env:SECRET}'])
  with pytest.raises(ValueError, match='oc\\.'):
    cls._reject_oc(('ok', {'x': '${ OC.env:SECRET}'}))
  with pytest.raises(ValueError, match='oc\\.'):
    cls._reject_oc({'a': ['${oc.create:dict}']})


def test_settings_resolve_environment_variables_clean_and_blocked():
  '''_resolve_environment_variables allows clean, blocks oc.'''
  from forecast_ml_framework import settings
  cls = settings.CONFIG_LOADER_CLASS
  assert cls is not None
  assert cls._resolve_environment_variables(
    OmegaConf.create({'x': 'y', 'lst': ['a']})) is None
  with pytest.raises(ValueError, match='oc\\.'):
    cls._resolve_environment_variables(
      OmegaConf.create({'x': '${oc.env:SECRET}'}))
  with pytest.raises(ValueError, match='oc\\.'):
    plain = {'a': '$', 'b': '{oc.env:SECRET}', 'x': '${a}${b}'}
    cls._resolve_environment_variables(OmegaConf.create(plain))


# --- loop5: unquoted bracket keys in cross-key expansion (RED) ---

def test_assert_no_cross_key_oc_unquoted_bracket_key():
  '''Unquoted bracket key d[k1] recombining into oc.env is blocked.'''
  from forecast_ml_framework.core.config_schema import (
    assert_no_cross_key_oc)
  plain = {'parameters': {
    'd': {'k1': '$', 'k2': '{oc.env:SECRET}'},
    'x': '${parameters.d[k1]}${parameters.d[k2]}'}}
  with pytest.raises(ValueError, match='oc\\.'):
    assert_no_cross_key_oc(plain)


def test_assert_no_cross_key_oc_unquoted_bracket_hyphen_key():
  '''Unquoted bracket key d[k-1] recombining into oc.env is blocked.'''
  from forecast_ml_framework.core.config_schema import (
    assert_no_cross_key_oc)
  plain = {'parameters': {
    'd': {'k-1': '$', 'k-2': '{oc.env:SECRET}'},
    'x': '${parameters.d[k-1]}${parameters.d[k-2]}'}}
  with pytest.raises(ValueError, match='oc\\.'):
    assert_no_cross_key_oc(plain)


def test_assert_no_cross_key_oc_unquoted_bracket_nested():
  '''Nested unquoted bracket keys d[a][b] are expanded and blocked.'''
  from forecast_ml_framework.core.config_schema import (
    assert_no_cross_key_oc)
  plain = {'parameters': {
    'd': {'a': {'b': '$', 'c': '{oc.env:SECRET}'}},
    'x': '${parameters.d[a][b]}${parameters.d[a][c]}'}}
  with pytest.raises(ValueError, match='oc\\.'):
    assert_no_cross_key_oc(plain)


# --- loop7: _blocked_oc_resolvers thread race (RED) ---

def test_blocked_oc_resolvers_threaded_no_leak_no_crash(monkeypatch):
  '''Concurrent _blocked_oc_resolvers entries never leak env.'''
  import threading

  from forecast_ml_framework.core.config_schema import _blocked_oc_resolvers
  monkeypatch.setenv('SECRET', 'leaked-secret-value')
  errors = []
  leaks = []
  barrier = threading.Barrier(4)

  def _worker():
    try:
      barrier.wait(timeout=10)
      for _ in range(25):
        with _blocked_oc_resolvers():
          cfg = OmegaConf.create({'x': '${oc.env:SECRET}'})
          try:
            OmegaConf.resolve(cfg)
          except ValueError:
            pass
          else:
            leaks.append(OmegaConf.to_container(cfg, resolve=True))
    except Exception as exc:  # noqa: BLE001 - record any crash
      errors.append(exc)

  threads = [threading.Thread(target=_worker) for _ in range(4)]
  for thread in threads:
    thread.start()
  for thread in threads:
    thread.join(timeout=30)
  assert errors == []
  assert leaks == []
  assert 'leaked-secret-value' not in str(leaks)
  # Resolvers restored after all exits: oc.env resolves again.
  cfg = OmegaConf.create({'x': '${oc.env:SECRET}'})
  OmegaConf.resolve(cfg)
  assert OmegaConf.to_container(cfg, resolve=True)['x'] == (
    'leaked-secret-value')


def test_blocked_oc_resolvers_interleaved_no_leak(monkeypatch):
  '''Thread A exit must not restore resolvers while B is inside.'''
  import threading

  from forecast_ml_framework.core.config_schema import _blocked_oc_resolvers
  monkeypatch.setenv('SECRET', 'leaked-secret-value')
  entered_a = threading.Event()
  entered_b = threading.Event()
  exit_a = threading.Event()
  exit_b = threading.Event()
  results = {}

  def _thread_a():
    with _blocked_oc_resolvers():
      entered_a.set()
      exit_a.wait(timeout=10)
    results['a_done'] = True

  def _thread_b():
    with _blocked_oc_resolvers():
      entered_b.set()
      exit_b.wait(timeout=10)
      cfg = OmegaConf.create({'x': '${oc.env:SECRET}'})
      try:
        OmegaConf.resolve(cfg)
        results['b'] = OmegaConf.to_container(cfg, resolve=True)
      except ValueError as exc:
        results['b_error'] = str(exc)

  thread_a = threading.Thread(target=_thread_a)
  thread_b = threading.Thread(target=_thread_b)
  thread_a.start()
  assert entered_a.wait(timeout=10)
  thread_b.start()
  # Fixed code: B blocks on the lock until A exits, so entered_b
  # times out; old code: B enters immediately. Either way,
  # release A first, then B resolves only while blocked.
  entered_b.wait(timeout=1)
  exit_a.set()
  thread_a.join(timeout=10)
  assert entered_b.wait(timeout=10)
  exit_b.set()
  thread_b.join(timeout=10)
  assert 'b' not in results
  assert 'oc.' in results.get('b_error', '')
  assert 'leaked-secret-value' not in str(results)
