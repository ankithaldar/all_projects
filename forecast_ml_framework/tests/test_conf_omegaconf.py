#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Base conf loads and resolves via OmegaConf (pydantic-era conf).'''

from pathlib import Path

import pytest

omegaconf = pytest.importorskip('omegaconf')
from omegaconf import OmegaConf  # noqa: E402


def _parameters():
  '''Load base parameters by merging every conf/base/parameters/*.yml.'''
  root = Path(__file__).resolve().parents[1]
  params_dir = root / 'conf' / 'base' / 'parameters'
  merged = OmegaConf.create({})
  for path in sorted(params_dir.glob('*.yml')):
    merged = OmegaConf.merge(merged, OmegaConf.load(path))
  return merged


def _merged():
  '''Load base conf merged as variables + parameters.'''
  root = Path(__file__).resolve().parents[1]
  base = root / 'conf' / 'base'
  cfg = OmegaConf.merge(
      {'variables': OmegaConf.load(base / 'variables.yml'),
       'parameters': _parameters()})
  OmegaConf.resolve(cfg)
  return cfg


def test_interpolations_resolve():
  '''Interpolations resolve to the single target_col definition.'''
  cfg = _merged()
  assert cfg.variables.target_col == 'target'
  assert cfg.parameters.global_params.target_col == 'target'
  assert cfg.parameters.sampling_params.stratify_col == 'target'
  assert 'stratify_col' not in cfg.parameters.sampling_reg_params
  assert cfg.parameters.train_test_split_params.stratify_col == 'target'
  assert cfg.variables.bucket_segment_col == 'region'


def test_key_sections_exist_and_fully_resolved():
  '''Key sections exist and no interpolation is left unresolved.'''
  cfg = _merged()
  for section in ('data_prep_params', 'modeling_params',
                  'feature_selection_params', 'artifact_store_params',
                  'spark_params', 'global_params'):
    assert section in cfg.parameters, section
  plain = OmegaConf.to_container(cfg, resolve=True)
  assert plain['parameters']['global_params']['target_col'] == 'target'
  assert '${' not in str(plain)


def test_parameters_directory_merged_superset():
  '''parameters/ dir merges to a superset of the old monolith keys.

  Every bucket file parses as YAML and the merged top-level keys
  cover all sections the monolith defined (incl. per-bucket spot
  checks), with no key defined in two bucket files.
  '''
  import yaml
  root = Path(__file__).resolve().parents[1]
  params_dir = root / 'conf' / 'base' / 'parameters'
  assert params_dir.is_dir(), 'conf/base/parameters/ directory missing'
  assert not (root / 'conf' / 'base' / 'parameters.yml').exists(), (
      'monolith parameters.yml must be deleted after split')
  expected_bucket_keys = {
      'global.yml': {'global_params', 'score_model_id', 'brand'},
      'data.yml': {'data_params', 'sampling_params',
                   'pandas_conversion_params',
                   'train_test_split_params', 'validation_params'},
      'preprocessing.yml': {'data_prep_params',
                            'feature_selection_params'},
      'evaluation.yml': {'evaluation_params'},
      'modeling.yml': {'modeling_params', 'modeling_reg_params',
                       'models_to_use', 'calibration_params',
                       'cv_params', 'h2o_automl_params'},
      'platform.yml': {'spark_params', 'artifact_store_params',
                       'mlflow_params', 'dvc_params', 'shap_params',
                       'chart_params', 'registry_params',
                       'scoring_params', 'infra_params',
                       'monitoring_params', 'retraining_params',
                       'feature_store_params', 'ab_params',
                       'cost_params', 'governance_params'},
  }
  merged = {}
  for name, expected in expected_bucket_keys.items():
    path = params_dir / name
    assert path.is_file(), f'bucket file missing: {name}'
    with path.open() as handle:
      doc = yaml.safe_load(handle)
    assert isinstance(doc, dict) and doc, name
    assert expected <= set(doc), (name, expected - set(doc))
    dupes = set(doc) & set(merged)
    assert not dupes, (name, dupes)
    merged.update(doc)
  # Superset of every section the monolith documented.
  for section in ('global_params', 'data_params', 'sampling_params',
                  'sampling_reg_params', 'pandas_conversion_params',
                  'evaluation_params', 'train_test_split_params',
                  'data_prep_params', 'feature_selection_params',
                  'modeling_params', 'modeling_reg_params',
                  'calibration_params', 'cv_params', 'spark_params',
                  'artifact_store_params', 'mlflow_params',
                  'dvc_params', 'shap_params', 'chart_params',
                  'validation_params', 'registry_params',
                  'scoring_params', 'infra_params',
                  'monitoring_params', 'retraining_params',
                  'feature_store_params', 'ab_params', 'cost_params',
                  'governance_params'):
    assert section in merged, section
  assert 'metrics_to_calculate' in merged['evaluation_params']
