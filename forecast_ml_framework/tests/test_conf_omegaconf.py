#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Base conf loads and resolves via OmegaConf (pydantic-era conf).'''

from pathlib import Path

import pytest

omegaconf = pytest.importorskip('omegaconf')
from omegaconf import OmegaConf  # noqa: E402


def _merged():
  '''Load base conf merged as variables + parameters.'''
  root = Path(__file__).resolve().parents[1]
  base = root / 'conf' / 'base'
  cfg = OmegaConf.merge(
      {'variables': OmegaConf.load(base / 'variables.yml'),
       'parameters': OmegaConf.load(base / 'parameters.yml')})
  OmegaConf.resolve(cfg)
  return cfg


def test_interpolations_resolve():
  '''Interpolations resolve to the single target_col definition.'''
  cfg = _merged()
  assert cfg.variables.target_col == 'target'
  assert cfg.parameters.global_params.target_col == 'target'
  assert cfg.parameters.sampling_params.stratify_col == 'target'
  assert cfg.parameters.sampling_reg_params.stratify_col == 'target'
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
