#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Loss-function keys in shipped parameters.yml (config only).'''

from pathlib import Path

import yaml


def _load_params():
  '''Load shipped base parameters.yml.'''
  root = Path(__file__).resolve().parents[1]
  with (root / 'conf' / 'base' / 'parameters.yml').open() as handle:
    return yaml.safe_load(handle)


def test_top_level_loss_function():
  '''Top-level loss_function exists for clf and reg.'''
  params = _load_params()
  for section in ('modeling_params', 'modeling_reg_params'):
    value = params[section]['loss_function']
    assert isinstance(value, str) and value


def test_per_model_loss_keys():
  '''Per-model model_params carry loss/objective keys.'''
  params = _load_params()
  for section in ('modeling_params', 'modeling_reg_params'):
    per_model = params[section]['model_params']
    assert per_model['xgboost']['objective']
    assert per_model['xgboost']['loss_function']
    assert per_model['lightgbm']['objective']
    assert per_model['lightgbm']['loss_function']
    assert per_model['catboost']['loss_function']
    assert per_model['sklearn']['loss_function']
    assert per_model['h2o_automl']['loss_function']
    for name, cfg in per_model.items():
      assert isinstance(
          cfg.get('loss_function') or cfg.get('objective'), str), name


def test_model_params_by_name_loss_keys():
  '''model_params_by_name entries carry loss/objective keys.'''
  params = _load_params()
  by_name = params['model_params_by_name']
  assert by_name['xgboost']['objective'] or by_name['xgboost'].get(
      'loss_function')
  assert by_name['lightgbm']['objective'] or by_name['lightgbm'].get(
      'loss_function')
  assert by_name['catboost']['loss_function']
  assert by_name['sklearn']['loss_function']
