#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Fail-fast validation for local profile (clean-ux Task 8).'''

from pathlib import Path

import pandas as pd
import pytest
import yaml


def _root():
  '''Return project root.'''
  return Path(__file__).resolve().parents[1]


def test_local_validation_is_fail_fast():
  '''Local overlay validation_params fail_fast is True.'''
  local = yaml.safe_load(
    (_root() / 'conf/local/parameters.yml').read_text())
  assert local['validation_params']['fail_fast'] is True
  assert local['validation_params']['max_null_ratio'] == 0.5


def test_base_validation_default_unchanged():
  '''Base validation_params fail_fast stays False.'''
  base = yaml.safe_load(
    (_root() / 'conf/base/parameters/data.yml').read_text())
  assert base['validation_params']['fail_fast'] is False


def test_validate_data_fail_fast_raises_on_missing_col():
  '''validate_data with fail_fast raises on missing column.'''
  from forecast_ml_framework.pipelines.data_prep.nodes import (
    validate_data,
  )
  df = pd.DataFrame({'id': [1]})
  variables = {'target_col': 'target', 'feature_cols': ['f1']}
  with pytest.raises(ValueError, match='Missing required columns'):
    validate_data(df, {'validation_params': {'fail_fast': True}},
                  variables)
