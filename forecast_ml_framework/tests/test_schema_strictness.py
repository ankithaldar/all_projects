#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Strictness tests: typed schema and single task_type default.'''

import pytest

pytest.importorskip('pydantic')
pytest.importorskip('omegaconf')

from pydantic import ValidationError  # noqa: E402

from forecast_ml_framework.core.config_loader import (  # noqa: E402
  ConfigLoader,
)
from forecast_ml_framework.core.config_schema import (  # noqa: E402
  EdaParams,
)


def test_eda_params_rejects_wrong_types():
  '''EdaParams rejects non-numeric values for numeric fields.'''
  with pytest.raises(ValidationError):
    EdaParams(missing_threshold='not-a-float')
  with pytest.raises(ValidationError):
    EdaParams(cardinality_threshold='not-an-int')
  with pytest.raises(ValidationError):
    EdaParams(leakage_threshold='not-a-float')
  with pytest.raises(ValidationError):
    EdaParams(zscore_threshold='not-a-float')
  with pytest.raises(ValidationError):
    EdaParams(top_correlations='not-an-int')
  with pytest.raises(ValidationError):
    EdaParams(top_values='not-an-int')


def test_task_type_single_default_is_classification():
  '''ConfigLoader defaults to classification (single default).

  Empty config and unknown hints fall back to classification;
  an explicit regression hint still wins.
  '''
  assert ConfigLoader().task_type() == 'classification'
  assert ConfigLoader(
    parameters={'target_col': 'price'}).task_type() == 'classification'
  assert ConfigLoader(
    parameters={'task_type': 'regression'}).task_type() == 'regression'
  assert ConfigLoader(
    parameters={'task_type': 'unknown-hint'}).task_type() == (
    'classification')
