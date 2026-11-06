#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''This framework metrics parity: ntile eval-metric parsing/assignment.'''

import numpy as np
import pandas as pd
import pytest

from forecast_ml_framework.pipelines.modeling.ntile import (
  assign_ntile,
  parse_eval_metric,
  select_ntile_row,
)


def test_parse_eval_metric_classification_defaults_lift_decile():
  '''Classification parses to lift/decile/10 by default.'''
  assert parse_eval_metric('lift', 'classification') == (
      'lift', 'decile', 10)
  assert parse_eval_metric('lift_top_decile', 'classification') == (
      'lift', 'decile', 10)
  assert parse_eval_metric(None, 'classification') == (
      'lift', 'decile', 10)


def test_parse_eval_metric_regression_defaults_rmse_overall():
  '''Regression parses to rmse/overall/0 by default.'''
  assert parse_eval_metric('rmse', 'regression') == (
      'rmse', 'overall', 0)
  assert parse_eval_metric('mae', 'regression') == (
      'mae', 'overall', 0)
  assert parse_eval_metric('rmse_centile_100', 'regression') == (
      'rmse', 'centile', 100)
  assert parse_eval_metric(None, 'regression') == (
      'rmse', 'overall', 0)


def test_parse_eval_metric_parity_grammar_first():
  '''Parity grammar LIFT_DECILE_10 / RMSE_CENTILE_100 canonicalises.'''
  assert parse_eval_metric('LIFT_DECILE_10', 'classification') == (
      'lift', 'decile', 10)
  assert parse_eval_metric('RMSE_CENTILE_100', 'regression') == (
      'rmse', 'centile', 100)
  assert parse_eval_metric('PRECISION_DECILE_5', 'classification') == (
      'precision', 'decile', 5)
  assert parse_eval_metric('RECALL_CENTILE_20', 'classification') == (
      'recall', 'centile', 20)
  assert parse_eval_metric('ACCURACY_DECILE_10', 'classification') == (
      'accuracy', 'decile', 10)
  assert parse_eval_metric('F1_CENTILE_50', 'classification') == (
      'f1', 'centile', 50)
  assert parse_eval_metric('AUC_DECILE_10', 'classification') == (
      'auc', 'decile', 10)
  assert parse_eval_metric('BRIER_SCORE_DECILE_10',
                           'classification') == (
      'brier_score', 'decile', 10)
  assert parse_eval_metric('MAE_CENTILE_100', 'regression') == (
      'mae', 'centile', 100)
  assert parse_eval_metric('MAPE_DECILE_10', 'regression') == (
      'mape', 'decile', 10)
  assert parse_eval_metric('MSE_DECILE_10', 'regression') == (
      'mse', 'decile', 10)
  assert parse_eval_metric('MSLE_CENTILE_100', 'regression') == (
      'msle', 'centile', 100)
  assert parse_eval_metric('RMSLE_DECILE_10', 'regression') == (
      'rmsle', 'decile', 10)
  assert parse_eval_metric('ADJ_R2_CENTILE_100', 'regression') == (
      'adj_r2', 'centile', 100)
  assert parse_eval_metric('R2_DECILE_10', 'regression') == (
      'r2', 'decile', 10)
  assert parse_eval_metric('RMSE', 'regression') == (
      'rmse', 'overall', 0)
  assert parse_eval_metric('LIFT', 'classification') == (
      'lift', 'decile', 10)


def test_parse_eval_metric_forecast_aliases_map_to_canonical():
  '''Forecast names alias to the same canonical metric.'''
  assert parse_eval_metric('roc_auc', 'classification') == (
      'auc', 'decile', 10)
  assert parse_eval_metric('precision_weighted', 'classification') == (
      'precision', 'decile', 10)
  assert parse_eval_metric('recall_weighted', 'classification') == (
      'recall', 'decile', 10)
  assert parse_eval_metric('f1_weighted', 'classification') == (
      'f1', 'decile', 10)
  assert parse_eval_metric('brier_score', 'classification') == (
      'brier_score', 'decile', 10)
  assert parse_eval_metric('root_mean_squared_error', 'regression') == (
      'rmse', 'overall', 0)


def test_parse_eval_metric_forecast_aliases_with_kind_count():
  '''Forecast aliases with _<KIND>[_<n>] keep canonical+count.'''
  assert parse_eval_metric('ROC_AUC_DECILE_10', 'classification') == (
      'auc', 'decile', 10)
  assert parse_eval_metric('PRECISION_WEIGHTED_DECILE_10',
                           'classification') == ('precision', 'decile', 10)
  assert parse_eval_metric('PRECISION_WEIGHTED', 'classification') == (
      'precision', 'decile', 10)
  assert parse_eval_metric('ROOT_MEAN_SQUARED_ERROR_DECILE',
                           'regression') == ('rmse', 'decile', 10)
  assert parse_eval_metric('ROOT_MEAN_SQUARED_ERROR_DECILE_10',
                           'regression') == ('rmse', 'decile', 10)
  assert parse_eval_metric('ROC_AUC_CENTILE_100', 'classification') == (
      'auc', 'centile', 100)


def test_parse_eval_metric_unknown_parity_warns_and_defaults():
  '''Unknown parity grammar warns and falls back to task default.'''
  with pytest.warns(UserWarning, match='unknown'):
    assert parse_eval_metric('FOO_DECILE_3', 'classification') == (
        'lift', 'decile', 10)
  with pytest.warns(UserWarning, match='unknown'):
    assert parse_eval_metric('FOO_CENTILE_3', 'regression') == (
        'rmse', 'overall', 0)


def test_parse_eval_metric_unknown_warns_and_defaults():
  '''Unknown metric warns and falls back to the task default.'''
  with pytest.warns(UserWarning, match='unknown'):
    assert parse_eval_metric('not_a_metric', 'classification') == (
        'lift', 'decile', 10)
  with pytest.warns(UserWarning, match='unknown'):
    assert parse_eval_metric('not_a_metric', 'regression') == (
        'rmse', 'overall', 0)


def test_parse_eval_metric_kind_suffix_without_count_maps_canonical():
  '''Suffix forms map canonical metric + kind default count.'''
  assert parse_eval_metric('PRECISION_DECILE', 'classification') == (
      'precision', 'decile', 10)
  assert parse_eval_metric('AUC_CENTILE', 'classification') == (
      'auc', 'centile', 100)
  assert parse_eval_metric('MAE_DECILE', 'regression') == (
      'mae', 'decile', 10)
  assert parse_eval_metric('LIFT_QUINTILE', 'classification') == (
      'lift', 'quintile', 5)
  assert parse_eval_metric('LIFT_QUARTILE', 'classification') == (
      'lift', 'quartile', 4)
  assert parse_eval_metric('RMSE_CENTILE', 'regression') == (
      'rmse', 'centile', 100)
  assert parse_eval_metric('F1_QUINTILE', 'classification') == (
      'f1', 'quintile', 5)


def test_assign_ntile_is_one_based_qcut():
  '''assign_ntile gives 1-based qcut tiles, lowest scores first.'''
  scores = np.arange(100, dtype=float)
  tiles = assign_ntile(scores, 'decile', 10)
  assert len(tiles) == 100
  assert tiles.min() == 1 and tiles.max() == 10
  assert tiles[0] == 1 and tiles[-1] == 10
  assert (tiles[:10] == 1).all() and (tiles[-10:] == 10).all()
  assert assign_ntile(scores, 'overall', 0).tolist() == [0] * 100
  assert assign_ntile(np.full(20, 0.5), 'decile', 10).tolist() == (
      [1] * 20)


def test_select_ntile_row_falls_back_to_max():
  '''select_ntile_row picks the ntile row, else the max row.'''
  df = pd.DataFrame({'ntile': [1, 2, 3], 'metric': [0.1, 0.2, 0.3]})
  assert select_ntile_row(df, 'ntile', 2)['metric'] == 0.2
  assert select_ntile_row(df, 'ntile', 99)['metric'] == 0.3
  assert select_ntile_row(df, 'ntile', 99)['ntile'] == 3
  with pytest.raises(ValueError, match='empty'):
    select_ntile_row(df.iloc[0:0], 'ntile', 1)
  with pytest.raises(ValueError, match='missing column'):
    select_ntile_row(df, 'nope', 1)


def test_assign_ntile_nan_scores_map_to_tile_1():
  '''NaN scores map to tile 1 (never tile 0); tiles stay 1-based.'''
  scores = np.array([0.1, 0.2, np.nan, 0.9, 0.8, 0.3, 0.4, 0.5,
                     0.6, 0.7])
  tiles = assign_ntile(scores, 'decile', 10)
  assert tiles.min() >= 1
  assert tiles[2] == 1
  assert tiles.max() <= 10
