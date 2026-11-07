#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Malformed-input tests for chart functions.

Every chart must return None (never raise, never leak a
figure) on empty/None/mismatched input, and return a
figure on valid input so these tests cannot pass vacuously.
'''

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from forecast_ml_framework.pipelines.interpretability.charts import (
  calibration_curve,
  confidence_histogram,
  error_by_segment_bar,
  pdp_1d_chart,
  permutation_importance_bar,
  prediction_distribution_by_class,
  residual_vs_fitted,
  shap_bar,
  shap_beeswarm,
  shap_heatmap,
)
from forecast_ml_framework.pipelines.modeling.charts import (
  bias_variance,
  error_correlation_heatmap,
  ice_plot,
  model_comparison_box,
  pr_overlay,
)


def _assert_no_fig_leak(before):
  '''Assert no figure numbers leaked.

  Args:
    before: Set of figure numbers captured before the call.
  '''
  assert set(plt.get_fignums()) == before


def test_bias_variance_empty_returns_none():
  '''Empty scores return None, no figure leak.

  Fails if bias_variance plots empty scores instead of
  returning None.
  '''
  before = set(plt.get_fignums())
  assert bias_variance([], []) is None
  _assert_no_fig_leak(before)


def test_bias_variance_single_point_returns_none():
  '''Fewer than two points return None.

  Fails if bias_variance plots a single point.
  '''
  assert bias_variance([1.0, 2.0], [1.0]) is None


def test_bias_variance_mismatched_truncates():
  '''Mismatched lengths truncate to the shorter side.

  Fails if bias_variance raises on mismatched lengths.
  '''
  before = set(plt.get_fignums())
  assert bias_variance(
    [1.0, 2.0, 3.0, 4.0], [1.0, 2.0]) is not None
  _assert_no_fig_leak(before)


def test_bias_variance_nonnumeric_returns_none():
  '''Non-numeric scores return None.

  Fails if bias_variance raises on malformed scores.
  '''
  assert bias_variance(['a', 'b'], ['c', 'd']) is None


def test_bias_variance_valid_returns_figure():
  '''Valid scores return a figure.

  Fails if bias_variance returns None for valid input.
  '''
  before = set(plt.get_fignums())
  assert bias_variance(
    [1.0, 2.0, 3.0], [0.5, 0.6, 0.7]) is not None
  _assert_no_fig_leak(before)


def test_ice_plot_empty_returns_none():
  '''Empty matrix returns None.

  Fails if ice_plot plots empty data.
  '''
  from sklearn.linear_model import LogisticRegression
  model = LogisticRegression().fit(
    np.random.RandomState(0).randn(20, 2),
    (np.random.RandomState(1).randn(20) > 0).astype(int))
  assert ice_plot(model, []) is None
  assert ice_plot(model, np.array([1.0, 2.0, 3.0])) is None
  assert ice_plot(
    model, np.random.RandomState(2).randn(4, 2)) is None


def test_ice_plot_valid_returns_figure():
  '''Valid frame returns a figure.

  Fails if ice_plot returns None for valid input.
  '''
  from sklearn.linear_model import LogisticRegression
  frame = pd.DataFrame(
    np.random.RandomState(0).randn(20, 2), columns=['a', 'b'])
  labels = (frame['a'] > 0).astype(int)
  model = LogisticRegression().fit(frame, labels)
  before = set(plt.get_fignums())
  assert ice_plot(model, frame) is not None
  _assert_no_fig_leak(before)


def test_model_comparison_box_empty_returns_none():
  '''Empty and None cv_scores return None.

  Fails if model_comparison_box plots nothing.
  '''
  assert model_comparison_box({}) is None
  assert model_comparison_box(None) is None
  assert model_comparison_box({'a': [1.0]}) is None


def test_model_comparison_box_valid_returns_figure():
  '''Valid cv_scores return a figure.

  Fails if model_comparison_box returns None for valid input.
  '''
  before = set(plt.get_fignums())
  assert model_comparison_box(
    {'a': [1.0, 2.0], 'b': [1.5, 2.5]}) is not None
  _assert_no_fig_leak(before)


def test_pr_overlay_single_model_returns_none():
  '''Fewer than two models return None.

  Fails if pr_overlay plots a single model.
  '''
  assert pr_overlay([0, 1, 0, 1], {'a': [.1, .9, .2, .8]}) is None
  assert pr_overlay([], {}) is None
  assert pr_overlay([0, 1], None) is None


def test_error_correlation_heatmap_bad_inputs_none():
  '''None preds and mismatched y return None.

  Fails if the heatmap plots without two models.
  '''
  assert error_correlation_heatmap(
    None, [1.0, 2.0, 3.0, 4.0, 5.0]) is None
  assert error_correlation_heatmap(
    {'a': [1.0, 2.0, 3.0, 4.0, 5.0],
     'b': [1.0, 2.0, 3.0, 4.0, 5.0]}, [1.0, 2.0]) is None


def test_error_correlation_heatmap_valid_returns_figure():
  '''Valid preds and y return a figure.

  Fails if the heatmap returns None for valid input.
  '''
  before = set(plt.get_fignums())
  assert error_correlation_heatmap(
    {'a': [1.0, 2.0, 3.0, 4.0, 5.0],
     'b': [1.0, 2.0, 3.0, 4.0, 5.0]},
    [1.0, 2.0, 3.0, 4.0, 5.0]) is not None
  _assert_no_fig_leak(before)


def test_interpretability_shap_empty_none():
  '''Empty SHAP inputs return None.

  Fails if shap charts plot empty input.
  '''
  assert shap_beeswarm(None) is None
  assert shap_beeswarm([]) is None
  assert shap_heatmap(None) is None
  assert shap_bar(None) is None
  assert shap_bar([]) is None
  assert shap_bar({}) is None


def test_interpretability_pdp_mismatch_none():
  '''Mismatched PDP grid/values return None.

  Fails if pdp_1d_chart raises on mismatched lengths.
  '''
  assert pdp_1d_chart(None) is None
  assert pdp_1d_chart({}) is None
  assert pdp_1d_chart(
    {'grid': [1.0, 2.0, 3.0], 'values': [1.0]}) is None


def test_interpretability_pdp_valid_returns_figure():
  '''Valid PDP result returns a figure.

  Fails if pdp_1d_chart returns None for valid input.
  '''
  before = set(plt.get_fignums())
  assert pdp_1d_chart(
    {'grid': [1.0, 2.0, 3.0],
     'values': [1.0, 2.0, 3.0]}) is not None
  _assert_no_fig_leak(before)


def test_interpretability_permutation_empty_none():
  '''Empty permutation importance returns None.

  Fails if the bar chart plots empty input.
  '''
  assert permutation_importance_bar(None) is None
  assert permutation_importance_bar([]) is None


def test_interpretability_diagnostics_empty_none():
  '''Empty diagnostics inputs return None.

  Fails if diagnostics charts raise on empty input.
  '''
  assert calibration_curve(None) is None
  assert residual_vs_fitted(None, None) is None
  assert residual_vs_fitted([1.0, 2.0], [1.0]) is None
  assert error_by_segment_bar(None) is None
  assert error_by_segment_bar([]) is None
  assert prediction_distribution_by_class(None, None) is None
  assert confidence_histogram(None) is None
  assert confidence_histogram([]) is None


def test_interpretability_valid_returns_figure():
  '''Valid diagnostics inputs return a figure.

  Fails if diagnostics charts return None for valid input.
  '''
  before = set(plt.get_fignums())
  assert residual_vs_fitted(
    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]) is not None
  _assert_no_fig_leak(before)


def test_charts_do_not_raise_on_garbage():
  '''Garbage inputs return None instead of raising.

  Fails if any chart raises on malformed input.
  '''
  assert bias_variance(None, None) is None
  assert model_comparison_box('not-a-dict') is None
  assert pr_overlay('x', 'y') is None
