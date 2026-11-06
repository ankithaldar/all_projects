#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Tests for the interpretability pipeline (Phase 3).'''

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from forecast_ml_framework.pipelines.interpretability.nodes import (
  diagnostics,
  explain_model,
  interpret_features,
  lime_explain,
  pdp_1d,
  permutation_importance,
  run_diagnostics,
  shap_explain,
)
from forecast_ml_framework.pipelines.interpretability.pipeline import (
  create_pipeline,
)


def _data(n=60, seed=42):
  '''Return synthetic regression data + fitted forest.'''
  rng = np.random.RandomState(seed)
  X = pd.DataFrame({
    'a': rng.normal(size=n),
    'b': rng.normal(size=n),
    'c': rng.normal(size=n),
  })
  y = 3.0 * X['a'] - 2.0 * X['b'] + rng.normal(scale=0.1, size=n)
  model = RandomForestRegressor(
    n_estimators=10, random_state=seed).fit(X, y)
  return model, X, y


def _params(**overrides):
  '''Return interpretability params for tests.'''
  base = {
    'max_rows': 5,
    'top_k': 3,
    'n_repeats': 2,
    'random_state': 42,
    'n_grid': 5,
    'n_calibration_bins': 3,
    'pdp_features': ['a', 'b'],
  }
  base.update(overrides)
  return base


def test_shap_explain_sorted_global_importance():
  '''Global importance is sorted descending by importance.'''
  model, X, _y = _data()
  out = shap_explain(model, X, _params())
  assert out['global_importance']
  values = [e['importance'] for e in out['global_importance']]
  assert values == sorted(values, reverse=True)
  assert {e['feature'] for e in out['global_importance']} == {
    'a', 'b', 'c'}
  assert 'a' in {e['feature'] for e in out['global_importance'][:1]}


def test_shap_explain_per_row_capped_and_baseline():
  '''Per-row SHAPs are capped at max_rows; baseline is scalar.'''
  model, X, _y = _data()
  out = shap_explain(model, X, _params(max_rows=4))
  assert len(out['per_row']) <= 4
  assert out['per_row']
  assert set(out['per_row'][0]['shap_values']) == {'a', 'b', 'c'}
  assert isinstance(float(out['baseline']), float)


def test_permutation_importance_sorted():
  '''Permutation importance ranks a first on synthetic data.'''
  model, X, y = _data()
  out = permutation_importance(model, X, y, _params())
  assert isinstance(out, list) and len(out) == 3
  values = [e['importance'] for e in out]
  assert values == sorted(values, reverse=True)
  assert out[0]['feature'] == 'a'


def test_pdp_1d_grid_and_values():
  '''PDP returns percentile grid and mean predictions.'''
  model, X, _y = _data()
  out = pdp_1d(model, X, 'a', _params(n_grid=5))
  assert len(out['grid']) == len(out['values']) == 5
  assert out['grid'] == sorted(out['grid'])
  assert all(np.isfinite(v) for v in out['values'])
  # a has positive effect: later grid values predict higher.
  assert out['values'][-1] > out['values'][0]


def test_pdp_1d_unknown_feature_warns_not_crashes():
  '''Unknown PDP feature returns empty grid/values + warning.'''
  model, X, _y = _data()
  out = pdp_1d(model, X, 'nope', _params())
  assert out['grid'] == [] and out['values'] == []
  assert out['warnings']


def test_diagnostics_structure():
  '''Diagnostics dict has calibration/residual/segment/summary.'''
  model, X, y = _data()
  out = diagnostics(model, X, y, _params())
  assert out['calibration_bins']
  assert out['prediction_summary']['count'] == len(X)
  assert out['residual_stats']['mae'] >= 0
  assert out['error_by_segment']
  for seg in out['error_by_segment']:
    assert {'segment', 'count', 'mae'} <= set(seg)


def test_diagnostics_segment_col():
  '''error_by_segment groups by params segment_col when present.'''
  model, X, y = _data()
  X = X.copy()
  X['grp'] = np.array(['x', 'y'] * (len(X) // 2))
  out = diagnostics(model, X, y, _params(segment_col='grp'))
  assert {s['segment'] for s in out['error_by_segment']} == {'x', 'y'}


def test_lime_explain_fallback_structure():
  '''LIME node returns per-row top features (fallback documented).'''
  model, X, _y = _data()
  out = lime_explain(model, X, _params(top_k=2))
  assert out['method'] in ('lime', 'permutation_surrogate')
  assert out['rows']
  assert out['rows'][0]['top_features']
  assert len(out['rows'][0]['top_features']) <= 2


def test_empty_data_warns_not_crashes():
  '''Empty X yields warnings and empty structures, no crash.'''
  model, X, y = _data()
  empty = X.iloc[0:0]
  for fn in (lambda: shap_explain(model, empty, _params()),
             lambda: permutation_importance(model, empty, y[:0], _params()),
             lambda: pdp_1d(model, empty, 'a', _params()),
             lambda: diagnostics(model, empty, y[:0], _params()),
             lambda: lime_explain(model, empty, _params())):
    out = fn()
    if isinstance(out, list):
      assert out == []
    else:
      assert out.get('warnings') or out.get('rows') == [] \
        or out.get('grid') == [] or out.get('global_importance') == []


def test_tiny_data_not_crashes():
  '''One-row X does not crash any node.'''
  model, X, y = _data()
  tiny, tiny_y = X.iloc[:1], y[:1]
  assert shap_explain(model, tiny, _params())['global_importance']
  assert permutation_importance(model, tiny, tiny_y, _params())
  assert pdp_1d(model, tiny, 'a', _params())['grid']
  assert diagnostics(model, tiny, tiny_y, _params())
  assert lime_explain(model, tiny, _params())['rows']


def test_pipeline_nodes_and_wrappers():
  '''Pipeline has >=3 nodes; wrapper nodes produce dicts.'''
  model, X, y = _data()
  pipe = create_pipeline()
  assert len(pipe.nodes) >= 3
  names = {n.name for n in pipe.nodes}
  assert {'explain_model', 'interpret_features',
          'run_diagnostics'} <= names
  explained = explain_model(model, X, y, _params())
  assert explained['shap']['global_importance']
  assert explained['permutation']
  interpreted = interpret_features(model, X, _params())
  assert interpreted['pdp']['a']['grid']
  assert interpreted['lime']['rows']
  report = run_diagnostics(model, X, y, _params())
  assert report['residual_stats']['mae'] >= 0


# --- Charts (Phase 3 follow-up) ---

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

from forecast_ml_framework.pipelines.interpretability.charts import (
  calibration_curve,
  confidence_histogram,
  cooks_distance,
  cv_box,
  decision_boundary_chart,
  error_by_segment_bar,
  learning_curve,
  lime_bar,
  loss_vs_epoch,
  overfitting_gap,
  pdp_1d_chart,
  pdp_2d_heatmap,
  permutation_importance_bar,
  prediction_distribution_by_class,
  rec_curve,
  residual_hist_qq,
  residual_vs_fitted,
  residuals_over_time_chart,
  scale_location,
  shap_bar,
  shap_beeswarm,
  shap_dependence,
  shap_force,
  shap_heatmap,
  shap_waterfall,
  tree_plot,
  validation_curve,
)
from forecast_ml_framework.pipelines.interpretability.nodes import (
  make_diagnostics_charts,
  make_interpretability_charts,
)


def _shap_matrix(model, X):
  '''Return SHAP matrix + names from shap_explain output.'''
  out = shap_explain(model, X, _params(max_rows=20))
  names = [e['feature'] for e in out['global_importance']]
  rows = out['per_row']
  matrix = np.array([[r['shap_values'][n] for n in names]
                     for r in rows])
  return matrix, names, out


def _check_fig(fig):
  '''Assert figure returned and no figure slot leaked.'''
  assert fig is not None
  plt.close(fig)


def test_shap_charts_beeswarm_bar_heatmap():
  '''Beeswarm, bar, and heatmap return figures on SHAP data.'''
  model, X, _y = _data()
  matrix, names, out = _shap_matrix(model, X)
  before = set(plt.get_fignums())
  _check_fig(shap_beeswarm(matrix, names))
  _check_fig(shap_bar(out['global_importance']))
  _check_fig(shap_heatmap(out['per_row'], names))
  assert set(plt.get_fignums()) == before


def test_shap_charts_dependence_waterfall_force():
  '''Dependence, waterfall, and force return figures.'''
  model, X, _y = _data()
  matrix, names, out = _shap_matrix(model, X)
  before = set(plt.get_fignums())
  _check_fig(shap_dependence(matrix, X.to_numpy(), 0))
  _check_fig(shap_waterfall(matrix[0], names))
  _check_fig(shap_force(matrix[0], names))
  assert set(plt.get_fignums()) == before


def test_pdp_charts_1d_ice_and_2d():
  '''PDP 1D (+ICE overlay) and PDP 2D heatmap return figures.'''
  model, X, _y = _data()
  pdp = pdp_1d(model, X, 'a', _params(n_grid=5))
  ice = [list(np.array(pdp['values']) + s) for s in (-0.5, 0.5)]
  before = set(plt.get_fignums())
  _check_fig(pdp_1d_chart(pdp, ice_curves=ice))
  _check_fig(pdp_2d_heatmap(model, X, 'a', 'b'))
  assert set(plt.get_fignums()) == before


def test_importance_lime_boundary_tree_charts():
  '''Permutation/LIME bars, boundary, and tree return figures.'''
  model, X, y = _data()
  perm = permutation_importance(model, X, y, _params())
  lime = lime_explain(model, X, _params())
  tree = DecisionTreeRegressor(max_depth=3, random_state=42).fit(X, y)
  before = set(plt.get_fignums())
  _check_fig(permutation_importance_bar(perm))
  _check_fig(lime_bar(lime['rows']))
  _check_fig(decision_boundary_chart(X, y, model))
  _check_fig(tree_plot(tree, list(X.columns)))
  _check_fig(tree_plot(model, list(X.columns)))  # RF estimators_[0]
  assert set(plt.get_fignums()) == before


def test_diagnostics_charts_all():
  '''All diagnostics charts return figures on synthetic data.'''
  model, X, y = _data()
  diag = diagnostics(model, X, y, _params())
  preds = model.predict(X)
  proba = np.column_stack([1 - (preds - preds.min()) /
                           (np.ptp(preds) or 1), (preds - preds.min()) /
                           (np.ptp(preds) or 1)])
  train_scores = [0.7, 0.8, 0.85, 0.88]
  val_scores = [0.6, 0.72, 0.78, 0.8]
  before = set(plt.get_fignums())
  _check_fig(calibration_curve(diag['calibration_bins']))
  _check_fig(residual_vs_fitted(y, preds))
  _check_fig(residual_hist_qq(y, preds))
  _check_fig(scale_location(y, preds))
  _check_fig(error_by_segment_bar(diag['error_by_segment']))
  _check_fig(prediction_distribution_by_class(
    (y > np.median(y)).astype(int), preds))
  _check_fig(confidence_histogram(proba))
  _check_fig(residuals_over_time_chart(y, preds))
  _check_fig(cooks_distance(y, preds, X))
  _check_fig(rec_curve(y, preds))
  _check_fig(learning_curve(train_scores, val_scores))
  _check_fig(validation_curve([1, 2, 4, 8], train_scores, val_scores))
  _check_fig(loss_vs_epoch([1.0, 0.6, 0.4, 0.3], [1.1, 0.7, 0.5, 0.4]))
  _check_fig(cv_box({'a': train_scores, 'b': val_scores}))
  _check_fig(overfitting_gap(train_scores, val_scores))
  assert set(plt.get_fignums()) == before


def test_charts_none_on_tiny_data():
  '''Every chart returns None (not raise) on tiny data.'''
  model, X, y = _data()
  tiny_X, tiny_y = X.iloc[:1], y[:1]
  assert shap_beeswarm(np.empty((0, 3))) is None
  assert shap_bar([]) is None
  assert shap_heatmap([]) is None
  assert shap_dependence(np.empty((0, 3))) is None
  assert shap_waterfall([]) is None
  assert shap_force([]) is None
  assert pdp_1d_chart({'grid': [], 'values': []}) is None
  assert pdp_2d_heatmap(model, tiny_X, 'a', 'b') is None
  assert permutation_importance_bar([]) is None
  assert lime_bar([]) is None
  assert decision_boundary_chart(tiny_X, tiny_y, model) is None
  assert tree_plot(object()) is None
  assert calibration_curve([]) is None
  assert residual_vs_fitted(tiny_y, tiny_y) is None
  assert residual_hist_qq(tiny_y, tiny_y) is None
  assert scale_location(tiny_y, tiny_y) is None
  assert error_by_segment_bar([]) is None
  assert prediction_distribution_by_class([], []) is None
  assert confidence_histogram(np.empty((0, 2))) is None
  assert residuals_over_time_chart(tiny_y, tiny_y) is None
  assert cooks_distance(tiny_y, tiny_y) is None
  assert rec_curve(tiny_y, tiny_y) is None
  assert learning_curve([1], [1]) is None
  assert validation_curve([1], [1], [1]) is None
  assert loss_vs_epoch([1]) is None
  assert cv_box({'a': [1]}) is None
  assert overfitting_gap([1], [1]) is None


def test_make_charts_nodes_and_pipeline():
  '''Chart nodes return figure dicts; pipeline has chart nodes.'''
  model, X, y = _data()
  explained = explain_model(model, X, y, _params(max_rows=20))
  interpreted = interpret_features(model, X, _params())
  combined = dict(explained, pdp=interpreted['pdp'],
                  lime=interpreted['lime'])
  before = set(plt.get_fignums())
  figs = make_interpretability_charts(combined, _params())
  assert figs and all(f is not None for f in figs.values())
  assert any('shap' in name for name in figs)
  diag = diagnostics(model, X, y, _params())
  diag = dict(diag, y_true=y, y_pred=model.predict(X))
  dfigs = make_diagnostics_charts(diag, _params())
  assert dfigs and all(f is not None for f in dfigs.values())
  assert 'calibration_curve' in dfigs
  for fig in list(figs.values()) + list(dfigs.values()):
    plt.close(fig)
  assert set(plt.get_fignums()) == before
  names = {n.name for n in create_pipeline().nodes}
  assert {'make_interpretability_charts',
          'make_diagnostics_charts'} <= names


class _HostileModel:
  '''Model whose predict always raises.'''

  def predict(self, X):
    raise RuntimeError('hostile model')


def test_shap_explain_hostile_model_returns_none_not_crash():
  '''Hostile model does not crash shap_explain; warns + none.'''
  _model, X, _y = _data()
  out = shap_explain(_HostileModel(), X, _params())
  assert out['method'] == 'none'
  assert out['global_importance'] == []
  assert out['per_row'] == []
  assert out['warnings']


def test_permutation_importance_hostile_model_returns_empty(caplog):
  '''Hostile model does not crash permutation_importance.'''
  import logging
  _model, X, y = _data()
  with caplog.at_level(logging.WARNING):
    assert permutation_importance(_HostileModel(), X, y, _params()) == []
    assert permutation_importance(_HostileModel(), X, None, _params()) == []
  assert caplog.records


def test_lime_explain_hostile_model_returns_none():
  '''Hostile model does not crash lime_explain; warns + none.'''
  _model, X, _y = _data()
  out = lime_explain(_HostileModel(), X, _params())
  assert out['method'] == 'none'
  assert out['rows'] == []
  assert out['warnings']


def test_pdp_1d_hostile_model_returns_empty():
  '''Hostile model does not crash pdp_1d; warns + empty.'''
  _model, X, _y = _data()
  out = pdp_1d(_HostileModel(), X, 'a', _params())
  assert out['grid'] == []
  assert out['values'] == []
  assert out['warnings']


def test_diagnostics_hostile_model_returns_empty():
  '''Hostile model does not crash diagnostics; warns + empty.'''
  _model, X, y = _data()
  out = diagnostics(_HostileModel(), X, y, _params())
  assert out['calibration_bins'] == []
  assert out['residual_stats'] == {}
  assert out['error_by_segment'] == []
  assert out['prediction_summary'] == {}
  assert out['warnings']


def test_shap_explain_scalar_string_dict_x_returns_none():
  '''Scalar/string/dict X returns method none + warning, no crash.'''
  model, _X, _y = _data()
  for bad in (5, 3.14, 'not-a-frame', {'a': [1, 2]}, object()):
    out = shap_explain(model, bad, _params())
    assert out['method'] == 'none', bad
    assert out['warnings'], bad


def test_make_charts_malformed_does_not_crash():
  '''Malformed chart inputs are skipped per-chart, not raise.'''
  bad_interp = {
    'shap': {
      'global_importance': [{'importance': 1.0}, 'bad', None],
      'per_row': [{'shap_values': {'a': 'bad'}}, 'bad', None],
    },
    'permutation': [{'importance': 1.0}, 'bad'],
    'pdp': {'a': {'grid': 'bad'}},
    'lime': {'rows': 'bad'},
  }
  figs = make_interpretability_charts(bad_interp, _params())
  assert isinstance(figs, dict)
  for fig in figs.values():
    plt.close(fig)
  # Non-dict top-level inputs must not raise either.
  assert make_interpretability_charts('bad', _params()) == {}
  assert make_interpretability_charts(None, _params()) == {}
  assert make_diagnostics_charts('bad', _params()) == {}
  assert make_diagnostics_charts(None, _params()) == {}
  bad_diag = {
    'error_by_segment': [{'count': 1}, 'bad'],
    'calibration_bins': 'bad',
    'y_true': [1, 2],
    'y_pred': [1, 2],
  }
  dfigs = make_diagnostics_charts(bad_diag, _params())
  assert isinstance(dfigs, dict)
  for fig in dfigs.values():
    plt.close(fig)


# --- Cycle-2 dedupe: canonical params + direct chart imports ---

def test_flatten_params_canonical():
  '''core.params.flatten_params unwraps nested key, top wins.'''
  from forecast_ml_framework.core.params import flatten_params
  assert flatten_params(None, 'eda_params') == {}
  assert flatten_params({'a': 1}, 'eda_params') == {'a': 1}
  assert flatten_params(
      {'eda_params': {'a': 1, 'b': 2}, 'b': 3},
      'eda_params') == {'a': 1, 'b': 3}
  assert flatten_params(
      {'monitoring_params': {'bins': 5}}, 'monitoring_params') == {
      'bins': 5}
  import pytest
  with pytest.raises(ValueError, match='params must be a dict'):
    flatten_params('bad', 'eda_params')


def test_interpretability_charts_import_duplicates_directly():
  '''Duplicated charts are modeling.charts objects, no shim.'''
  from forecast_ml_framework.pipelines.interpretability import (
    charts as interp_charts,
  )
  from forecast_ml_framework.pipelines.modeling import (
    charts as modeling_charts,
  )
  assert not hasattr(interp_charts, '_delegate')
  assert interp_charts.shap_dependence is \
    modeling_charts.shap_dependence
  assert interp_charts.shap_waterfall is \
    modeling_charts.shap_waterfall
  assert interp_charts.shap_force is modeling_charts.shap_force
  assert interp_charts.cooks_distance is \
    modeling_charts.cooks_distance
  assert interp_charts.rec_curve is modeling_charts.rec_curve
  assert interp_charts.learning_curve is \
    modeling_charts.learning_curve
  assert interp_charts.validation_curve is \
    modeling_charts.validation_curve
  assert interp_charts.loss_vs_epoch is modeling_charts.loss_vs_epoch
  assert interp_charts.overfitting_gap is \
    modeling_charts.overfitting_gap
  assert interp_charts.cv_box is modeling_charts.cv_scores_box
  assert interp_charts.decision_boundary_chart is \
    modeling_charts.decision_boundary
  assert interp_charts.residuals_over_time_chart is \
    modeling_charts.residuals_over_time
  # Unique interpretability charts remain.
  for name in ('shap_beeswarm', 'shap_bar', 'shap_heatmap',
               'pdp_1d_chart', 'pdp_2d_heatmap',
               'permutation_importance_bar', 'calibration_curve',
               'lime_bar', 'tree_plot'):
    assert callable(getattr(interp_charts, name)), name


def test_no_local_params_dict_copies():
  '''Nodes import flatten_params, no local _params_dict defs.'''
  import inspect
  from forecast_ml_framework.pipelines.eda import nodes as eda_nodes
  from forecast_ml_framework.pipelines.monitoring import (
    nodes as mon_nodes,
  )
  from forecast_ml_framework.pipelines.interpretability import (
    nodes as interp_nodes,
  )
  for mod in (eda_nodes, mon_nodes, interp_nodes):
    assert not hasattr(mod, '_params_dict'), mod.__name__
    assert hasattr(mod, 'flatten_params'), mod.__name__


# --- loop7 chart helpers dedupe (RED) ---

def test_interpretability_chart_helpers_are_canonical():
  '''_safe_fig/_new_fig/_pairs are modeling.charts objects.'''
  from forecast_ml_framework.pipelines.interpretability import (
    charts as interp_charts,
  )
  from forecast_ml_framework.pipelines.modeling import (
    charts as modeling_charts,
  )
  assert interp_charts._safe_fig is modeling_charts._safe_fig
  assert interp_charts._new_fig is modeling_charts._new_fig
  assert interp_charts._pairs is modeling_charts._pairs
