#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Kedro pipeline wiring for interpretability.

Final parameters.yml naming (documented for catalog wiring):
- ``params:interpretability_params`` -> dict with keys
  ``max_rows``, ``top_k``, ``n_repeats``, ``random_state``
  (alias ``seed``), ``n_grid``, ``pdp_features``,
  ``n_calibration_bins``, ``segment_col``, ``scorer``
  (see ``pipelines.interpretability.nodes`` docstring)
- ``model`` (catalog) -> fitted model with ``predict``
- ``X`` / ``y`` (catalog) -> feature frame/array + target
- ``explanations`` output of explain_model (SHAP +
  permutation)
- ``interpretation`` output of interpret_features
  (PDP per feature + LIME rows)
- ``diagnostics_report`` output of run_diagnostics
- ``interpretability_charts`` output of
  make_interpretability_charts (name -> Figure)
- ``diagnostics_charts`` output of
  make_diagnostics_charts (name -> Figure)
'''

from kedro.pipeline import Pipeline, node

from forecast_ml_framework.pipelines.interpretability.nodes import (
  explain_model,
  interpret_features,
  make_diagnostics_charts,
  make_interpretability_charts,
  run_diagnostics,
)


def create_pipeline(**kwargs):
  '''Create Kedro pipeline with three interpretability nodes.

  Args:
    **kwargs: Forwarded to Pipeline (e.g. namespace).

  Returns:
    Kedro Pipeline with nodes: explain_model ->
    interpret_features -> run_diagnostics.
  '''
  explain_node = node(
    func=explain_model,
    inputs=['model', 'X', 'y', 'params:interpretability_params'],
    outputs='explanations',
    name='explain_model',
  )
  interpret_node = node(
    func=interpret_features,
    inputs=['model', 'X', 'params:interpretability_params'],
    outputs='interpretation',
    name='interpret_features',
  )
  diagnostics_node = node(
    func=run_diagnostics,
    inputs=['model', 'X', 'y', 'params:interpretability_params'],
    outputs='diagnostics_report',
    name='run_diagnostics',
  )
  interp_charts_node = node(
    func=make_interpretability_charts,
    inputs=['explanations', 'params:interpretability_params'],
    outputs='interpretability_charts',
    name='make_interpretability_charts',
  )
  diag_charts_node = node(
    func=make_diagnostics_charts,
    inputs=['diagnostics_report', 'params:interpretability_params'],
    outputs='diagnostics_charts',
    name='make_diagnostics_charts',
  )
  return Pipeline([explain_node, interpret_node, diagnostics_node,
                   interp_charts_node, diag_charts_node], **kwargs)
