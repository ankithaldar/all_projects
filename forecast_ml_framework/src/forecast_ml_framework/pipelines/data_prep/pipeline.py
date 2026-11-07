#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Kedro pipeline wiring for data preparation.

Split-first (A9, breaking at 0.1.0): validate ->
split raw (``train_raw`` / ``test_raw``) -> fit
preprocessing on train only -> apply preprocessing
to test -> select features on train only -> apply
the train-fitted selection to test. The old flow
that fitted preprocessing/selection on the whole
``modeling_input`` frame and split afterwards is
removed: no node consumes ``modeling_input``, and
``preprocessed_data`` now holds the train-only
preprocessed frame. Modeling consumes the
pre-split ``train_data_selected`` /
``test_data_selected`` frames with
``split_meta.sampling_method='presplit'``.

Note: ``conf/base/parameters/*.yml`` bucket naming
(documented for catalog wiring):
- ``params:data_prep_params`` -> preprocess prep_params
- ``params:validation_params`` -> validate_data params
- ``params:feature_selection_params`` -> select_features params
- ``params:pandas_conversion_params`` -> Spark conversion
- ``params:modeling_params`` -> split_raw_data_with_meta params
  (split config is owned by modeling_params, so the
  raw split and the modeling side share one source)
- ``variables`` (catalog) -> variables dict
- ``train_data`` (catalog) -> validate_data df input
- ``validated_data`` / ``validation_report`` outputs
- ``train_raw`` / ``test_raw`` / ``split_meta_raw`` outputs of
  split_raw_data_with_meta
- ``preprocessed_data`` (train-only) / ``fitted_registry``
  outputs of fit_preprocess (catalog ``fitted_registry``
  dataset, persisted for scoring replay)
- ``test_preprocessed_data`` output of apply_preprocess
- ``train_data_selected`` / ``selected_features`` /
  ``selection_report`` outputs of select_features
- ``test_data_selected`` output of apply_selection

The stateless ``preprocess`` node function is kept in
``nodes.py`` for back-compat imports, but the pipeline no
longer uses it: training must go through ``fit_preprocess``
so a fitted registry is produced from the same fit.
'''

from kedro.pipeline import Pipeline, node

from forecast_ml_framework.pipelines.data_prep.nodes import (
  apply_feature_selection,
  apply_preprocess,
  fit_preprocess,
  select_features,
  split_raw_data_with_meta,
  validate_data,
)


def create_pipeline(**kwargs):
  '''Create Kedro pipeline with ingestion, split, prep.

  Args:
    **kwargs: Forwarded to Pipeline (e.g. namespace).

  Returns:
    Kedro Pipeline with validate -> split_raw ->
    fit_preprocess -> apply_preprocess ->
    feature_selection -> apply_selection nodes.
  '''
  validate_node = node(
    func=validate_data,
    inputs=['train_data', 'params:validation_params', 'variables'],
    outputs=['validated_data', 'validation_report'],
    name='data_ingestion_validate',
  )
  split_node = node(
    func=split_raw_data_with_meta,
    inputs=['validated_data', 'params:modeling_params',
            'variables'],
    outputs=['train_raw', 'test_raw', 'split_meta_raw'],
    name='split_raw',
  )
  preprocess_node = node(
    func=fit_preprocess,
    inputs=['train_raw', 'params:data_prep_params',
            'variables', 'params:pandas_conversion_params'],
    outputs=['preprocessed_data', 'fitted_registry'],
    name='fit_preprocess',
  )
  apply_node = node(
    func=apply_preprocess,
    inputs=['test_raw', 'fitted_registry'],
    outputs=['test_preprocessed_data'],
    name='apply_preprocess',
  )
  select_node = node(
    func=select_features,
    inputs=['preprocessed_data',
            'params:feature_selection_params', 'variables'],
    outputs=['train_data_selected', 'selected_features',
             'selection_report'],
    name='feature_selection',
  )
  apply_sel_node = node(
    func=apply_feature_selection,
    inputs=['test_preprocessed_data', 'selected_features',
            'variables'],
    outputs=['test_data_selected'],
    name='apply_selection',
  )
  return Pipeline([validate_node, split_node, preprocess_node,
                   apply_node, select_node, apply_sel_node],
                  **kwargs)
