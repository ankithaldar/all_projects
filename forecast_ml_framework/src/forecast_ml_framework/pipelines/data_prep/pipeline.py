#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Kedro pipeline wiring for data preparation.

TODO parameters.yml naming (documented for catalog wiring):
- ``params:data_prep_params`` -> preprocess prep_params
- ``params:validation_params`` -> validate_data params
- ``params:feature_selection_params`` -> select_features params
- ``params:pandas_conversion_params`` -> Spark conversion
- ``variables`` (catalog) -> variables dict
- ``train_data`` (catalog) -> validate_data df input
- ``validated_data`` / ``validation_report`` outputs
- ``preprocessed_data`` output of preprocess
- ``train_data_selected`` / ``selected_features`` /
  ``selection_report`` outputs of select_features
'''

from kedro.pipeline import Pipeline, node

from forecast_ml_framework.pipelines.data_prep.nodes import (
  preprocess,
  select_features,
  validate_data,
)


def create_pipeline(**kwargs):
  '''Create Kedro pipeline with ingestion, prep, selection.

  Args:
    **kwargs: Forwarded to Pipeline (e.g. namespace).

  Returns:
    Kedro Pipeline with at least three nodes.
  '''
  validate_node = node(
    func=validate_data,
    inputs=['train_data', 'params:validation_params', 'variables'],
    outputs=['validated_data', 'validation_report'],
    name='data_ingestion_validate',
  )
  preprocess_node = node(
    func=preprocess,
    inputs=['validated_data', 'params:data_prep_params',
            'variables', 'params:pandas_conversion_params'],
    outputs='preprocessed_data',
    name='preprocessing',
  )
  select_node = node(
    func=select_features,
    inputs=['preprocessed_data',
            'params:feature_selection_params', 'variables'],
    outputs=['train_data_selected', 'selected_features',
             'selection_report'],
    name='feature_selection',
  )
  return Pipeline([validate_node, preprocess_node, select_node],
                  **kwargs)
