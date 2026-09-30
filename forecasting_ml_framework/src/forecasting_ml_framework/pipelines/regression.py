#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Regression pipeline factories.

The regression topology mirrors the classification one with three differences,
and each is a consequence of the task type rather than a separate design:

1. **No calibration dimension.** A continuous target has no second score family,
   so there is no fan-out — and therefore no sequencing tokens. That is the
   clearest demonstration in the codebase that the tokens exist to work around a
   consequence of the dual-score design, not because parallelism control was ever
   the need.
2. **No lift node.** There is no positive class to lift over, so the decile error
   profile emitted by the evaluation node is the analogue.
3. **A different training-mode vocabulary**, so the promotion gate's node is
   selected on its own token.

The factories delegate to the classification ones and substitute the node
functions, so a change to the graph's *structure* -- the conditional load node,
the promotion branch -- is made once and the regression graph inherits it.

Regression shares the metric vocabulary, the metric spellings and the
promotion-direction logic with classification rather than restating them, so a
metric that is higher-is-better in one tier is higher-is-better in the other by
construction. All of it lives in
:mod:`forecasting_ml_framework.modeling.metrics`, which defines each metric
exactly once -- and defining a metric's direction in one place rather than
inferring it from its sign is what keeps a loss from being gated on the wrong
direction.
"""

from __future__ import annotations

from kedro.pipeline import Pipeline, node

from forecasting_ml_framework.nodes import feature_nodes, model_nodes, validation_nodes
from forecasting_ml_framework.pipelines.classification import (
  create_fe_scoring_pipeline as _classification_fe_scoring,
)
from forecasting_ml_framework.pipelines.classification import (
  create_fe_training_pipeline as _classification_fe_training,
)
from forecasting_ml_framework.pipelines.classification import (
  dataset_stem,
)
from forecasting_ml_framework.pipelines.registry import PipelineTopology

#: The regression training-mode token that selects the comparison gate. It is
#: distinct from the classification token so the two vocabularies cannot collide.
REGRESSION_COMPARE_MODE = 'train_compare_update'


def create_fe_training_pipeline(topology: PipelineTopology) -> Pipeline:
  """Build the feature-engineering pipeline for regression training.

  The graph is identical to the classification one; only the sampling node
  differs, because a continuous target has no class weights to compute. The
  factory therefore delegates the structural work and substitutes the node,
  rather than re-declaring the whole chain and risking drift.

  Args:
    topology: The resolved graph topology.

  Returns:
    The pipeline.
  """
  suffix = topology.suffix
  template = _classification_fe_training(topology)
  frame_in = f'train_df_{suffix}' if topology.split else f'df_{suffix}'
  metadata_in = f'train_df_metadata_{suffix}' if topology.split else f'df_metadata_{suffix}'

  nodes: list = []
  for existing in template.nodes:
    if existing.name == f'sampling_{suffix}':
      nodes.append(
        node(
          func=feature_nodes.sampling_node_regression,
          inputs=[frame_in, metadata_in, 'params:sampling_reg_params'],
          outputs=list(existing.outputs),
          name=existing.name,
        )
      )
    else:
      nodes.append(existing)
  return Pipeline(nodes)


def create_model_training_pipeline(topology: PipelineTopology) -> Pipeline:
  """Build the modelling pipeline for regression training.

  Args:
    topology: The resolved graph topology.

  Returns:
    The pipeline.
  """
  suffix = topology.suffix
  compare = 'compare' in topology.train_mode.lower()

  nodes: list = [
    node(
      func=model_nodes.spark_to_pandas,
      inputs=[f'train_pp_data_df_{suffix}', f'train_pp_metadata_{suffix}', 'params:pandas_conversion_params'],
      outputs=[f'train_pp_data_pdf_{suffix}', f'train_pp_metadata_pdf_{suffix}'],
      name=f'train_spark_to_pandas_{suffix}',
    ),
    node(
      func=model_nodes.model_training_regression,
      inputs=[
        'parameters',
        f'train_pp_data_pdf_{suffix}',
        f'train_pp_metadata_pdf_{suffix}',
        'params:global_params.model_type',
      ],
      outputs=[
        f'model_handle_candidate_{suffix}',
        f'fitted_model_candidate_{suffix}',
        f'calibrated_model_candidate_{suffix}',
        f'training_data_{suffix}',
        f'validation_data_{suffix}',
      ],
      name=f'model_training_{suffix}',
    ),
  ]

  # Exactly one node writes the production-named trio, under a name distinct from
  # its inputs -- Kedro rejects a node that reads and writes the same dataset.
  # See the equivalent note in the classification factory.
  promotion = (
    validation_nodes.validate_and_promote_regression if compare
    else validation_nodes.promote_without_comparison
  )
  nodes.append(
    node(
      func=promotion,
      inputs=[
        f'model_handle_candidate_{suffix}',
        f'fitted_model_candidate_{suffix}',
        f'calibrated_model_candidate_{suffix}',
        f'validation_data_{suffix}',
        'parameters',
        *( [f'model_handle_previous_{suffix}', f'fitted_model_previous_{suffix}',
            f'calibrated_model_previous_{suffix}'] if compare else [] ),
      ],
      outputs=[f'model_handle_{suffix}', f'fitted_model_{suffix}', f'calibrated_model_{suffix}'],
      name=(f'model_validation_{suffix}' if compare else f'model_promotion_{suffix}'),
    )
  )

  model_triple = [f'model_handle_{suffix}', f'fitted_model_{suffix}', f'calibrated_model_{suffix}']
  for slice_label, data_dataset in (
    ('train', f'training_data_{suffix}'),
    ('valid', f'validation_data_{suffix}'),
  ):
    prediction_frame = f'{slice_label}_prediction_df_{suffix}'
    # The regression graph carries no sequencing tokens: the token exists to order
    # the classification fan-out, and a continuous target has no calibration
    # dimension, so there is no fan-out to order. The token datasets were still
    # declared as outputs, which both wrote artefacts nothing consumed and made the
    # node's declared arity disagree with its return value.
    #
    # The metric and lift outputs carry the `_reg_` infix, matching the catalog's
    # dedicated regression tables. Without it the names collided with the
    # classification entries, so a regression run wrote MAE/RMSE/R2 rows into the
    # *classification* train-metrics table under the same model key -- truncating
    # that model's classification snapshot on every run -- while the six
    # `*_reg_*` tables the catalog defines stayed permanently empty.
    nodes.extend(
      [
        node(
          func=model_nodes.model_scoring_regression,
          inputs=[*model_triple, data_dataset, f'train_pp_metadata_pdf_{suffix}', f'params:{slice_label}'],
          outputs=[prediction_frame, f'{slice_label}_calib_prediction_df_{suffix}'],
          name=f'{slice_label}_model_scoring_{suffix}',
        ),
        node(
          func=model_nodes.model_evaluation_regression,
          inputs=[
            prediction_frame,
            'parameters',
            f'train_pp_metadata_pdf_{suffix}',
            'params:modeling_reg_params.prediction_col',
            f'params:{slice_label}',
          ],
          outputs=[f'{slice_label}_reg_model_evaluation_{suffix}'],
          name=f'{slice_label}_model_evaluation_{suffix}',
        ),
        node(
          func=model_nodes.lift_calculation_regression,
          inputs=[
            prediction_frame,
            'parameters',
            f'train_pp_metadata_pdf_{suffix}',
            'params:modeling_reg_params.prediction_col',
            f'params:{slice_label}',
          ],
          outputs=[f'{slice_label}_reg_lift_df_{suffix}'],
          name=f'{slice_label}_lift_calculation_{suffix}',
        ),
      ]
    )
  return Pipeline(nodes)


def create_fe_scoring_pipeline(topology: PipelineTopology) -> Pipeline:
  """Build the feature-engineering pipeline for regression scoring.

  The graph is identical to the classification one; the preprocessed scoreset is
  consumed by a different trainer, not by a different feature pipeline.

  Args:
    topology: The resolved graph topology.

  Returns:
    The pipeline.
  """
  return _classification_fe_scoring(topology)


def create_model_scoring_pipeline(topology: PipelineTopology) -> Pipeline:
  """Build the modelling pipeline for regression scoring.

  Args:
    topology: The resolved graph topology.

  Returns:
    The pipeline.
  """
  suffix = topology.suffix
  nodes: list = [
    node(
      func=model_nodes.spark_to_pandas,
      inputs=[f'scoreset_data_{suffix}', f'score_pp_metadata_{suffix}', 'params:pandas_conversion_params'],
      outputs=[f'score_pp_data_pdf_{suffix}', f'score_pp_metadata_pdf_{suffix}'],
      name=f'test_spark_to_pandas_{suffix}',
    ),
    node(
      func=model_nodes.model_scoring_regression,
      inputs=[
        f'model_handle_{suffix}',
        f'fitted_model_{suffix}',
        f'calibrated_model_{suffix}',
        f'score_pp_data_pdf_{suffix}',
        f'score_pp_metadata_pdf_{suffix}',
        'params:test',
      ],
      outputs=[f'prediction_df_{suffix}', f'calib_prediction_df_{suffix}'],
      name=f'model_scoring_{suffix}',
    ),
  ]
  # `spark_to_pandas` returns the pair (frame, contract). The scoring node
  # previously declared one output, which Kedro rejects at run time because the
  # node returned two values against one declared name. Both are declared, exactly
  # as on the training path, and the converted contract is what the scoring node
  # consumes so the two stages agree on which contract they are working with.

  if topology.is_evaluation:
    # The two score families are evaluated in parallel from a common parent. No
    # sequencing token is needed: there is no fan-out to order, because each
    # evaluation reads a *different* prediction frame rather than sharing one, so
    # the two nodes are already unordered and independent.
    for slice_label, frame in (('test', f'prediction_df_{suffix}'), ('calib_test', f'calib_prediction_df_{suffix}')):
      stem = dataset_stem(slice_label)
      nodes.extend(
        [
          node(
            func=model_nodes.model_evaluation_regression,
            inputs=[
              frame,
              'parameters',
              f'score_pp_metadata_pdf_{suffix}',
              'params:modeling_reg_params.prediction_col',
              f'params:{slice_label}',
            ],
            outputs=[f'{stem}_reg_model_evaluation_{suffix}'],
            name=f'{slice_label}_model_evaluation_{suffix}',
          ),
          # The band error profile is emitted on the scoring path too, matching the
          # training path. Without it the catalog's `test_reg_lift_df` entry was
          # unreachable and a scored regression run published a single flat error
          # row with no distribution behind it -- so there was no way to see *where*
          # along the predicted range the model is weak.
          node(
            func=model_nodes.lift_calculation_regression,
            inputs=[
              frame,
              'parameters',
              f'score_pp_metadata_pdf_{suffix}',
              'params:modeling_reg_params.prediction_col',
              f'params:{slice_label}',
            ],
            outputs=[f'{stem}_reg_lift_df_{suffix}'],
            name=f'{slice_label}_lift_calculation_{suffix}',
          ),
        ]
      )
  return Pipeline(nodes)
