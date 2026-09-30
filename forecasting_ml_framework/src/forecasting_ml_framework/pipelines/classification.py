#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Classification pipeline factories.

Four factories, one per pipeline stage, all parameterised by a
:class:`~forecasting_ml_framework.pipelines.registry.PipelineTopology`.

**The two-axis conditional structure.** One axis selects the promotion mode and
determines whether the champion/challenger comparison runs at all. The other
expands the data-slice by calibration evaluation matrix, and that expansion is
what creates the fan-out the sequencing tokens exist to order.

**The sequencing-token pattern.** Two evaluation nodes consume the same
prediction frame, so the topological sort leaves them unordered and eligible for
parallel execution. A deterministic order is forced with throwaway datasets —
``sequence_ind1`` … ``sequence_ind10`` — threaded as a trailing input and output
through the chain, each node returning the token it received. This is a
**dependency-token idiom**: a synthetic edge inserted purely to constrain
execution order. Its cost is that every evaluation node declaration carries a
trailing token with no meaning, and inserting a node in the middle requires
renumbering the chain. The regression pipelines need no tokens, because they have
no calibration dimension and therefore no fan-out.

a's ``model_validation`` node *rewrote* the production
model datasets, returning the promoted trio under the production names so every
downstream node transparently scored whichever model won. That worked, but it
made the node's output contract depend on whether an incumbent existed — a
mode-dependent shape the catalog cannot describe. The promotion is resolved by the
validation node writing the winning trio to the production names in every mode,
so the graph's shape no longer varies with the outcome.
"""

from __future__ import annotations

from kedro.pipeline import Pipeline, node

from forecasting_ml_framework import constants
from forecasting_ml_framework.exceptions import PipelineTopologyError
from forecasting_ml_framework.nodes import feature_nodes, model_nodes, validation_nodes
from forecasting_ml_framework.pipelines.registry import PipelineTopology

#: The number of sequencing tokens the training pipeline consumes: two slices,
#: each a five-node block (score, evaluate, lift, evaluate-calibrated,
#: lift-calibrated). The blocks are independent, so five and ten are the two
#: dangling tokens at the end of each block.
TRAINING_SEQUENCE_TOKENS = 10
#: The number consumed by the scoring pipeline in evaluation mode: one token from
#: the scoring node, then a two-node block per slice, chained, each block ending
#: with a trailing token. The chain therefore runs 11 through 15.
SCORING_SEQUENCE_TOKENS = 15


def dataset_stem(slice_label: str) -> str:
  """Derive a dataset-name stem from a slice label.

  The slice label carries two independent facts, and they are ordered
  differently in the two vocabularies they feed. ``calib_test`` selects the
  *parameter block* ``params:calib_test`` and the calibrated *column* family, so
  the qualifier leads there. The *dataset* names put the qualifier after the base
  slice -- ``test_calib_model_evaluation_``, matching the catalog and matching
  what the training blocks already produce (``train_calib_model_evaluation_``).

  The scoring block emitted the label verbatim, so it wrote
  ``calib_test_model_evaluation_`` against a catalog that declares
  ``test_calib_model_evaluation_``. The run then failed with
  ``MissingDatasetException`` on the calibrated slice -- *after* the raw slice had
  already been appended to, so the score-metric table was left half-written.

  Deriving the stem through one function is what keeps the two vocabularies
  related by construction instead of by coincidence.

  Args:
    slice_label: A slice label such as ``'train'`` or ``'calib_test'``.

  Returns:
    The dataset-name stem, e.g. ``'test_calib'``.
  """
  if slice_label.startswith(constants.CALIBRATION_PREFIX):
    return f'{slice_label[len(constants.CALIBRATION_PREFIX):]}_calib'
  return slice_label


def create_fe_training_pipeline(topology: PipelineTopology) -> Pipeline:
  """Build the feature-engineering pipeline for training.

  ``load -> [split] -> sample -> build registry -> fit registry``

  Args:
    topology: The resolved graph topology.

  Returns:
    The pipeline.
  """
  suffix = topology.suffix
  nodes: list = [
    node(
      func=feature_nodes.load_input_data,
      inputs=['parameters', 'params:global_params', f'train_input_{suffix}'],
      outputs=[f'df_{suffix}', f'df_metadata_{suffix}'],
      name=f'load_train_{suffix}',
    )
  ]

  frame_in, metadata_in = f'df_{suffix}', f'df_metadata_{suffix}'
  if topology.split:
    nodes.append(
      node(
        func=feature_nodes.split_frame,
        inputs=[
          frame_in,
          metadata_in,
          'params:train_test_split_params.training_split_ratio',
          'params:train_test_split_params.test_split_ratio',
        ],
        outputs=[
          f'train_df_{suffix}',
          f'train_df_metadata_{suffix}',
          f'score_df_{suffix}',
          f'score_metadata_{suffix}',
        ],
        name=f'train_test_split_{suffix}',
      )
    )
    frame_in, metadata_in = f'train_df_{suffix}', f'train_df_metadata_{suffix}'

  nodes.extend(
    [
      node(
        func=feature_nodes.sampling_node,
        inputs=[frame_in, metadata_in, 'params:sampling_params'],
        outputs=[f'sampled_df_{suffix}', f'train_metadata_{suffix}'],
        name=f'sampling_{suffix}',
      ),
      node(
        func=feature_nodes.build_registry,
        inputs=['params:global_params', 'params:data_prep_params'],
        outputs=[f'registry_{suffix}'],
        name=f'get_available_registries_{suffix}',
      ),
      node(
        func=feature_nodes.fit_registry,
        inputs=[f'sampled_df_{suffix}', f'train_metadata_{suffix}', f'registry_{suffix}', 'parameters'],
        outputs=[f'fitted_registry_{suffix}', f'train_pp_data_df_{suffix}', f'train_pp_metadata_{suffix}'],
        name=f'fit_registry_{suffix}',
      ),
    ]
  )
  return Pipeline(nodes)


def create_model_training_pipeline(topology: PipelineTopology) -> Pipeline:
  """Build the modelling pipeline for training.

  Args:
    topology: The resolved graph topology.

  Returns:
    The pipeline.
  """
  suffix = topology.suffix
  compare = topology.is_compare_mode

  nodes: list = [
    node(
      func=model_nodes.spark_to_pandas,
      inputs=[f'train_pp_data_df_{suffix}', f'train_pp_metadata_{suffix}', 'params:pandas_conversion_params'],
      outputs=[f'train_pp_data_pdf_{suffix}', f'train_pp_metadata_pdf_{suffix}'],
      name=f'train_spark_to_pandas_{suffix}',
    ),
    node(
      func=model_nodes.model_training,
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

  # Exactly one node writes the production-named trio, and it always takes the
  # candidate under a *different* dataset name. Kedro rejects a node that lists
  # the same dataset as both an input and an output, so the design --
  # where model_validation rewrote the production datasets in place -- is not
  # expressible in the graph at all. Routing the candidate through a distinct
  # name makes the promotion explicit, keeps the output contract identical in
  # every mode (which is what the catalog describes), and makes it impossible for
  # a node to read and write the same artefact.
  if compare:
    nodes.append(
      node(
        func=validation_nodes.validate_and_promote,
        inputs=[
          f'model_handle_candidate_{suffix}',
          f'fitted_model_candidate_{suffix}',
          f'calibrated_model_candidate_{suffix}',
          f'validation_data_{suffix}',
          'parameters',
          f'model_handle_previous_{suffix}',
          f'fitted_model_previous_{suffix}',
          f'calibrated_model_previous_{suffix}',
        ],
        outputs=[f'model_handle_{suffix}', f'fitted_model_{suffix}', f'calibrated_model_{suffix}'],
        name=f'model_validation_{suffix}',
      )
    )
  else:
    nodes.append(
      node(
        func=validation_nodes.promote_without_comparison,
        inputs=[
          f'model_handle_candidate_{suffix}',
          f'fitted_model_candidate_{suffix}',
          f'calibrated_model_candidate_{suffix}',
          f'validation_data_{suffix}',
          'parameters',
        ],
        outputs=[f'model_handle_{suffix}', f'fitted_model_{suffix}', f'calibrated_model_{suffix}'],
        name=f'model_promotion_{suffix}',
      )
    )

  model_triple = [f'model_handle_{suffix}', f'fitted_model_{suffix}', f'calibrated_model_{suffix}']
  token = 1
  for slice_label, data_dataset, metadata_dataset in (
    ('train', f'training_data_{suffix}', f'train_pp_metadata_pdf_{suffix}'),
    ('valid', f'validation_data_{suffix}', f'train_pp_metadata_pdf_{suffix}'),
  ):
    nodes.extend(
      _training_slice_block(suffix, slice_label, [*model_triple, data_dataset, metadata_dataset], token)
    )
    token += 5
  return Pipeline(nodes)


def create_fe_scoring_pipeline(topology: PipelineTopology) -> Pipeline:
  """Build the feature-engineering pipeline for scoring.

  The load node is conditional. When the Spark-level split ran, the scoring frame
  already exists as that node's output and is reused — the two are the same
  frame, so loading it twice would be wasted work. Otherwise a dedicated input is
  read: the test frame in evaluation mode, the score frame in production.

  Args:
    topology: The resolved graph topology.

  Returns:
    The pipeline.
  """
  suffix = topology.suffix
  nodes: list = []

  if not topology.split:
    source = f'test_input_{suffix}' if topology.is_evaluation else f'score_input_{suffix}'
    nodes.append(
      node(
        func=feature_nodes.load_input_data,
        inputs=['parameters', 'params:global_params', source],
        outputs=[f'score_df_{suffix}', f'score_metadata_{suffix}'],
        name=f"load_{'test' if topology.is_evaluation else 'score'}_{suffix}",
      )
    )

  nodes.append(
    node(
      func=feature_nodes.apply_registry,
      inputs=[
        f'score_df_{suffix}',
        f'score_metadata_{suffix}',
        'parameters',
        'params:test',
        f'fitted_registry_{suffix}',
      ],
      outputs=[f'scoreset_data_{suffix}', f'score_pp_metadata_{suffix}'],
      name=f'apply_registry_score_{suffix}',
    )
  )
  return Pipeline(nodes)


def create_model_scoring_pipeline(topology: PipelineTopology) -> Pipeline:
  """Build the modelling pipeline for scoring.

  In production mode this pipeline is exactly two nodes and produces only the
  score table. The target column is deliberately withheld upstream, so the label
  never enters the scoreset materialisation and the scoring table cannot leak
  outcomes — a data-governance control rather than an optimisation.

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
      func=model_nodes.model_scoring,
      inputs=[
        f'model_handle_{suffix}',
        f'fitted_model_{suffix}',
        f'calibrated_model_{suffix}',
        f'score_pp_data_pdf_{suffix}',
        f'score_pp_metadata_pdf_{suffix}',
        'params:test',
      ],
      outputs=[f'prediction_df_{suffix}', f'calib_prediction_df_{suffix}', 'sequence_ind11'],
      name=f'model_scoring_{suffix}',
    ),
  ]
  # `spark_to_pandas` returns the pair (frame, contract). The scoring node
  # previously declared one output against a two-value return, which Kedro rejects
  # at run time. Both are declared here, exactly as on the training path, and the
  # converted contract is what the scoring node consumes so both stages work
  # against the same contract rather than a Spark-side copy.

  if topology.is_evaluation:
    # The two slice blocks chain rather than each starting fresh: the token the
    # scoring node emits is consumed by the first block, and that block's output
    # is consumed by the second. A gap here would leave the two blocks unordered
    # relative to each other -- which is the exact problem the tokens exist to
    # prevent, and the reason the dangling-token count is asserted in the tests.
    nodes.extend(_scoring_slice_block(suffix, 'test', f'prediction_df_{suffix}', 11))
    nodes.extend(_scoring_slice_block(suffix, 'calib_test', f'calib_prediction_df_{suffix}', 13))
    # The last block leaves sequence_ind15 dangling, exactly as each training slice
    # leaves its own final token dangling.
  return Pipeline(nodes)


# --------------------------------------------------------------------------- #
# Evaluation blocks
# --------------------------------------------------------------------------- #
def _training_slice_block(
  suffix: str,
  slice_label: str,
  model_inputs: list[str],
  first_token: int,
) -> list:
  """Build the five-node evaluation block for one training slice.

  The block scores the slice, evaluates it, computes its lift, then repeats the
  last two against the calibrated column family. Every step carries the
  sequencing token, so the chain is strictly ordered even though two nodes
  consume the same prediction frame.

  Args:
    suffix: The namespace token.
    slice_label: The slice token, ``'train'`` or ``'valid'``.
    model_inputs: The model trio, the slice's data frame and its contract.
    first_token: The index of the first sequencing token in this block.

  Returns:
    A list of five nodes.

  Raises:
    PipelineTopologyError: If the token index would exceed the declared budget.
  """
  if first_token + 4 > TRAINING_SEQUENCE_TOKENS:
    raise PipelineTopologyError(
      f'The training pipeline needs {TRAINING_SEQUENCE_TOKENS} sequencing tokens but index '
      f'{first_token + 4} was requested. A slice was added without extending the chain.',
      first_token=first_token,
    )

  prediction_frame = f'{slice_label}_prediction_df_{suffix}'
  calibrated_frame = f'{slice_label}_calib_prediction_df_{suffix}'
  metadata_dataset = model_inputs[-1]
  # The two evaluation families are named in the catalog's spelling. The training
  # slice labels are never calibration-prefixed ('train', 'valid'), so the raw
  # and calibrated stems are simply the label with and without the qualifier.
  base = dataset_stem(slice_label)
  calib = f'{base}_calib'

  return [
    node(
      func=model_nodes.model_scoring,
      inputs=[*model_inputs, f'params:{slice_label}'],
      outputs=[prediction_frame, calibrated_frame, f'sequence_ind{first_token}'],
      name=f'{slice_label}_model_scoring_{suffix}',
    ),
    node(
      func=model_nodes.model_evaluation,
      inputs=[
        prediction_frame,
        'parameters',
        metadata_dataset,
        'params:modeling_params.prediction_col',
        'params:modeling_params.prediction_probability_col',
        f'params:{slice_label}',
        'params:evaluation_params.multiclass_flag',
        f'sequence_ind{first_token}',
      ],
      outputs=[f'{base}_model_evaluation_{suffix}', f'sequence_ind{first_token + 1}'],
      name=f'{slice_label}_model_evaluation_{suffix}',
    ),
    node(
      func=model_nodes.lift_calculation,
      inputs=[
        prediction_frame,
        'parameters',
        metadata_dataset,
        'params:modeling_params.prediction_col',
        'params:modeling_params.prediction_probability_col',
        f'params:{slice_label}',
        f'sequence_ind{first_token + 1}',
      ],
      outputs=[f'{base}_lift_df_{suffix}', f'sequence_ind{first_token + 2}'],
      name=f'{slice_label}_lift_calculation_{suffix}',
    ),
    node(
      func=model_nodes.model_evaluation,
      inputs=[
        calibrated_frame,
        'parameters',
        metadata_dataset,
        'params:modeling_params.prediction_col',
        'params:modeling_params.prediction_probability_col',
        f'params:calib_{slice_label}',
        'params:evaluation_params.multiclass_flag',
        f'sequence_ind{first_token + 2}',
      ],
      outputs=[f'{calib}_model_evaluation_{suffix}', f'sequence_ind{first_token + 3}'],
      name=f'{slice_label}_calib_model_evaluation_{suffix}',
    ),
    node(
      func=model_nodes.lift_calculation,
      inputs=[
        calibrated_frame,
        'parameters',
        metadata_dataset,
        'params:modeling_params.prediction_col',
        'params:modeling_params.prediction_probability_col',
        f'params:calib_{slice_label}',
        f'sequence_ind{first_token + 3}',
      ],
      outputs=[f'{calib}_lift_df_{suffix}', f'sequence_ind{first_token + 4}'],
      name=f'{slice_label}_calib_lift_calculation_{suffix}',
    ),
  ]


def _scoring_slice_block(
  suffix: str,
  slice_label: str,
  prediction_frame: str,
  first_token: int,
) -> list:
  """Build the two-node evaluation block for one scoring slice.

  The scoring path has no ``model_scoring`` node of its own, so it consumes the
  prediction frames the scoring node emitted. The calibrated slice consumes the
  calibrated frame directly rather than re-deriving the column family from the
  slice label, which keeps the column-name convention in exactly one place.

  Args:
    suffix: The namespace token.
    slice_label: The slice token, ``'test'`` or ``'calib_test'``.
    prediction_frame: The frame this slice evaluates.
    first_token: The index of the sequencing token this block consumes.

  Returns:
    A list of two nodes. The lift node emits a trailing token so the next block
    can order itself against this one, which is what makes the two slice blocks
    one continuous chain rather than two unordered ones.

  Raises:
    PipelineTopologyError: If the token index would exceed the declared budget.
  """
  if first_token + 2 > SCORING_SEQUENCE_TOKENS:
    raise PipelineTopologyError(
      f'The scoring pipeline needs {SCORING_SEQUENCE_TOKENS} sequencing tokens but index '
      f'{first_token + 2} was requested.',
      first_token=first_token,
    )
  # The calibrated slice is labelled 'calib_test' so that it selects the
  # `params:calib_test` block and the calibrated column family, but its datasets
  # are named with the qualifier after the base slice, matching the catalog.
  base = dataset_stem(slice_label)
  return [
    node(
      func=model_nodes.model_evaluation,
      inputs=[
        prediction_frame,
        'parameters',
        f'score_pp_metadata_pdf_{suffix}',
        'params:modeling_params.prediction_col',
        'params:modeling_params.prediction_probability_col',
        f'params:{slice_label}',
        'params:evaluation_params.multiclass_flag',
        f'sequence_ind{first_token}',
      ],
      outputs=[f'{base}_model_evaluation_{suffix}', f'sequence_ind{first_token + 1}'],
      name=f'{slice_label}_model_evaluation_{suffix}',
    ),
    node(
      func=model_nodes.lift_calculation,
      inputs=[
        prediction_frame,
        'parameters',
        f'score_pp_metadata_pdf_{suffix}',
        'params:modeling_params.prediction_col',
        'params:modeling_params.prediction_probability_col',
        f'params:{slice_label}',
        f'sequence_ind{first_token + 1}',
      ],
      outputs=[f'{base}_lift_df_{suffix}', f'sequence_ind{first_token + 2}'],
      name=f'{slice_label}_lift_calculation_{suffix}',
    ),
  ]
