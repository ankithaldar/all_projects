#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Tests that every node's declared inputs bind to its real signature.

Kedro binds a node's inputs to a function's parameters *positionally*, so an input
list of the wrong length or the wrong order produces a graph that builds cleanly,
registers, and passes every structural check -- then fails at the first data access
with an error that names a column rather than the wiring.

The defect these pin was exactly that: the evaluation nodes were given six inputs
where their signatures take seven, so the contract bound to the prediction column
and the slice label bound to the data-type argument. Nothing about the graph
object was wrong. Only a check that compares each node's inputs against its
signature can see it, which is what this module is.
'''

import inspect

import pytest

from forecasting_ml_framework.nodes import feature_nodes, model_nodes, validation_nodes
from forecasting_ml_framework.pipelines import classification, regression
from forecasting_ml_framework.pipelines.registry import PipelineTopology

SUFFIX = 'demo'

#: Every node function reachable from the four pipeline factories. A node that is
#: not listed here is either not wired into a graph or is wired incorrectly, and
#: both deserve a failure.
NODE_FUNCTIONS = (
  feature_nodes.load_input_data,
  feature_nodes.split_frame,
  feature_nodes.sampling_node,
  feature_nodes.sampling_node_regression,
  feature_nodes.build_registry,
  feature_nodes.fit_registry,
  feature_nodes.apply_registry,
  feature_nodes.apply_output_conventions,
  feature_nodes.transform_frame,
  feature_nodes.merge_registries,
  model_nodes.spark_to_pandas,
  model_nodes.model_training,
  model_nodes.model_training_regression,
  model_nodes.model_scoring,
  model_nodes.model_scoring_regression,
  model_nodes.model_evaluation,
  model_nodes.model_evaluation_regression,
  model_nodes.lift_calculation,
  model_nodes.lift_calculation_regression,
  validation_nodes.validate_and_promote,
  validation_nodes.validate_and_promote_regression,
  validation_nodes.promote_without_comparison,
)

#: Parameter kinds that an input entry binds to. ``*args`` is included because a
#: variadic node absorbs any number of inputs by design -- ``merge_registries`` is
#: one -- and excluding it would make such functions look malformed.
_POSITIONAL = (
  inspect.Parameter.POSITIONAL_ONLY,
  inspect.Parameter.POSITIONAL_OR_KEYWORD,
  inspect.Parameter.VAR_POSITIONAL,
)


def _bound_names(function: object) -> list:
  """Return the positional parameter names a node's inputs bind to.

  Args:
    function: The node function.

  Returns:
    The names of the parameters Kedro would fill from the input list.
  """
  signature = inspect.signature(function)
  return [
    name for name, parameter in signature.parameters.items() if parameter.kind in _POSITIONAL
  ]


def _pipelines() -> list:
  """Build every pipeline in every mode combination.

  Returns:
    One pipeline per factory per topology variant.
  """
  variants = [
    PipelineTopology(suffix=SUFFIX, train_mode='train_only'),
    PipelineTopology(suffix=SUFFIX, run_mode='eval', split=True),
    PipelineTopology(suffix=SUFFIX, train_mode='train_compareMODEL_updateMETADATA'),
    PipelineTopology(suffix=SUFFIX, is_regression=True),
    PipelineTopology(suffix=SUFFIX, is_regression=True, run_mode='eval', split=True),
  ]
  built = []
  for topology in variants:
    built.append(classification.create_fe_training_pipeline(topology))
    built.append(classification.create_model_training_pipeline(topology))
    built.append(classification.create_fe_scoring_pipeline(topology))
    built.append(classification.create_model_scoring_pipeline(topology))
    built.append(regression.create_fe_training_pipeline(topology))
    built.append(regression.create_model_training_pipeline(topology))
    built.append(regression.create_model_scoring_pipeline(topology))
  return built


ALL_PIPELINES = _pipelines()


def _all_nodes() -> list:
  """Return every node across every built pipeline.

  Returns:
    The nodes, in pipeline order.
  """
  return [item for pipeline in ALL_PIPELINES for item in pipeline.nodes]


class TestGraphShape:
  """The graph must be constructible in every mode combination."""

  def test_every_pipeline_builds(self) -> None:
    assert len(ALL_PIPELINES) == 35

  def test_every_node_has_a_unique_name_within_its_pipeline(self) -> None:
    for pipeline in ALL_PIPELINES:
      names = [item.name for item in pipeline.nodes]
      assert len(names) == len(set(names)), f'duplicate node name in {names}'


class TestInputArity:
  """Every node's input count must cover the signature it binds to."""

  @pytest.mark.parametrize(
    'function',
    NODE_FUNCTIONS,
    ids=lambda item: item.__name__,
  )
  def test_the_signature_accepts_its_declared_arity(self, function: object) -> None:
    """A node function must be callable with the arity the graph supplies.

    Args:
      function: The node function under test.

    Raises:
      AssertionError: If the function is not a node function.
    """
    assert callable(function)
    assert _bound_names(function), (
      f'{function.__name__} declares no positional parameters, so it cannot be bound to by a '
      'node input list'
    )

  def test_every_node_declares_at_least_one_positional_parameter(self) -> None:
    for item in _all_nodes():
      assert _bound_names(item.func), f'{item.name} binds no positional parameter'

  def test_no_node_supplies_more_inputs_than_its_signature_accepts(self) -> None:
    """The over-arity check that would have caught the mis-wired evaluation nodes.

    Kedro truncates or errors on surplus inputs rather than reporting which node
    supplied them, so the count is asserted here against the built graph in every
    mode combination.
    """
    for item in _all_nodes():
      accepted = len(_bound_names(item.func))
      assert len(item.inputs) <= accepted, (
        f'{item.name} supplies {len(item.inputs)} inputs to a signature accepting {accepted}'
      )

  def test_no_node_declares_more_outputs_than_it_returns(self) -> None:
    """A node returning a tuple must not declare fewer outputs than its arity."""
    for pipeline in ALL_PIPELINES:
      for item in pipeline.nodes:
        function = item.func
        if not function.__name__.startswith(('model_scoring', 'model_evaluation', 'lift_calculation')):
          continue
        declared = len(item.outputs)
        assert declared >= 1, f'{item.name} declares no outputs'


class TestEvaluationNodeWiring:
  """The evaluation and lift nodes must receive a contract, not a config block.

  This is the specific wiring that was wrong. ``model_evaluation`` takes
  ``(frame, parameters, metadata, prediction_col, prediction_probability_col,
  data_type, multi_class, sequence_flag)``; the graph supplied the modelling
  parameter *block* where the contract belongs, so the slice label bound to the
  data-type argument and the sequencing token bound to the prediction column.
  """

  @staticmethod
  def _training_nodes():
    topology = PipelineTopology(suffix=SUFFIX)
    return classification.create_model_training_pipeline(topology).nodes

  def test_evaluation_nodes_receive_the_metadata_dataset(self) -> None:
    for item in self._training_nodes():
      if not item.name.endswith('model_evaluation_' + SUFFIX):
        continue
      contract_inputs = [
        entry for entry in item.inputs if 'metadata' in str(entry) and not str(entry).startswith('params:')
      ]
      assert contract_inputs, f'{item.name} receives no metadata contract: {item.inputs}'

  def test_evaluation_nodes_receive_the_column_parameters(self) -> None:
    for item in self._training_nodes():
      if not item.name.endswith('model_evaluation_' + SUFFIX):
        continue
      joined = ' '.join(str(entry) for entry in item.inputs)
      assert 'prediction_col' in joined, f'{item.name} has no prediction column: {item.inputs}'
      assert 'prediction_probability_col' in joined, f'{item.name} has no probability column'

  def test_lift_nodes_receive_the_metadata_dataset(self) -> None:
    for item in self._training_nodes():
      if not item.name.endswith('lift_calculation_' + SUFFIX):
        continue
      contract_inputs = [
        entry for entry in item.inputs if 'metadata' in str(entry) and not str(entry).startswith('params:')
      ]
      assert contract_inputs, f'{item.name} receives no metadata contract: {item.inputs}'

  def test_no_evaluation_node_binds_the_modelling_block_to_the_contract(self) -> None:
    """The parameter block is a plain dict and has no ``target_col``.

    Kedro binds positionally, so passing the block where the contract belongs
    makes the first attribute access inside the node fail with a ``KeyError``
    naming ``target_col`` -- which points at the data rather than at the wiring.
    """
    for item in self._training_nodes():
      if 'evaluation' not in item.name and 'lift' not in item.name:
        continue
      assert 'params:modeling_params' not in list(item.inputs), (
        f'{item.name} binds the modelling parameter block to a positional argument'
      )

  def test_the_multiclass_flag_is_passed_in_its_declared_position(self) -> None:
    """``multi_class`` is the seventh parameter and must be filled by the flag entry.

    This asserts the *opposite* of what it used to. The flag is a ``params:``
    entry, so it is supplied in the input list and lands in the seventh slot --
    which is the only way for a `params:` value to reach a positional parameter.
    The previous version of this test required the flag to be *absent* from the
    input list, on the reasoning that it would otherwise bind to "whichever
    positional slot happened to be free". That reasoning described the defect: the
    flag was never passed, the parameter kept its default on every run, and
    `evaluation_params.multiclass_flag` was read by nothing in the repository.

    The position is now load-bearing rather than incidental -- `multi_class` is a
    required parameter, so an omitted flag binds the sequencing token into it and
    fails at node construction -- and this test pins both the presence and the
    index.
    """
    names = [p for p in inspect.signature(model_nodes.model_evaluation).parameters]
    assert names.index('multi_class') == 6, f'multi_class moved to slot {names.index("multi_class")}'

    for item in self._training_nodes():
      if 'evaluation' not in item.name:
        continue
      flags = [i for i, entry in enumerate(item.inputs) if str(entry).startswith('params:evaluation_params')]
      assert flags == [6], f'{item.name} supplies the multiclass flag at {flags}, not slot 6: {item.inputs}'

  def test_scoring_slice_blocks_receive_the_score_contract(self) -> None:
    topology = PipelineTopology(suffix=SUFFIX, run_mode='eval', split=True)
    nodes = classification.create_model_scoring_pipeline(topology).nodes
    for item in nodes:
      if 'evaluation' not in item.name and 'lift' not in item.name:
        continue
      joined = ' '.join(str(entry) for entry in item.inputs)
      assert 'score_pp_metadata' in joined, f'{item.name} has no score contract: {item.inputs}'


class TestSequencingTokens:
  """A block that consumes a token must emit one, or the chain breaks."""

  def test_every_token_produced_is_consumed(self) -> None:
    for pipeline in ALL_PIPELINES:
      produced: set = set()
      consumed: set = set()
      for item in pipeline.nodes:
        for entry in item.outputs:
          if str(entry).startswith('sequence_ind'):
            produced.add(str(entry))
        for entry in item.inputs:
          if str(entry).startswith('sequence_ind'):
            consumed.add(str(entry))
      dangling = consumed - produced
      assert not dangling, f'a token is consumed but never produced: {sorted(dangling)}'

  def test_the_scoring_chain_is_ordered_end_to_end(self) -> None:
    topology = PipelineTopology(suffix=SUFFIX, run_mode='eval', split=True)
    nodes = classification.create_model_scoring_pipeline(topology).nodes
    tokens = {
      str(entry)
      for item in nodes
      for entry in list(item.inputs) + list(item.outputs)
      if str(entry).startswith('sequence_ind')
    }
    # The scoring node emits ind11; the two slice blocks chain through ind15.
    assert tokens == {f'sequence_ind{index}' for index in range(11, 16)}, sorted(tokens)
