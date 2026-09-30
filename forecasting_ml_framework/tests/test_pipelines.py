#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for the pipeline graph.

The graph is a function of five environment variables rather than a static
declaration, which is what lets one codebase serve an unbounded number of model
instances — and which is also why a topology error only surfaces at pipeline-run
time unless something builds the graph deliberately. These tests build every
reachable topology, so a graph-shape regression is caught here rather than in a
cluster job.
"""

from __future__ import annotations

import pytest
from kedro.pipeline import Pipeline

from forecasting_ml_framework import constants
from forecasting_ml_framework.exceptions import PipelineTopologyError
from forecasting_ml_framework.pipelines.registry import (
  PipelineTopology,
  describe_pipelines,
  register_pipelines,
)

#: One topology per meaningful combination of the specialisation axes.
TOPOLOGIES = {
  'classification_prod_train_only': PipelineTopology(),
  'classification_prod_compare': PipelineTopology(
    suffix='tf', train_mode='train_compareMODEL_updateMETADATA'
  ),
  'classification_eval_update_metadata': PipelineTopology(
    suffix='tf', run_mode='eval', train_mode='train_updateMETADATA'
  ),
  'classification_eval_compare_split': PipelineTopology(
    suffix='tf', split=True, run_mode='eval', train_mode='train_compareMODEL_updateMETADATA'
  ),
  'regression_prod_train_only': PipelineTopology(suffix='vz', is_regression=True),
  'regression_eval_compare': PipelineTopology(
    suffix='vz', is_regression=True, run_mode='eval', train_mode='train_compare_update'
  ),
}


def _nodes(pipeline: Pipeline) -> list:
  """Return a pipeline's node list across Kedro versions.

  Args:
    pipeline: The pipeline to inspect.

  Returns:
    The node list.
  """
  return pipeline.nodes() if callable(pipeline.nodes) else pipeline.nodes


@pytest.fixture(autouse=True)
def _restore_environment():
  """Restore the topology environment variables after each test.

  The pipeline registry reads five environment variables at call time, so a test
  that exports a topology leaks that configuration into every later test. The
  registry is order-dependent, which makes that leak a source of tests that pass
  or fail depending on which ran first.

  Yields:
    Nothing.
  """
  import os  # noqa: PLC0415

  keys = (
    constants.ENV_SUFFIX, constants.ENV_SPLIT, constants.ENV_IS_REGRESSION,
    constants.ENV_RUN_MODE, constants.ENV_TRAIN_MODE,
  )
  saved = {key: os.environ.get(key) for key in keys}
  yield
  for key, value in saved.items():
    if value is None:
      os.environ.pop(key, None)
    else:
      os.environ[key] = value


@pytest.mark.parametrize('label', sorted(TOPOLOGIES))
def test_every_topology_builds_a_valid_dag(label: str) -> None:
  """Every reachable topology must build and be a valid DAG.

  Kedro rejects a node that lists the same dataset as both an input and an
  output, and that is a structural rule rather than a validation message -- so a
  graph which violates it fails to build at all. Building every topology is
  therefore the check that keeps the candidate-dataset routing honest, and it is
  the one assertion that would catch a graph whose shape only works for the
  topology someone happened to run.

  Args:
    label: The topology's name.
  """
  TOPOLOGIES[label].export_to_environment()
  pipelines = register_pipelines()
  assert set(pipelines) == {
    constants.PIPELINE_FE_TRAINING, constants.PIPELINE_MODEL_TRAINING,
    constants.PIPELINE_FE_SCORING, constants.PIPELINE_MODEL_SCORING, constants.PIPELINE_DEFAULT,
  }
  for name, pipeline in pipelines.items():
    assert isinstance(pipeline, Pipeline)
    assert _nodes(pipeline), f'{name} built with no nodes'
    # `inputs()` returns every dataset the graph reads, free inputs included, and
    # computing it walks the graph -- so a cycle raises here. Its length is not
    # the node count, which is why only the walk is asserted.
    assert pipeline.inputs()
    assert set(pipeline.outputs())

  default = pipelines[constants.PIPELINE_DEFAULT]
  assert len(_nodes(default)) == sum(
    len(_nodes(pipelines[name]))
    for name in (
      constants.PIPELINE_FE_TRAINING, constants.PIPELINE_MODEL_TRAINING,
      constants.PIPELINE_FE_SCORING, constants.PIPELINE_MODEL_SCORING,
    )
  ), 'the default pipeline is not the sum of the four stages'


@pytest.mark.parametrize('label', sorted(TOPOLOGIES))
def test_no_node_reads_and_writes_the_same_dataset(label: str) -> None:
  """No node may list a dataset as both an input and an output.

  Args:
    label: The topology's name.
  """
  TOPOLOGIES[label].export_to_environment()
  for pipeline in register_pipelines().values():
    for graph_node in _nodes(pipeline):
      assert not (set(graph_node.inputs) & set(graph_node.outputs)), (
        f'{graph_node.name} both reads and writes {sorted(set(graph_node.inputs) & set(graph_node.outputs))}'
      )


@pytest.mark.parametrize('label', sorted(TOPOLOGIES))
def test_exactly_one_node_writes_each_persisted_dataset(label: str) -> None:
  """Two nodes writing the same dataset would make the graph's output ambiguous.

  Args:
    label: The topology's name.
  """
  TOPOLOGIES[label].export_to_environment()
  pipelines = register_pipelines()
  # __default__ is the sum of the other four, so counting it would report every
  # node as a duplicate writer.
  stage_names = (
    constants.PIPELINE_FE_TRAINING, constants.PIPELINE_MODEL_TRAINING,
    constants.PIPELINE_FE_SCORING, constants.PIPELINE_MODEL_SCORING,
  )
  writers: dict[str, list[str]] = {}
  for stage in stage_names:
    for graph_node in _nodes(pipelines[stage]):
      for output in graph_node.outputs:
        writers.setdefault(output, []).append(graph_node.name)
  duplicated = {name: who for name, who in writers.items() if len(who) > 1}
  assert not duplicated, f'dataset(s) written by more than one node: {duplicated}'


class TestTopologyResolution:
  """The five specialisation axes and their validation."""

  def test_defaults_match_the_documented_values(self) -> None:
    """With no environment set, the documented defaults apply."""
    topology = PipelineTopology.from_environment()
    assert (topology.suffix, topology.split, topology.is_regression) == ('params', False, False)
    assert (topology.run_mode, topology.train_mode) == ('prod', 'train_only')

  def test_environment_round_trips(self) -> None:
    """Exporting and re-reading a topology must be lossless."""
    original = PipelineTopology(
      suffix='acqn', split=True, is_regression=True, run_mode='eval', train_mode='train_compare_update'
    )
    original.export_to_environment()
    assert PipelineTopology.from_environment().suffix == 'acqn'
    assert PipelineTopology.from_environment().is_compare_mode is True
    assert PipelineTopology.from_environment().is_evaluation is True

  def test_unsupported_run_mode_is_rejected(self) -> None:
    """An unrecognised run mode is a configuration issue, not a default."""
    with pytest.raises(PipelineTopologyError):
      PipelineTopology(run_mode='staging')

  def test_string_booleans_are_coerced(self) -> None:
    """The hook writes booleans as strings into the environment."""
    import os  # noqa: PLC0415

    os.environ[constants.ENV_SPLIT] = 'True'
    os.environ[constants.ENV_IS_REGRESSION] = 'False'
    topology = PipelineTopology.from_environment()
    assert topology.split is True
    assert topology.is_regression is False


class TestGraphDescription:
  """The graph must be inspectable without executing a run."""

  def test_description_names_every_pipeline_and_node(self) -> None:
    """The graph must be inspectable without executing a run."""
    PipelineTopology().export_to_environment()
    text = describe_pipelines(register_pipelines())
    for name in (constants.PIPELINE_FE_TRAINING, constants.PIPELINE_MODEL_TRAINING,
                 constants.PIPELINE_DEFAULT):
      assert name in text
    assert 'load_train_params' in text
    assert 'Pipeline topology:' in text

  def test_description_is_deterministic(self) -> None:
    """Two renders of the same topology must agree."""
    PipelineTopology().export_to_environment()
    assert describe_pipelines(register_pipelines()) == describe_pipelines(register_pipelines())


class TestSequencingTokens:
  """The dependency-token idiom, and the regression tier's exemption from it."""

  def test_classification_training_pipeline_chains_ten_tokens(self) -> None:
    """Two slices, each evaluated raw and calibrated, need ten tokens."""
    PipelineTopology(train_mode='train_compareMODEL_updateMETADATA').export_to_environment()
    training = register_pipelines()[constants.PIPELINE_MODEL_TRAINING]
    tokens = {
      name
      for graph_node in _nodes(training)
      for name in (*graph_node.inputs, *graph_node.outputs)
      if name.startswith('sequence_ind')
    }
    assert len(tokens) == 10

  def test_tokens_form_unbroken_chains(self) -> None:
    """No token may be consumed inside a pipeline without being produced there.

    A gap would leave two evaluation nodes unordered, which is the exact problem
    the tokens exist to solve. Each slice starts its own five-token chain rather
    than continuing one, so the number of dangling tokens equals the number of
    independent chains -- one per ``model_scoring`` node.

    """
    PipelineTopology(
      suffix='tf', run_mode='eval', train_mode='train_compareMODEL_updateMETADATA'
    ).export_to_environment()
    pipelines = register_pipelines()
    for stage in (constants.PIPELINE_MODEL_TRAINING, constants.PIPELINE_MODEL_SCORING):
      nodes = _nodes(pipelines[stage])
      produced, consumed = set(), set()
      for graph_node in nodes:
        produced.update(n for n in graph_node.outputs if n.startswith('sequence_ind'))
        consumed.update(n for n in graph_node.inputs if n.startswith('sequence_ind'))
      assert consumed <= produced, f'{stage} consumes a token nothing produces: {consumed - produced}'
      # Each `model_scoring` node starts one five-token chain: score, evaluate,
      # lift, evaluate-calibrated, lift-calibrated.
      chains = sum(1 for n in nodes if 'model_scoring' in n.name)
      assert chains, f'{stage} has no scoring node to start a chain'
      assert len(produced - consumed) == chains, (
        f'{stage}: {len(produced - consumed)} dangling tokens for {chains} chains'
      )

  def test_regression_pipeline_needs_no_tokens(self) -> None:
    """With no calibration dimension there is no fan-out, so no tokens."""
    PipelineTopology(is_regression=True, train_mode='train_compare_update').export_to_environment()
    training = register_pipelines()[constants.PIPELINE_MODEL_TRAINING]
    tokens = [
      name
      for graph_node in _nodes(training)
      for name in (*graph_node.inputs, *graph_node.outputs)
      if name.startswith('sequence_ind')
    ]
    assert not tokens
