#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Bad-kwargs and passthrough tests for pipeline wrappers.

Each ``create_pipeline`` wrapper must reject unknown
kwargs (forwarded to Kedro Pipeline) and wire its default
node names, so a dropped node turns red.
'''

import pytest

from forecast_ml_framework.pipelines.cost_tracking.pipeline import (
  create_pipeline as create_cost_pipeline,
)
from forecast_ml_framework.pipelines.data_prep.pipeline import (
  create_pipeline as create_data_prep_pipeline,
)
from forecast_ml_framework.pipelines.eda.pipeline import (
  create_pipeline as create_eda_pipeline,
)
from forecast_ml_framework.pipelines.interpretability.pipeline import (
  create_pipeline as create_interp_pipeline,
)
from forecast_ml_framework.pipelines.modeling.pipeline import (
  create_pipeline as create_modeling_pipeline,
)
from forecast_ml_framework.pipelines.monitoring.pipeline import (
  create_pipeline as create_monitoring_pipeline,
)
from forecast_ml_framework.pipelines.retraining.pipeline import (
  create_pipeline as create_retraining_pipeline,
)
from forecast_ml_framework.pipelines.scoring.pipeline import (
  create_pipeline as create_scoring_pipeline,
)


def test_data_prep_wrapper_nodes_and_bad_kwarg():
  '''Data prep wrapper wires nodes, rejects bad kwargs.

  Fails if a data prep node goes missing or unknown kwargs
  stop raising TypeError.
  '''
  pipe = create_data_prep_pipeline()
  assert [node.name for node in pipe.nodes] == [
    'data_ingestion_validate', 'split_raw', 'fit_preprocess',
    'apply_preprocess', 'feature_selection', 'apply_selection']
  split = {node.name: node for node in pipe.nodes}['split_raw']
  assert split.func.__name__ == 'split_raw_data_with_meta'
  assert split.outputs == [
    'train_raw', 'test_raw', 'split_meta_raw']
  with pytest.raises(TypeError):
    create_data_prep_pipeline(bogus_kwarg_xyz=1)


def test_eda_wrapper_nodes_and_bad_kwarg():
  '''EDA wrapper wires nodes, rejects bad kwargs.

  Fails if an EDA node goes missing or unknown kwargs stop
  raising TypeError.
  '''
  pipe = create_eda_pipeline()
  assert [node.name for node in pipe.nodes] == [
    'eda_summary', 'eda_report']
  with pytest.raises(TypeError):
    create_eda_pipeline(bogus_kwarg_xyz=1)


def test_modeling_wrapper_nodes_and_bad_kwarg():
  '''Modeling wrapper wires the split node, rejects bad kwargs.

  Fails if the split node goes missing or unknown kwargs
  stop raising TypeError.
  '''
  pipe = create_modeling_pipeline()
  assert 'split_data_node' in [
    node.name for node in pipe.nodes]
  with pytest.raises(TypeError):
    create_modeling_pipeline(bogus_kwarg_xyz=1)


def test_scoring_wrapper_defaults_and_overrides():
  '''Scoring wrapper defaults and output overrides work.

  Fails if scoring defaults change or overrides stop wiring.
  '''
  pipe = create_scoring_pipeline()
  assert [node.name for node in pipe.nodes] == [
    'score_data_prep', 'predict', 'write_predictions']
  assert pipe.nodes[-1].outputs == ['predictions']
  overridden = create_scoring_pipeline({'output_dataset': 'my_out'})
  assert overridden.nodes[-1].outputs == ['my_out']
  with pytest.raises(TypeError):
    create_scoring_pipeline(bogus_kwarg_xyz=1)


def test_monitoring_wrapper_defaults_and_overrides():
  '''Monitoring wrapper defaults and output overrides work.

  Fails if monitoring defaults change or overrides stop wiring.
  '''
  pipe = create_monitoring_pipeline()
  assert pipe.nodes[-1].outputs == ['alerts']
  overridden = create_monitoring_pipeline(
    {'output_dataset': 'my_alerts'})
  assert overridden.nodes[-1].outputs == ['my_alerts']
  with pytest.raises(TypeError):
    create_monitoring_pipeline(bogus_kwarg_xyz=1)


def test_retraining_wrapper_nodes_and_bad_kwarg():
  '''Retraining wrapper wires nodes, rejects bad kwargs.

  Fails if a retraining node goes missing or unknown kwargs
  stop raising TypeError.
  '''
  pipe = create_retraining_pipeline()
  assert [node.name for node in pipe.nodes] == [
    'should_retrain', 'build_retrain_plan']
  with pytest.raises(TypeError):
    create_retraining_pipeline(bogus_kwarg_xyz=1)


def test_cost_tracking_wrapper_nodes_and_bad_kwarg():
  '''Cost tracking wrapper wires nodes, rejects bad kwargs.

  Fails if a cost node goes missing or unknown kwargs stop
  raising TypeError.
  '''
  pipe = create_cost_pipeline()
  assert [node.name for node in pipe.nodes] == [
    'estimate_cost', 'summarize_costs', 'track_run_cost']
  with pytest.raises(TypeError):
    create_cost_pipeline(bogus_kwarg_xyz=1)


def test_interpretability_wrapper_nodes_and_bad_kwarg():
  '''Interpretability wrapper wires nodes, rejects bad kwargs.

  Fails if an interpretability node goes missing or unknown
  kwargs stop raising TypeError.
  '''
  pipe = create_interp_pipeline()
  assert 'explain_model' in [
    node.name for node in pipe.nodes]
  assert 'make_interpretability_charts' in [
    node.name for node in pipe.nodes]
  with pytest.raises(TypeError):
    create_interp_pipeline(bogus_kwarg_xyz=1)
