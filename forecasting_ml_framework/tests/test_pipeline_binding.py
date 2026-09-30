#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''End-to-end execution of the node graph against a real local back end.

Every other graph test inspects the graph object. This module runs it, which is the
only way to see that a node's inputs resolve, that each stage's output contract
matches what the next stage reads, and that the values travelling between them are
the values the downstream code expects.

The back end is the embedded relational store and the compute runtime is a local
Spark session, both reached through the same node functions a production run uses.
Nothing here is mocked: a defect in the wiring shows up as a failure, and a defect
in a contract shows up as a value that is quietly wrong.
'''

import json
import pathlib
import sqlite3

import pandas as pd
import pytest

from forecasting_ml_framework.core.metadata import Metadata
from forecasting_ml_framework.exceptions import FrameworkError
from forecasting_ml_framework.local.sqlite_warehouse import SQLiteWarehouse

CONF_ROOT = pathlib.Path(__file__).resolve().parents[1] / 'conf'

#: Rows of labelled, numeric-only feature data. The positive rate is low, which is
#: the regime the framework is built for: it makes the threshold-selection path and
#: the prior-shift correction observable rather than incidental.
ROWS = 600
FEATURES = ('usage_feature_01', 'usage_feature_02', 'tenure_days', 'arpu_30d')
TARGET = 'label'
IDENTIFIER = 'cust_id'


def _frame(rows: int = ROWS) -> pd.DataFrame:
  """Build a deterministic labelled feature frame.

  Args:
    rows: The number of rows to build.

  Returns:
    The frame.
  """
  generator = pd.Series(range(rows), dtype='float64')
  signal = ((generator * 37) % 11 < 2).astype('int64')
  return pd.DataFrame(
    {
      IDENTIFIER: [f'cust_{index:05d}' for index in range(rows)],
      FEATURES[0]: (generator % 97) / 97.0,
      FEATURES[1]: ((generator * 13) % 89) / 89.0,
      FEATURES[2]: (generator % 400) + 1,
      FEATURES[3]: ((generator * 7) % 53) / 53.0,
      TARGET: signal,
    }
  )


def _metadata(rows: int = ROWS) -> Metadata:
  """Build the schema contract describing the frame.

  Args:
    rows: The number of rows the contract describes.

  Returns:
    The contract.
  """
  return Metadata(
    target_col=TARGET,
    feature_cols=list(FEATURES),
    numerical_cols=list(FEATURES),
    categorical_cols=[],
    seq_cols=[],
    id_cols=[IDENTIFIER],
  )


def _parameters() -> dict:
  """Build a parameters document sufficient for the modelling nodes.

  Returns:
    The document.
  """
  return {
    'model_key': 'graph_test',
    'modeling_params': {
      'model_handle': 'sklearn.tree.DecisionTreeClassifier',
      'model_params': {'max_depth': 4, 'random_state': 42},
      'eval_size': 0.25,
      'threshold_selection_method': 'roc_curve',
      'eval_metric': 'LIFT_DECILE_10',
    },
    'metrics_path': '',
  }


class TestTrainAndScoreNodes:
  '''The modelling tier, run for real.'''

  @pytest.fixture(scope='class')
  @classmethod
  def trained(cls) -> tuple:
    """Train a model on the synthetic frame.

    Returns:
      A five-tuple of the trainer, the fitted model, the calibrator and the two
      slices, exactly as the training node returns it.
    """
    from forecasting_ml_framework.nodes.model_nodes import model_training  # noqa: PLC0415

    frame = _frame()
    return model_training(_parameters(), frame, _metadata())

  def test_training_returns_the_declared_five_tuple(self, trained: tuple) -> None:
    assert len(trained) == 5

  def test_the_holdout_is_disjoint_from_the_training_slice(self, trained: tuple) -> None:
    """An overlapping holdout would inflate every reported metric."""
    train_frame, holdout_frame = trained[3], trained[4]
    assert not holdout_frame.empty
    assert set(train_frame[IDENTIFIER]).isdisjoint(set(holdout_frame[IDENTIFIER]))

  def test_the_slices_reconstruct_the_input(self, trained: tuple) -> None:
    train_frame, holdout_frame = trained[3], trained[4]
    assert len(train_frame) + len(holdout_frame) == ROWS

  def test_a_threshold_is_selected(self, trained: tuple) -> None:
    trainer = trained[0]
    assert 0.0 < trainer.optimal_model_threshold < 1.0

  def test_the_feature_vector_is_ordered_and_resolved(self, trained: tuple) -> None:
    trainer = trained[0]
    assert trainer.features_to_use == list(FEATURES)
    assert trainer.target_col == TARGET

  def test_the_population_rate_is_recorded(self, trained: tuple) -> None:
    """Prior-shift calibration is meaningless without the pre-resampling rate."""
    trainer = trained[0]
    assert trainer.real_rate and 0.0 < trainer.real_rate < 1.0

  def test_scoring_produces_both_projections(self, trained: tuple) -> None:
    from forecasting_ml_framework.nodes.model_nodes import model_scoring  # noqa: PLC0415

    trainer, fitted, calibrated, unused_train, holdout = trained
    del unused_train
    raw, calibrated_frame, token = model_scoring(trainer, fitted, calibrated, holdout, _metadata(), 'valid')
    assert not raw.empty
    assert not calibrated_frame.empty
    assert token is True
    assert 'score_value' in raw.columns
    assert 'orig_score_value' in raw.columns

  def test_the_identifier_survives_scoring(self, trained: tuple) -> None:
    """A score row without its key cannot be joined to anything."""
    from forecasting_ml_framework.nodes.model_nodes import model_scoring  # noqa: PLC0415

    trainer, fitted, calibrated, unused_train, holdout = trained
    del unused_train
    raw, unused_calibrated, unused_token = model_scoring(
      trainer, fitted, calibrated, holdout, _metadata(), 'valid'
    )
    del unused_calibrated, unused_token
    assert IDENTIFIER in raw.columns

  def test_the_score_projection_is_narrow(self, trained: tuple) -> None:
    """A narrow, stable schema is what lets many models share one score table."""
    from forecasting_ml_framework.modeling.scoring import (  # noqa: PLC0415
      project_score_frame,
    )
    from forecasting_ml_framework.nodes.model_nodes import model_scoring  # noqa: PLC0415

    trainer, fitted, calibrated, unused_train, holdout = trained
    del unused_train
    raw, unused_calibrated, unused_token = model_scoring(
      trainer, fitted, calibrated, holdout, _metadata(), 'valid'
    )
    del unused_calibrated, unused_token
    projected = project_score_frame(
      raw, [IDENTIFIER], trainer.target_col, None, None, 'graph_test', calibrated=True
    )
    assert IDENTIFIER in projected.columns
    assert len(projected) == len(raw)


class TestEvaluationNodes:
  '''The evaluation tier, run on a real scored frame.'''

  @pytest.fixture(scope='class')
  @classmethod
  def scored(cls) -> pd.DataFrame:
    """Train, then score the holdout.

    Returns:
      The scored frame.
    """
    from forecasting_ml_framework.nodes.model_nodes import model_scoring, model_training  # noqa: PLC0415

    trainer, fitted, calibrated, unused_train, holdout = model_training(_parameters(), _frame(), _metadata())
    del unused_train
    raw, unused_calibrated, unused_token = model_scoring(
      trainer, fitted, calibrated, holdout, _metadata(), 'valid'
    )
    del unused_calibrated, unused_token
    return raw

  def test_the_slice_metric_row_is_produced(self, scored: pd.DataFrame) -> None:
    from forecasting_ml_framework.nodes.model_nodes import model_evaluation  # noqa: PLC0415

    row, token = model_evaluation(
      scored,
      _parameters(),
      _metadata(),
      'prediction',
      'score_value',
      'valid',
      False,
      True,
    )
    assert not row.empty
    assert row.iloc[0]['data_type'] == 'valid'
    assert token is True

  def test_the_metric_row_reports_accuracy_within_range(self, scored: pd.DataFrame) -> None:
    from forecasting_ml_framework.nodes.model_nodes import model_evaluation  # noqa: PLC0415

    row, unused_token = model_evaluation(
      scored, _parameters(), _metadata(), 'prediction', 'score_value', 'valid', False, True
    )
    del unused_token
    assert 0.0 <= float(row.iloc[0]['accuracy']) <= 1.0

  def test_the_lift_table_has_one_row_per_band(self, scored: pd.DataFrame) -> None:
    from forecasting_ml_framework.nodes.model_nodes import lift_calculation  # noqa: PLC0415

    lift, token = lift_calculation(
      scored, _parameters(), _metadata(), 'prediction', 'score_value', 'valid', True
    )
    assert not lift.empty
    assert 'score_ntile' in lift.columns
    assert lift['score_ntile'].nunique() <= 10
    assert token is True

  def test_the_lift_is_a_ratio_against_the_base_rate(self, scored: pd.DataFrame) -> None:
    """A lift table in which every band reports the same value is not a lift table."""
    from forecasting_ml_framework.nodes.model_nodes import lift_calculation  # noqa: PLC0415

    lift, unused_token = lift_calculation(
      scored, _parameters(), _metadata(), 'prediction', 'score_value', 'valid', True
    )
    del unused_token
    assert lift['LIFT'].nunique() > 1


class TestRegressionNodes:
  '''The regression tier, run for real.'''

  @pytest.fixture(scope='class')
  @classmethod
  def continuous(cls) -> pd.DataFrame:
    """Build a continuous-target frame.

    Returns:
      The frame.
    """
    frame = _frame()
    frame['spend'] = frame[FEATURES[0]] * 100.0 + frame[FEATURES[3]] * 50.0
    return frame

  def test_regression_training_returns_the_five_tuple(self, continuous: pd.DataFrame) -> None:
    from forecasting_ml_framework.nodes.model_nodes import model_training  # noqa: PLC0415

    metadata = _metadata()
    metadata.target_col = 'spend'
    metadata.numerical_cols = [*FEATURES, 'spend']
    metadata.feature_cols = [*FEATURES, 'spend']
    parameters = {
      'model_key': 'graph_test',
      'modeling_params': {
        'model_handle': 'sklearn.tree.DecisionTreeRegressor',
        'model_params': {'max_depth': 4, 'random_state': 42},
        'eval_size': 0.25,
        'eval_metric': 'RMSE',
      },
    }
    trainer, fitted, calibrated, train_frame, holdout = model_training(parameters, continuous, metadata)
    assert trainer is not None
    assert fitted is not None
    assert calibrated is not None
    assert not train_frame.empty
    assert not holdout.empty

  def test_the_regression_metric_row_is_produced(self, continuous: pd.DataFrame) -> None:
    from forecasting_ml_framework.nodes.model_nodes import (  # noqa: PLC0415
      model_evaluation_regression,
      model_scoring_regression,
      model_training,
    )

    metadata = _metadata()
    metadata.target_col = 'spend'
    metadata.numerical_cols = [*FEATURES, 'spend']
    metadata.feature_cols = [*FEATURES, 'spend']
    parameters = {
      'model_key': 'graph_test',
      'modeling_params': {
        'model_handle': 'sklearn.tree.DecisionTreeRegressor',
        'model_params': {'max_depth': 4, 'random_state': 42},
        'eval_size': 0.25,
      },
    }
    trainer, fitted, calibrated, unused_train, holdout = model_training(
      parameters, continuous, metadata
    )
    del unused_train
    # Two values, not three: the regression graph has no calibration fan-out and
    # therefore no sequencing token to thread, so the node returns the raw and
    # calibrated projections alone.
    raw, unused_calibrated = model_scoring_regression(
      trainer, fitted, calibrated, holdout, metadata, 'valid'
    )
    del unused_calibrated
    row = model_evaluation_regression(raw, parameters, metadata, 'prediction', 'valid')
    assert row.iloc[0]['data_type'] == 'valid'
    assert float(row.iloc[0]['RMSE']) >= 0.0

  def test_the_regression_projection_carries_an_empty_high_score_flag(
    self, continuous: pd.DataFrame
  ) -> None:
    """The flag is retained purely so the score-table schema stays stable."""
    from forecasting_ml_framework import constants  # noqa: PLC0415
    from forecasting_ml_framework.nodes.model_nodes import (  # noqa: PLC0415
      model_scoring_regression,
      model_training,
    )

    metadata = _metadata()
    metadata.target_col = 'spend'
    metadata.numerical_cols = [*FEATURES, 'spend']
    metadata.feature_cols = [*FEATURES, 'spend']
    parameters = {
      'model_key': 'graph_test',
      'modeling_params': {
        'model_handle': 'sklearn.tree.DecisionTreeRegressor',
        'model_params': {'max_depth': 4, 'random_state': 42},
        'eval_size': 0.25,
      },
    }
    trainer, fitted, calibrated, unused_train, holdout = model_training(
      parameters, continuous, metadata
    )
    del unused_train
    # Two values, not three: the regression graph has no calibration fan-out and
    # therefore no sequencing token to thread, so the node returns the raw and
    # calibrated projections alone.
    raw, unused_calibrated = model_scoring_regression(
      trainer, fitted, calibrated, holdout, metadata, 'valid'
    )
    del unused_calibrated
    assert constants.HIGH_SCORE_IND in raw.columns
    assert set(raw[constants.HIGH_SCORE_IND]) == {''}


class TestPositiveClassResolution:
  """The positive class must be resolved from the estimator, not assumed."""

  def test_a_classifier_yields_a_probability(self) -> None:
    from sklearn.linear_model import LogisticRegression  # noqa: PLC0415

    from forecasting_ml_framework.modeling.scoring import positive_probability  # noqa: PLC0415

    frame = pd.DataFrame({'x': [0.0, 1.0, 2.0, 3.0]})
    model = LogisticRegression().fit(frame, [0, 0, 1, 1])
    score = positive_probability(model, frame)
    assert len(score) == len(frame)
    assert ((score >= 0.0) & (score <= 1.0)).all()

  def test_a_non_binary_target_selects_the_highest_class(self) -> None:
    """A hard-coded column index would make class 2 positive by accident."""
    from sklearn.linear_model import LogisticRegression  # noqa: PLC0415

    from forecasting_ml_framework.modeling.scoring import positive_probability  # noqa: PLC0415

    frame = pd.DataFrame({'x': [0.0, 1.0, 2.0, 3.0]})
    model = LogisticRegression().fit(frame, [1, 1, 2, 2])
    score = positive_probability(model, frame)
    assert list(model.classes_) == [1, 2]
    # Column 1 is the one returned, so its values must match the estimator's own.
    assert score == pytest.approx(model.predict_proba(frame)[:, 1])

  def test_a_regressor_score_is_the_prediction_itself(self) -> None:
    """The defect: ``predict_proba`` was called unconditionally.

    The score is returned unscaled, because a regression metric compares the
    prediction to the target on the target's own scale. Rescaling onto ``(0, 1)``
    would make the score depend on the other rows in the batch.
    """
    from sklearn.tree import DecisionTreeRegressor  # noqa: PLC0415

    from forecasting_ml_framework.modeling.scoring import positive_probability  # noqa: PLC0415

    frame = pd.DataFrame({'x': [0.0, 10.0, 20.0, 30.0]})
    model = DecisionTreeRegressor().fit(frame, [0.0, 100.0, 200.0, 300.0])
    score = positive_probability(model, frame)
    assert score == pytest.approx(model.predict(frame))

  def test_a_regressor_score_does_not_depend_on_the_batch(self) -> None:
    """Two runs over different populations must produce comparable scores."""
    from sklearn.tree import DecisionTreeRegressor  # noqa: PLC0415

    from forecasting_ml_framework.modeling.scoring import positive_probability  # noqa: PLC0415

    train = pd.DataFrame({'x': [0.0, 1.0, 2.0, 3.0]})
    model = DecisionTreeRegressor().fit(train, [0.0, 1.0, 2.0, 3.0])
    first = positive_probability(model, pd.DataFrame({'x': [1.0]}))
    second = positive_probability(model, pd.DataFrame({'x': [1.0, 2.0, 3.0, 9.0]}))
    assert first[0] == pytest.approx(second[0])

  def test_a_decision_margin_is_brought_onto_the_unit_interval(self) -> None:
    from sklearn.svm import SVC  # noqa: PLC0415

    from forecasting_ml_framework.modeling.scoring import positive_probability  # noqa: PLC0415

    frame = pd.DataFrame({'x': [0.0, 1.0, 2.0, 3.0]})
    model = SVC(probability=False).fit(frame, [0, 0, 1, 1])
    score = positive_probability(model, frame)
    assert ((score >= 0.0) & (score <= 1.0)).all()

  def test_an_estimator_with_neither_is_reported(self) -> None:
    from forecasting_ml_framework.exceptions import ModelError  # noqa: PLC0415
    from forecasting_ml_framework.modeling.scoring import positive_probability  # noqa: PLC0415

    class Useless:
      """An estimator offering no way to produce a score."""

    with pytest.raises(ModelError):
      positive_probability(Useless(), pd.DataFrame({'x': [1.0]}))


class TestWarehouseRoundTrip:
  '''The embedded back end must round-trip what it wrote.'''

  def test_a_written_frame_reads_back_identically(self, tmp_path: pathlib.Path) -> None:
    with SQLiteWarehouse(str(tmp_path / 'local.db')) as warehouse:
      original = _frame(50)
      warehouse.write(original, 'features')
      assert warehouse.count('features') == len(original)
      reloaded = warehouse.query('SELECT * FROM features')
      assert list(reloaded.columns) == list(original.columns)
      assert reloaded[TARGET].sum() == original[TARGET].sum()

  def test_a_second_write_replaces_rather_than_appends(self, tmp_path: pathlib.Path) -> None:
    """Idempotency: a re-run for a date must not duplicate the rows."""
    with SQLiteWarehouse(str(tmp_path / 'local.db')) as warehouse:
      warehouse.write(_frame(50), 'features')
      warehouse.write(_frame(30), 'features')
      assert warehouse.count('features') == 30

  def test_the_scope_releases_the_handle(self, tmp_path: pathlib.Path) -> None:
    path = str(tmp_path / 'local.db')
    with SQLiteWarehouse(path) as warehouse:
      warehouse.write(_frame(10), 't')
    assert sqlite3.connect(path).execute('SELECT COUNT(*) FROM t').fetchone()[0] == 10

  def test_a_closed_handle_is_reported(self, tmp_path: pathlib.Path) -> None:
    from forecasting_ml_framework.local.sqlite_warehouse import (  # noqa: PLC0415
      SqliteWarehouseError,
    )

    warehouse = SQLiteWarehouse(str(tmp_path / 'local.db'))
    warehouse.close()
    with pytest.raises(SqliteWarehouseError):
      _ = warehouse.connection

  def test_closing_twice_is_harmless(self, tmp_path: pathlib.Path) -> None:
    warehouse = SQLiteWarehouse(str(tmp_path / 'local.db'))
    warehouse.close()
    warehouse.close()


class TestArtefactRoundTrip:
  '''The train/serve contract, exercised through real files.'''

  def test_a_pickle_artefact_round_trips(self, tmp_path: pathlib.Path) -> None:
    from forecasting_ml_framework.utils.storage import load_pickle, save_pickle  # noqa: PLC0415

    payload = {'feature': [1, 2, 3], 'label': 'demo'}
    path = save_pickle(payload, str(tmp_path / 'artefact.pkl'))
    assert load_pickle(path) == payload

  def test_a_missing_artefact_raises_the_framework_error(self, tmp_path: pathlib.Path) -> None:
    from forecasting_ml_framework.utils.storage import load_pickle  # noqa: PLC0415

    with pytest.raises(FrameworkError):
      load_pickle(str(tmp_path / 'absent.pkl'))

  def test_a_local_root_is_not_routed_through_the_cloud(self, tmp_path: pathlib.Path) -> None:
    """The address-form invariant: a bare path reaches the host filesystem."""
    from forecasting_ml_framework.utils.storage import get_file_system  # noqa: PLC0415

    filesystem = get_file_system(str(tmp_path / 'x'))
    assert type(filesystem).__name__ == 'LocalFileSystem'


class TestCatalogueCoverage:
  '''Every dataset a pipeline names must exist in the catalogue.'''

  @pytest.fixture(scope='class')
  @classmethod
  def catalog(cls) -> dict:
    """Load the base catalogue.

    Returns:
      The parsed document.
    """
    import yaml  # noqa: PLC0415

    return yaml.safe_load((CONF_ROOT / 'base' / 'catalog.yml').read_text(encoding='utf-8'))

  def test_every_named_entry_exists_or_is_a_transient(self, catalog: dict) -> None:
    """A transient dataset needs no entry; a persistent one missing is a load-time error."""
    from forecasting_ml_framework.pipelines import classification, regression  # noqa: PLC0415
    from forecasting_ml_framework.pipelines.registry import PipelineTopology  # noqa: PLC0415

    suffix = 'params'
    topology = PipelineTopology(suffix=suffix)
    names: set = set()
    for factory in (
      classification.create_fe_training_pipeline,
      classification.create_model_training_pipeline,
      classification.create_fe_scoring_pipeline,
      classification.create_model_scoring_pipeline,
      regression.create_fe_training_pipeline,
      regression.create_model_training_pipeline,
      regression.create_model_scoring_pipeline,
    ):
      for item in factory(topology).nodes:
        for entry in list(item.inputs) + list(item.outputs):
          text = str(entry)
          if text.startswith(('params:', 'sequence_ind', 'parameters')) or '{' in text:
            continue
          names.add(text.replace(f'_{suffix}', '_${suffix}'))

    absent = sorted(name for name in names if name not in catalog)
    assert not absent, f'pipeline datasets absent from the catalogue: {absent}'

  def test_the_catalogue_is_valid_json_ignorable(self, catalog: dict) -> None:
    """Anchors resolve to real entries, so a broken alias fails loudly at load."""
    assert catalog['_spark_bq_data']['type'].endswith('CustomSparkBQDataSet')
    json.dumps(catalog['_spark_bq_data'])
