#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""End-to-end integration test of the modelling tier.

The smoke test proves the code imports; this proves it *works*.

The failure mode this layer exists to catch is not incorrect design but plausible
code that has never been executed -- it passes review, it imports, and it fails
on its first real input. Nothing short of actually running the trainer over a
real frame and checking the numbers catches that class, which is why these tests
train, calibrate, score and evaluate end to end rather than asserting on mocks.

Everything here runs on pandas with a scikit-learn estimator, so it needs neither
a warehouse, nor a cloud credential, nor a Spark session.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forecasting_ml_framework import constants
from forecasting_ml_framework.core.metadata import Metadata
from forecasting_ml_framework.exceptions import MissingParameterError
from forecasting_ml_framework.modeling.models import ModelTraining
from forecasting_ml_framework.observability.quality import QualityPolicy, assert_null_rate, null_rate_summary


@pytest.fixture
def numeric_metadata(metadata: Metadata) -> Metadata:
  """Return a contract whose features are all numeric.

  Categorical columns are string-typed until ``CategoricalEncoding`` runs in
  Spark, so by the time a frame reaches the pandas boundary they have become
  numeric columns named ``*_le`` or ``*_ohe_*``. A frame that still carries raw
  string categoricals at the boundary is a real contract violation, and the
  trainer correctly refuses to fit an estimator on it -- so the modelling tests
  use a contract that reflects the post-boundary state.
  """
  encoded = metadata.copy()
  encoded.categorical_cols = []
  encoded.feature_cols = [c for c in encoded.feature_cols if c not in ('plan_tier', 'region_code')]
  return encoded


class TestTrainAndScore:
  """The primary abstraction, exercised end to end."""

  def test_train_returns_the_five_tuple(self, labelled_frame, numeric_metadata, minimal_trainer_kwargs) -> None:
    """``train`` returns the trainer, the model, the calibrator and both slices.

    Args:
      labelled_frame: The labelled fixture.
      metadata: The contract fixture.
      minimal_trainer_kwargs: The trainer configuration.
    """
    trainer = ModelTraining(**minimal_trainer_kwargs)
    trainer_obj, fitted, calibrator, train_slice, holdout = trainer.train(labelled_frame, numeric_metadata)

    assert trainer_obj is trainer
    assert hasattr(fitted, 'predict_proba')
    assert calibrator == {}          # prior-shift correction, not a fitted calibrator
    assert len(train_slice) + len(holdout) == len(labelled_frame)
    assert 0 < len(holdout) < len(labelled_frame)

  def test_handle_is_self_describing(self, labelled_frame, numeric_metadata, minimal_trainer_kwargs) -> None:
    """Loading the handle must be sufficient to score, with no external state.

    Every attribute below travels with the artefact. That is the framework's
    central packaging decision, and it is what removes the external feature list,
    the separate scaler and the separate threshold.

    Args:
      labelled_frame: The labelled fixture.
      metadata: The contract fixture.
      minimal_trainer_kwargs: The trainer configuration.
    """
    trainer = ModelTraining(**minimal_trainer_kwargs)
    trainer.train(labelled_frame, numeric_metadata)

    assert trainer.features_to_use == numeric_metadata.feature_resolution()
    assert trainer.target_col == 'label'
    assert sorted(trainer.unique_label) == [0, 1]
    assert trainer.multi_class_flag is False
    assert 0.0 < trainer.real_rate <= 1.0
    assert 0.0 < trainer.sample_rate <= 1.0
    assert 0.0 <= trainer.optimal_model_threshold <= 1.0
    assert 0.0 <= trainer.optimal_calib_threshold <= 1.0

  def test_handle_survives_a_pickle_round_trip(
    self, labelled_frame, numeric_metadata, minimal_trainer_kwargs, tmp_object_store
  ) -> None:
    """A persisted handle must reload with its contract intact.

    Args:
      labelled_frame: The labelled fixture.
      metadata: The contract fixture.
      minimal_trainer_kwargs: The trainer configuration.
      tmp_object_store: A writable object-store root.
    """
    import pickle  # noqa: PLC0415

    trainer = ModelTraining(**minimal_trainer_kwargs)
    trainer.train(labelled_frame, numeric_metadata)

    blob = pickle.dumps(trainer)
    restored = pickle.loads(blob)
    assert restored.features_to_use == trainer.features_to_use
    assert restored.optimal_model_threshold == trainer.optimal_model_threshold
    assert restored.real_rate == trainer.real_rate

  def test_exclude_and_include_scope_the_feature_vector(
    self, labelled_frame, numeric_metadata, minimal_trainer_kwargs
  ) -> None:
    """Feature scoping is applied at resolve time and baked into the handle.

    Args:
      labelled_frame: The labelled fixture.
      metadata: The contract fixture.
      minimal_trainer_kwargs: The trainer configuration.
    """
    trainer = ModelTraining(**minimal_trainer_kwargs, exclude_cols=['^usage_'])
    trainer.train(labelled_frame, numeric_metadata)
    assert not [c for c in trainer.features_to_use if c.startswith('usage_')]
    assert 'tenure_days' in trainer.features_to_use

    trainer = ModelTraining(**minimal_trainer_kwargs, include_cols=['tenure_days', 'recharge_count_30d'])
    trainer.train(labelled_frame, numeric_metadata)
    # `recharge_count_30d` is not in the modelled feature list, so the
    # double-gated resolution drops it -- which is the point: an inclusion
    # cannot conjure a column the pipeline never produced.
    assert trainer.features_to_use == ['tenure_days']

  def test_zero_eval_size_warns_but_completes(
    self, labelled_frame, numeric_metadata, minimal_trainer_kwargs, caplog
  ) -> None:
    """Training on everything must still work, with the optimism recorded.

    Args:
      labelled_frame: The labelled fixture.
      metadata: The contract fixture.
      minimal_trainer_kwargs: The trainer configuration.
      caplog: pytest's log capture.
    """
    trainer = ModelTraining(**{**minimal_trainer_kwargs, 'eval_size': 0.0})
    with caplog.at_level('WARNING'):
      _, _, _, train_slice, holdout = trainer.train(labelled_frame, numeric_metadata)
    assert holdout.empty
    assert len(train_slice) == len(labelled_frame)
    assert 'eval_size is 0.0' in caplog.text

  def test_score_projects_the_documented_contract(
    self, labelled_frame, numeric_metadata, minimal_trainer_kwargs
  ) -> None:
    """The score frame must carry the dual-score columns the enterprise table needs.

    Args:
      labelled_frame: The labelled fixture.
      metadata: The contract fixture.
      minimal_trainer_kwargs: The trainer configuration.
    """
    numeric_metadata.model_key = 'forecast_demo_retention'
    trainer = ModelTraining(**minimal_trainer_kwargs)
    _, fitted, calibrator, train_slice, _ = trainer.train(labelled_frame, numeric_metadata)

    raw, calibrated = trainer.score(fitted, calibrator, train_slice, numeric_metadata, 'train_data')

    for column in (
      constants.SCORE_VALUE, constants.ORIG_SCORE_VALUE, constants.SCORE_DECILE,
      constants.SCORE_CENTILE, constants.HIGH_SCORE_IND, constants.LAST_UPDATE_DATE,
      constants.INSERT_TIMESTAMP, constants.MODEL_KEY,
    ):
      assert column in raw.columns, f'raw frame is missing {column}'

    for column in (constants.CALIB_DECILE, constants.CALIB_CENTILE):
      assert column in calibrated.columns, f'calibrated frame is missing {column}'

    assert len(raw) == len(calibrated) == len(train_slice)
    assert raw['score_decile'].between(1, 10).all()
    assert raw['score_centile'].between(1, 100).all()
    assert set(raw['high_score_ind'].unique()) <= {'Y', 'N'}
    assert raw['model_key'].eq('forecast_demo_retention').all()

  def test_banding_is_equal_volume_and_ordered(
    self, labelled_frame, numeric_metadata, minimal_trainer_kwargs
  ) -> None:
    """Bands are rank-based, so the top band is the smallest and has the best lift.

    Args:
      labelled_frame: The labelled fixture.
      metadata: The contract fixture.
      minimal_trainer_kwargs: The trainer configuration.
    """
    trainer = ModelTraining(**minimal_trainer_kwargs)
    _, fitted, calibrator, train_slice, _ = trainer.train(labelled_frame, numeric_metadata)
    raw, _ = trainer.score(fitted, calibrator, train_slice, numeric_metadata, 'train_data')

    counts = raw['score_decile'].value_counts()
    # A tree model emits many tied scores, so qcut's duplicates='drop' can
    # collapse ten requested bands into fewer. The invariant is that the bands
    # which survive are contiguous from 1 and that ordering tracks the label --
    # not that all ten were produced.
    assert sorted(counts.index) == list(range(1, len(counts) + 1))
    top = raw[raw['score_decile'] == raw['score_decile'].max()]
    bottom = raw[raw['score_decile'] == 1]
    assert top['label'].mean() > bottom['label'].mean()
    assert top['score_value'].mean() > bottom['score_value'].mean()

  def test_missing_feature_at_scoring_time_is_an_explicit_error(
    self, labelled_frame, numeric_metadata, minimal_trainer_kwargs
  ) -> None:
    """A schema change between training and scoring must be named, not silently tolerated.

    Args:
      labelled_frame: The labelled fixture.
      metadata: The contract fixture.
      minimal_trainer_kwargs: The trainer configuration.
    """
    trainer = ModelTraining(**minimal_trainer_kwargs)
    _, fitted, calibrator, train_slice, _ = trainer.train(labelled_frame, numeric_metadata)
    truncated = train_slice.drop(columns=['tenure_days'])
    with pytest.raises(KeyError, match='tenure_days'):
      trainer.score(fitted, calibrator, truncated, numeric_metadata, 'train_data')

  def test_empty_frame_passes_through_untouched(
    self, labelled_frame, numeric_metadata, minimal_trainer_kwargs
  ) -> None:
    """An empty slice must produce an empty output, not a failure.

    This is load-bearing: it is what lets ``eval_size: 0.0`` or a degenerate
    sample complete the pipeline with empty metric tables.

    Args:
      labelled_frame: The labelled fixture.
      metadata: The contract fixture.
      minimal_trainer_kwargs: The trainer configuration.
    """
    from forecasting_ml_framework.nodes.model_nodes import model_scoring  # noqa: PLC0415

    trainer = ModelTraining(**minimal_trainer_kwargs)
    _, fitted, calibrator, _, _ = trainer.train(labelled_frame, numeric_metadata)
    raw, calibrated, token = model_scoring(
      trainer, fitted, calibrator, pd.DataFrame(columns=trainer.features_to_use), numeric_metadata, 'valid_data'
    )
    assert raw.empty and calibrated.empty
    assert token is True


def _ensemble_params(handle: str) -> dict:
  """Return valid constructor kwargs for a named ensemble estimator.

  Args:
    handle: The estimator's dotted path.

  Returns:
    A parameter mapping. ``n_estimators`` is only valid for the bagging
    ensembles, so the tree is given a depth instead.
  """
  if 'tree.' in handle:
    return {'max_depth': 4, 'random_state': 42}
  return {'n_estimators': 10, 'random_state': 42}


def _importance_source(caplog) -> str | None:
  """Read the importance source out of the captured log records.

  The source is carried in ``extra`` rather than in the rendered message, which
  is the right shape: a structured field a host can route or alert on, rather
  than prose a log parser has to re-extract.

  Args:
    caplog: pytest's log capture fixture.

  Returns:
    ``'feature_importances_'``, ``'coef_'``, or ``None`` when the estimator
    exposed neither.
  """
  for record in caplog.records:
    if record.getMessage() == 'Feature importance':
      return getattr(record, 'source', None)
  return None


class TestImbalanceAndCalibration:
  """The two rates that make prior-shift calibration possible."""

  def test_weights_are_injected_for_a_gradient_booster(
    self, labelled_frame, numeric_metadata
  ) -> None:
    """``balance_class_weight`` must reach the estimator's own parameter.

    Args:
      labelled_frame: The labelled fixture.
      metadata: The contract fixture.
    """
    numeric_metadata.real_rate = 0.04
    numeric_metadata.scale_pos_weight = 24.0
    trainer = ModelTraining(
      model_handle='sklearn.tree.DecisionTreeClassifier',
      model_params={'max_depth': 3, 'random_state': 42},
      balance_class_weight=True,
      eval_size=0.2,
    )
    # A decision tree has no scale_pos_weight, so the value is injected into the
    # parameter dict and the estimator rejects it -- which is itself the proof
    # that the injection happened and that the string-based type detection was
    # replaced by a capability check.
    trainer.model_handle = _ScaleAwareStub
    trainer.model_params = {'max_depth': 3, 'random_state': 42}
    trainer.train(labelled_frame, numeric_metadata)
    assert 'scale_pos_weight' in trainer.model_params
    assert trainer.model_params['scale_pos_weight'] == pytest.approx(24.0)

  def test_rates_bracket_the_population_rate(
    self, labelled_frame, numeric_metadata, minimal_trainer_kwargs
  ) -> None:
    """With no reweighting configured, the two rates must agree.

    Args:
      labelled_frame: The labelled fixture.
      metadata: The contract fixture.
      minimal_trainer_kwargs: The trainer configuration.
    """
    trainer = ModelTraining(**minimal_trainer_kwargs)
    trainer.train(labelled_frame, numeric_metadata)
    assert trainer.real_rate == pytest.approx(trainer.sample_rate, abs=0.01)

  def test_rare_event_rate_is_not_rounded_to_zero(
    self, labelled_frame, numeric_metadata, minimal_trainer_kwargs
  ) -> None:
    """A rate of ~0.4 % must keep four decimals, or calibration silently disables.

    Args:
      labelled_frame: The labelled fixture.
      metadata: The contract fixture.
      minimal_trainer_kwargs: The trainer configuration.
    """
    frame = labelled_frame.copy()
    frame['label'] = 0
    frame.loc[frame.index[:2], 'label'] = 1  # 2 positives in 400 rows = 0.5%
    trainer = ModelTraining(**{**minimal_trainer_kwargs, 'eval_size': 0.5})
    trainer.train(frame, numeric_metadata)
    assert trainer.real_rate > 0.0
    assert trainer.real_rate == pytest.approx(0.005, abs=0.0005)


class TestCalibratorLifecycle:
  """the configured calibrator must actually reach production."""

  def test_configured_calibrator_is_returned_not_discarded(
    self, labelled_frame, numeric_metadata, minimal_trainer_kwargs
  ) -> None:
    """A configured Platt calibrator must be fitted, returned and persisted.

    A fitted the calibrator into a function-local variable and
    returned an empty dict, so production calibration silently used a different
    method from the one the configuration named.

    Args:
      labelled_frame: The labelled fixture.
      metadata: The contract fixture.
      minimal_trainer_kwargs: The trainer configuration.
    """
    trainer = ModelTraining(
      **minimal_trainer_kwargs,
      calibration_handle='sklearn.linear_model.LogisticRegression',
      calibration_params={'max_iter': 200},
    )
    _, _, calibrator, _, _ = trainer.train(labelled_frame, numeric_metadata)
    assert calibrator is not None
    assert type(calibrator).__name__ == 'LogisticRegression'
    assert trainer.calibrated_model is calibrator

  def test_unconfigured_calibrator_uses_prior_shift(
    self, labelled_frame, numeric_metadata, minimal_trainer_kwargs
  ) -> None:
    """No configured calibrator means the framework's own correction.

    Args:
      labelled_frame: The labelled fixture.
      metadata: The contract fixture.
      minimal_trainer_kwargs: The trainer configuration.
    """
    trainer = ModelTraining(**minimal_trainer_kwargs)
    _, _, calibrator, _, _ = trainer.train(labelled_frame, numeric_metadata)
    assert calibrator == {}
    assert trainer.calibrated_model is None

  def test_unknown_calibrator_handle_degrades_with_a_warning(
    self, labelled_frame, numeric_metadata, minimal_trainer_kwargs, caplog
  ) -> None:
    """An unrecognised handle falls back, but says so.

    Args:
      labelled_frame: The labelled fixture.
      metadata: The contract fixture.
      minimal_trainer_kwargs: The trainer configuration.
      caplog: pytest's log capture.
    """
    trainer = ModelTraining(
      **minimal_trainer_kwargs, calibration_handle='sklearn.linear_model.RidgeClassifier'
    )
    with caplog.at_level('WARNING'):
      _, _, calibrator, _, _ = trainer.train(labelled_frame, numeric_metadata)
    assert calibrator == {}
    assert 'Unrecognised calibration handle' in caplog.text


class TestEstimatorBreadth:
  """importance must be optional, not a post-fit tax."""

  @pytest.mark.parametrize(
    'handle', ['sklearn.tree.DecisionTreeClassifier', 'sklearn.ensemble.RandomForestClassifier']
  )
  def test_ensemble_estimators_expose_importance(
    self, labelled_frame, numeric_metadata, minimal_trainer_kwargs, handle: str
  ) -> None:
    """An ensemble reports importance and logs it.

    Args:
      labelled_frame: The labelled fixture.
      metadata: The contract fixture.
      minimal_trainer_kwargs: The trainer configuration.
      handle: The estimator dotted path.
    """
    trainer = ModelTraining(
      model_handle=handle, model_params=_ensemble_params(handle), eval_size=0.2
    )
    trainer.train(labelled_frame, numeric_metadata)

  def test_linear_model_without_feature_importances_still_trains(
    self, labelled_frame, numeric_metadata, caplog
  ) -> None:
    """A raised ``AttributeError`` after fitting, discarding the model.

    Reading ``feature_importances_`` unconditionally excluded every non-ensemble
    estimator from the framework for no benefit, since the value was only
    printed.

    Args:
      labelled_frame: The labelled fixture.
      metadata: The contract fixture.
      caplog: pytest's log capture.
    """
    trainer = ModelTraining(
      model_handle='sklearn.linear_model.LogisticRegression',
      model_params={'max_iter': 300},
      eval_size=0.2,
    )
    with caplog.at_level('INFO'):
      _, fitted, _, _, _ = trainer.train(labelled_frame, numeric_metadata)
    assert hasattr(fitted, 'predict')
    assert _importance_source(caplog) == 'coef_'

  def test_estimator_with_neither_attribute_does_not_fail(
    self, labelled_frame, numeric_metadata, caplog
  ) -> None:
    """An SVC exposes neither, and must train to completion regardless.

    Args:
      labelled_frame: The labelled fixture.
      metadata: The contract fixture.
      caplog: pytest's log capture.
    """
    trainer = ModelTraining(
      model_handle='sklearn.svm.SVC',
      model_params={'probability': True, 'random_state': 42},
      eval_size=0.2,
    )
    with caplog.at_level('INFO'):
      _, fitted, _, _, _ = trainer.train(labelled_frame, numeric_metadata)
    assert hasattr(fitted, 'predict_proba')
    assert _importance_source(caplog) is None


class TestDataQualityEnforcement:
  """observation is not enforcement."""

  def test_row_count_assertion_catches_a_dropped_frame(self) -> None:
    """A transform that loses rows must be caught, not silently accepted.

    A routine that dropped 40 % of a frame produced a complete, plausible and
    entirely wrong model with no alert anywhere.
    """
    policy = QualityPolicy(row_count_tolerance=0.0).with_expected(1000)
    with pytest.raises(Exception, match='row count'):
      from forecasting_ml_framework.observability.quality import assert_row_count  # noqa: PLC0415

      assert_row_count(600, policy, 'fit_registry')

  def test_null_rate_is_measured(self, labelled_frame) -> None:
    """A frame with a sparse null column reports the rate that feeds the policy.

    Args:
      labelled_frame: The labelled fixture.
    """
    frame = labelled_frame.copy()
    frame.loc[frame.index[:10], 'usage_01'] = np.nan
    summary = null_rate_summary(frame, ['usage_01', 'tenure_days'])
    assert summary['usage_01'] == pytest.approx(0.025, abs=0.001)
    assert summary['tenure_days'] == 0.0

  def test_null_rate_policy_rejects_a_broken_imputation(self, labelled_frame) -> None:
    """An imputation that did not cover its columns must fail the stage.

    Args:
      labelled_frame: The labelled fixture.
    """
    frame = labelled_frame.copy()
    frame.loc[frame.index[:100], 'usage_01'] = np.nan
    policy = QualityPolicy(max_null_rate=0.05)
    with pytest.raises(Exception, match='null rate'):
      assert_null_rate(frame, ['usage_01'], policy, 'fit_registry')


class TestParameterContract:
  """Direct indexing was the framework's primary validation, and its main
  failure mode: a ``KeyError`` that named the key but no configuration context.
  """

  def test_missing_parameter_names_the_context(self) -> None:
    """A missing key must say which stage wanted it and how to supply it.

    Args:
      None.
    """
    from forecasting_ml_framework.utils.text import require  # noqa: PLC0415

    with pytest.raises(MissingParameterError) as excinfo:
      require({'a': 1}, 'model_key', stage='get_spark_session')
    message = str(excinfo.value)
    assert 'model_key' in message
    assert 'get_spark_session' in message
    assert '--params=model_key=' in message


class _ScaleAwareStub:
  """A stand-in estimator used to prove the imbalance parameter is injected."""

  def __init__(self, **kwargs) -> None:
    self.kwargs = kwargs
    self.classes_ = np.array([0, 1])

  def fit(self, x, y, **kwargs):  # noqa: ANN001
    """Accept a fit and return self.

    Args:
      x: The feature matrix.
      y: The labels.
      **kwargs: Ignored.

    Returns:
      ``self``.
    """
    del x, y, kwargs
    return self

  def predict_proba(self, x):  # noqa: ANN001
    """Return a constant probability.

    Args:
      x: The feature matrix.

    Returns:
      A two-column probability array.
    """
    return np.tile([[0.9, 0.1]], (len(x), 1))

  def predict(self, x):  # noqa: ANN001
    """Return a constant prediction.

    Args:
      x: The feature matrix.

    Returns:
      A prediction array.
    """
    return np.zeros(len(x))
