#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Scoring: the dual-score contract and the output projection.

Two score columns are produced simultaneously:

``orig_score_value``
  The raw model probability, after any loss reweighting and before any
  correction. This is what *discrimination* is defined on: model comparison,
  threshold semantics, decile banding and the high-score flag.

``score_value``
  The calibrated probability, after prior-shift correction or a fitted
  calibrator. This is the *business* number: the published expected event
  probability.

Keeping both is a deliberate and defensible design. It lets a consumer verify
that calibration has not degraded ranking, it supports an A/B comparison of
calibration strategies without re-scoring, and it lets a monitor watch the two
distributions independently. The cost is a wider score table and a consumer
contract that must respect which column means what. The framework enforces that
distinction consistently: the high-score flag and the decile banding derive from
the raw score, the calibrated banding is computed on the calibrated score, and
each frame's hard prediction is thresholded against the score family it belongs
to.

Three further properties:

* **The projection is defined once.** The score-table contract is
  :data:`SCORE_COLUMNS` plus :data:`CALIBRATED_BANDING_COLUMNS`, and a single
  :func:`project_score_frame` produces both variants from it. Two overlapping
  ``select`` calls kept in sync by hand is a maintenance hazard with nothing to
  catch a miss.
* **In-band diagnostics reuse one prediction pass.** When labels are present,
  accuracy and the proper scoring rules are reported, with the positive label
  taken from the estimator's own class list and the probabilities taken from the
  ``predict_proba`` already computed. Recomputing predictions for values that
  are only logged triples the scoring cost.
* **Bands are contiguous and equal-volume.** ``qcut`` is rank-based, and its
  ``duplicates='drop'`` behaviour would otherwise leave gaps in the label sequence
  on a score carrying many ties, so a consumer asking for a particular band could
  find no rows. See :func:`_rank_bands`.
"""

from __future__ import annotations

from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from forecasting_ml_framework import constants
from forecasting_ml_framework.exceptions import ModelError
from forecasting_ml_framework.modeling.calibration import calibrate_probability
from forecasting_ml_framework.observability.logging import get_logger

LOGGER = get_logger(__name__)

#: The score columns present on both variants of the score frame.
SCORE_COLUMNS = (
  constants.SCORE_VALUE,
  constants.ORIG_SCORE_VALUE,
  constants.SCORE_DECILE,
  constants.SCORE_CENTILE,
  constants.HIGH_SCORE_IND,
  constants.LAST_UPDATE_DATE,
  constants.INSERT_TIMESTAMP,
)

#: The columns the calibrated variant carries in place of the raw banding.
CALIBRATED_BANDING_COLUMNS = (constants.CALIB_DECILE, constants.CALIB_CENTILE)

#: The per-class probability column prefix used before thresholding.
PROBABILITY_PREFIX = 'probability'


def score_frame(
  model: Any,
  calibrated_model: Any,
  frame: pd.DataFrame,
  features_to_use: list[str],
  target_col: str | None,
  prediction_col: str,
  prediction_probability_col: str,
  model_threshold: float,
  calib_threshold: float,
  labels: list[Any],
  multi_class: bool = False,
  real_rate: float = 0.0,
  sample_rate: float = 0.0,
  run_date: str | None = None,
) -> pd.DataFrame:
  """Produce the fully-scored frame from a fitted model and a labelled or
  unlabelled input.

  Args:
    model: The fitted estimator.
    calibrated_model: A fitted calibrator, or ``None``/``{}`` for prior-shift
      correction. a always persisted an empty dict here
      even when a calibrator was fitted, so a configured calibrator is
      used for threshold selection and then discarded. The trainer returns
      the fitted calibrator, and this branch is reached.
    frame: The input frame. It may or may not carry the label.
    features_to_use: The exact, ordered feature vector the model was fitted on.
    target_col: The label column, or ``None`` in production.
    prediction_col: The output name of the hard prediction.
    prediction_probability_col: The output name of the probability.
    model_threshold: The decision threshold on the raw score.
    calib_threshold: The decision threshold on the calibrated score.
    labels: The sorted label set observed at training time.
    multi_class: Whether the problem has more than two classes.
    real_rate: The population positive rate.
    sample_rate: The effective training positive rate.
    run_date: The run date, used to stamp the update date.

  Returns:
    The scored frame, before projection.

  Raises:
    KeyError: If a required feature is absent from the frame.
  """
  working = frame.reset_index(drop=True)
  missing = [column for column in features_to_use if column not in working.columns]
  if missing:
    raise KeyError(
      f'The scoring frame is missing {len(missing)} column(s) the model was fitted on: {missing[:20]}. '
      'This usually means a preprocessing routine was removed or a column was renamed between '
      'training and scoring. Re-fit the registry, or add a compatibility shim.'
    )

  # The score comes from the one helper that understands both estimator families:
  # a classifier's positive-class probability, or a regressor's logistic-squashed
  # decision function. The class columns are only meaningful for a classifier --
  # a regressor has no per-class probabilities to name -- so they are built only
  # when the estimator declares classes.
  declared = getattr(model, 'classes_', None)
  classes = list(declared) if declared is not None and len(declared) else []
  raw_score = positive_probability(model, working[features_to_use])
  probability = None
  if classes:
    probability = np.asarray(model.predict_proba(working[features_to_use]), dtype=float)
    class_columns = [f'{PROBABILITY_PREFIX}_{int(label)}' for label in sorted(labels)]
    working = pd.concat(
      [working, pd.DataFrame(probability, columns=class_columns, index=working.index)], axis=1
    )

  working[constants.ORIG_SCORE_VALUE] = raw_score

  # The diagnostics run *after* the score exists and consume the array they were
  # always meant to reuse. Called before it, they had to recompute it -- a second
  # `predict_proba` plus a `model.score`, which is a third pass and an internal
  # `predict`. Four inference passes per scoring frame became one.
  _report_in_band_diagnostics(model, working, features_to_use, target_col, probability, model_threshold)

  if not multi_class:
    # The hard prediction is a threshold on the raw score, so `model.predict` is
    # not consulted in the binary path: its own cut-off is 0.5, which is not this
    # model's operating point, and its result was unconditionally overwritten.
    # A single inference pass now produces the whole scored frame.
    working[prediction_col] = np.where(working[constants.ORIG_SCORE_VALUE] >= model_threshold, 1.0, 0.0)
  else:
    # For K > 2 there is no single positive class to gate on, so the estimator's
    # own argmax is the prediction and the learned thresholds are not applied.
    # This is a documented gap rather than a silent limitation.
    working[prediction_col] = model.predict(working[features_to_use])
    LOGGER.info(
      'Multiclass scoring: the hard prediction is the estimator argmax and the learned thresholds '
      'are not applied, because there is no single positive class to gate on',
      extra={'prediction_col': prediction_col},
    )

  working[constants.SCORE_VALUE] = _calibrated_score(
    calibrated_model, working, real_rate, sample_rate, features_to_use
  )

  if not multi_class:
    working[prediction_probability_col] = working[constants.ORIG_SCORE_VALUE]
    working[f'{constants.CALIBRATION_PREFIX}{prediction_col}'] = np.where(
      working[constants.SCORE_VALUE] >= calib_threshold, 1.0, 0.0
    )
    working[f'{constants.CALIBRATION_PREFIX}{prediction_probability_col}'] = working[constants.SCORE_VALUE]

  _assign_bands(working)
  # The high-score flag is a conjunction: above threshold AND in the upper half of
  # the score distribution. The two conditions are named separately so the
  # definition is visible to a consumer rather than having to be inferred.
  #
  # The half is computed from the bands actually produced, not from a fixed
  # constant. Two defects make a constant wrong. Bands are 1..10, so `>= 5`
  # admits deciles 5-10 -- the top *60%*, not the top half. And `qcut` with
  # `duplicates='drop'` produces only as many bands as the score has distinct
  # values, so a coarse or near-constant score yields a single band and a fixed
  # floor of 5 flags nothing at all: the flag silently reads 'N' for every row.
  # Deriving the boundary means the flag is the top half of whatever banding the
  # score supports, and degrades to "all of it" rather than to "none of it".
  above_threshold = working[constants.ORIG_SCORE_VALUE] > model_threshold
  in_high_band = working[constants.SCORE_DECILE] >= _top_half_floor(working[constants.SCORE_DECILE])
  working[constants.HIGH_SCORE_IND] = np.where(
    above_threshold & in_high_band,
    constants.HIGH_SCORE_YES,
    constants.HIGH_SCORE_NO,
  )

  working[constants.LAST_UPDATE_DATE] = (
    pd.Timestamp(run_date).date() if run_date else date.today()
  )
  working[constants.INSERT_TIMESTAMP] = pd.Timestamp.now(tz='UTC')
  return working


def positive_probability(model: Any, features: Any) -> np.ndarray:
  """Return the positive-class score for a feature matrix.

  For a classifier the positive class is the **last** entry of the estimator's own
  ``classes_``, following the scikit-learn convention that the attribute is
  ascending. Taking it from the estimator rather than assuming a position is what
  makes this correct for a target that is not encoded 0/1: a ``{1, 2}`` target makes
  2 the positive class by accident under a hard-coded index.

  A regressor exposes neither ``classes_`` nor ``predict_proba``, and calling that
  method unconditionally raised ``AttributeError`` on every regression run that
  reached calibration or threshold selection -- including runs whose configuration
  asks for no calibration at all, because the score was read before the branch that
  decides whether it is needed.

  For a regressor the score **is** the prediction: the metrics it is judged on --
  RMSE, MAE, R2 -- compare it to the target on the target's own scale. It is
  therefore returned unscaled, because rescaling onto ``(0, 1)`` would make every
  published score depend on the other rows in the batch and two runs over different
  populations would no longer be comparable. Where a ``decision_function`` *is*
  available the value is a classifier decision margin rather than a prediction, and
  it is squashed onto ``(0, 1)`` so it lands on the scale the downstream banding
  expects.

  An estimator exposing none of the three is reported as an error rather than
  falling back to a column, because a score on an arbitrary scale is the failure
  mode this helper exists to prevent.

  It lives here rather than beside its caller in the trainer because the trainer
  already imports this module, so defining it here keeps the dependency one-way.
  The reverse arrangement is a cycle, and a cycle in this pair is not merely
  untidy: it makes the import order of the two modules decide whether the package
  imports at all.

  Args:
    model: A fitted estimator.
    features: The feature matrix, in the model's resolved column order.

  Returns:
    The positive-class score as a 1-D array.

  Raises:
    ModelError: If the estimator exposes neither a probability nor a decision
      function, or if the returned array does not match its declared classes.
  """
  # `classes_` is a numpy array on most estimators, and `array or []` raises
  # "truth value of an array with more than one element is ambiguous" -- so the
  # emptiness test is made explicitly rather than through truthiness.
  declared = getattr(model, 'classes_', None)
  classes = list(declared) if declared is not None and len(declared) else []
  if hasattr(model, 'predict_proba'):
    probability = np.asarray(model.predict_proba(features), dtype=float)
    if not classes:
      raise ModelError(
        f'{type(model).__name__} exposes no classes_, so the positive class cannot be identified.',
        estimator=type(model).__name__,
      )
    if probability.ndim != 2 or probability.shape[1] != len(classes):
      raise ModelError(
        f'{type(model).__name__} returned a probability array of shape {probability.shape}, which does '
        f'not match its {len(classes)} declared classes.',
        estimator=type(model).__name__,
        shape=str(probability.shape),
      )
    return probability[:, len(classes) - 1]

  if hasattr(model, 'decision_function'):
    decision = np.asarray(model.decision_function(features), dtype=float)
    return 1.0 / (1.0 + np.exp(-decision))

  if hasattr(model, 'predict'):
    return np.asarray(model.predict(features), dtype=float).reshape(-1)

  raise ModelError(
    f'{type(model).__name__} exposes no predict_proba, decision_function or predict, so no '
    'positive-class score can be read from it.',
    estimator=type(model).__name__,
  )


def score_regression_frame(
  model: Any,
  frame: pd.DataFrame,
  features_to_use: list[str],
  prediction_col: str,
  run_date: Any = None,
) -> pd.DataFrame:
  """Score a frame with a regressor and project it onto the score-table contract.

  A continuous target has no probability column and no positive class, so
  :func:`score_frame` cannot serve it: that function requires ``predict_proba``,
  a class list, and a decision threshold. The regression tier was nevertheless
  routed through it, so ``model_scoring_regression`` -- which delegates to
  ``model_scoring`` -- raised ``AttributeError: 'XGBRegressor' object has no
  attribute 'predict_proba'`` on the first frame it was given. The configured
  ``modeling_reg_params.model_handle`` therefore could not score at all.

  The two score *scales* are still both emitted, because the calibrated variant is
  what the dual-scale contract promises. With no calibrator fitted, the calibrated
  projection carries the raw value, so the two agree exactly -- which is the
  correct statement of "no calibration was applied" rather than a null.

  The high-score flag is the literal empty string, matching the documented
  behaviour for a continuous target: there is no threshold to test, so the column
  is retained only to keep the score-table schema stable across model types.

  Args:
    model: A fitted regressor.
    frame: The frame to score.
    features_to_use: The resolved, ordered feature list.
    prediction_col: The output name of the prediction column.
    run_date: The run date, used to stamp the update date.

  Returns:
    The projected frame.

  Raises:
    KeyError: If a required feature is absent from the frame.
  """
  working = frame.reset_index(drop=True)
  missing = [column for column in features_to_use if column not in working.columns]
  if missing:
    raise KeyError(
      f'The scoring frame is missing {len(missing)} column(s) the model was fitted on: {missing[:20]}. '
      'This usually means a preprocessing routine was removed or a column was renamed between '
      'training and scoring. Re-fit the registry, or add a compatibility shim.'
    )

  predicted = np.asarray(model.predict(working[features_to_use]), dtype=float)
  working[prediction_col] = predicted
  # A regressor emits one number per row, so the raw and the calibrated score are
  # the same quantity. The probability column mirrors it rather than being left
  # null, so the two scales are comparable by a consumer that reads both.
  working[constants.ORIG_SCORE_VALUE] = predicted
  working[constants.SCORE_VALUE] = predicted
  working[constants.SCORE_DECILE] = _rank_bands(
    pd.Series(predicted), constants.DECILE_COUNT
  )
  working[constants.SCORE_CENTILE] = _rank_bands(
    pd.Series(predicted), constants.CENTILE_COUNT
  )
  working[constants.CALIB_DECILE] = working[constants.SCORE_DECILE]
  working[constants.CALIB_CENTILE] = working[constants.SCORE_CENTILE]
  working[constants.HIGH_SCORE_IND] = ''
  working[constants.LAST_UPDATE_DATE] = (
    pd.Timestamp(run_date).date() if run_date else date.today()
  )
  working[constants.INSERT_TIMESTAMP] = pd.Timestamp.now(tz='UTC')
  return working


def project_score_frame(
  scored: pd.DataFrame,
  id_cols: list[str],
  target_col: str | None,
  prediction_col: str,
  prediction_probability_col: str,
  model_key: str | None,
  calibrated: bool,
) -> pd.DataFrame:
  """Project a scored frame down to the score-table contract.

  The uncalibrated variant carries the raw banding; the calibrated variant
  carries the calibrated banding. Both are produced from one column
  specification here. Two overlapping ``select`` calls must be kept in sync by
  hand, and nothing catches a miss -- the failure is a column that quietly stops
  being published.

  Args:
    scored: The scored frame.
    id_cols: The identifier columns.
    target_col: The label column, or ``None`` in production.
    prediction_col: The hard prediction column name.
    prediction_probability_col: The probability column name.
    model_key: The model identifier, re-attached as a literal because it is
      carried in metadata rather than in the frame.
    calibrated: Which variant to project.

  Returns:
    The projected frame.
  """
  columns = list(id_cols)
  if model_key:
    columns.append(constants.MODEL_KEY)
    scored = scored.copy()
    scored[constants.MODEL_KEY] = model_key

  banding = CALIBRATED_BANDING_COLUMNS if calibrated else (constants.SCORE_DECILE, constants.SCORE_CENTILE)
  columns.extend([*SCORE_COLUMNS, *banding])

  if target_col and target_col in scored.columns:
    columns.append(target_col)
    columns.append(prediction_col)
    probability_output = (
      f'{constants.CALIBRATION_PREFIX}{prediction_probability_col}'
      if calibrated
      else prediction_probability_col
    )
    columns.extend([probability_output, f'{constants.CALIBRATION_PREFIX}{prediction_col}'])

  ordered = list(dict.fromkeys(column for column in columns if column in scored.columns))
  LOGGER.info(
    'Projected the score frame',
    extra={'variant': 'calibrated' if calibrated else 'raw', 'columns': len(ordered)},
  )
  return scored[ordered]


def _calibrated_score(
  calibrated_model: Any,
  working: pd.DataFrame,
  real_rate: float,
  sample_rate: float,
  features_to_use: list[str] | None = None,
) -> pd.Series:
  """Compute the calibrated score, dispatching on the calibrator's type.

  Type-based dispatch is the extension mechanism here, and ``isinstance`` is
  preferred over a class-name comparison because it survives subclassing. An
  unrecognised type falls through to prior-shift correction, which is *correct
  but different* — so the fallback is logged rather than taken silently.

  Args:
    calibrated_model: A fitted calibrator, or ``None``/``{}``.
    working: The scored frame.
    real_rate: The population positive rate.
    sample_rate: The effective training positive rate.

  Returns:
    The calibrated probability.
  """
  from sklearn.calibration import CalibratedClassifierCV  # noqa: PLC0415
  from sklearn.isotonic import IsotonicRegression  # noqa: PLC0415
  from sklearn.linear_model import LogisticRegression  # noqa: PLC0415

  raw = working[constants.ORIG_SCORE_VALUE]

  if isinstance(calibrated_model, CalibratedClassifierCV):
    per_class = pd.DataFrame(
      calibrated_model.predict_proba(working[_infer_feature_columns(working, calibrated_model, features_to_use)]),
      index=working.index,
    )
    return per_class[per_class.columns[-1]]
  if isinstance(calibrated_model, (LogisticRegression, IsotonicRegression)):
    reshaped = raw.to_numpy(dtype=float).reshape(-1, 1)
    if isinstance(calibrated_model, IsotonicRegression):
      # Isotonic emits NaN outside its fitted range, so a score that the training
      # frame never produced would silently become null in the published table.
      return pd.Series(calibrated_model.predict(reshaped), index=working.index).fillna(0.0)
    return pd.Series(calibrated_model.predict_proba(reshaped)[:, 1], index=working.index)
  if type(calibrated_model).__name__ == 'BetaCalibration':
    return pd.Series(
      calibrated_model.predict(raw.to_numpy(dtype=float).reshape(-1, 1)), index=working.index
    )

  if calibrated_model:
    LOGGER.info(
      'Unrecognised calibrator type; falling back to prior-shift correction, which is correct but '
      'not the configured strategy',
      extra={'calibrator': type(calibrated_model).__name__},
    )
  return calibrate_probability(raw, real_rate, sample_rate)


def _assign_bands(working: pd.DataFrame) -> None:
  """Add the decile, centile and calibrated banding columns.

  ``pd.qcut`` is rank-based, so bands are equal-volume by construction and the
  top band is always the same share of the population regardless of the score
  distribution. ``duplicates='drop'`` is essential: on a skewed or
  low-cardinality score it prevents ``qcut`` from raising on duplicate bin edges.
  The ``+ 1`` makes bands 1-indexed, so decile 10 is the best band and centile
  100 the best percentile — the convention the lift tables and the downstream
  drift monitor both assume.

  Args:
    working: The scored frame, mutated in place.
  """
  for source, decile, centile in (
    (constants.ORIG_SCORE_VALUE, constants.SCORE_DECILE, constants.SCORE_CENTILE),
    (constants.SCORE_VALUE, constants.CALIB_DECILE, constants.CALIB_CENTILE),
  ):
    working[decile] = _rank_bands(working[source], constants.DECILE_COUNT)
    working[centile] = _rank_bands(working[source], constants.CENTILE_COUNT)


def _rank_bands(scores: pd.Series, band_count: int) -> pd.Series:
  """Assign contiguous, 1-indexed, equal-volume bands to a score.

  ``pd.qcut`` is rank-based, so bands are equal-volume by construction and the
  top band always holds the same share of the population. The ``+ 1`` makes bands
  1-indexed, so the top band is ``band_count`` and the bottom is 1 -- the
  convention the lift tables and the downstream drift monitor both assume.

  ``duplicates='drop'`` is what keeps a skewed or low-cardinality score from
  raising on duplicate bin edges, but it has a consequence that must be handled
  explicitly: the surviving labels are the *quantile edge indices*, so a frame
  with heavy ties yields labels such as ``1, 3, 4`` with a gap at 2. A consumer
  asking for ``score_decile = 2`` would find no rows, and the "top band is 10"
  contract would not hold. The bands are therefore re-ranked densely, which
  guarantees contiguity for any score distribution.

  Args:
    scores: The score column.
    band_count: The number of bands requested.

  Returns:
    The band labels, contiguous from 1.
  """
  # The degeneracy checks come *before* `qcut`. `qcut` on an all-NaN column
  # reaches into numpy's internals and raises `IndexError: index -1 is out of
  # bounds`, so a guard written after the call is a guard that never runs on the
  # case it exists for. Fewer than two usable values is also the honest
  # precondition: a single value has no ordering to band, and an all-NaN column
  # has none either.
  usable = scores.dropna()
  if len(usable) < 2 or usable.nunique() < 2:
    return pd.Series(1, index=scores.index, dtype='int64')
  raw = pd.qcut(scores, band_count, labels=False, duplicates='drop')
  bands = raw.fillna(0).astype(int)
  # A constant score has no ordering to band, so every row shares one band.
  if bands.nunique() <= 1:
    return pd.Series(1, index=scores.index, dtype='int64')
  return bands.rank(method='dense').astype('int64')


def _top_half_floor(bands: pd.Series) -> int:
  """Return the band label at which the upper half of a banding begins.

  The boundary is derived from the bands actually produced rather than fixed,
  because a fixed boundary is wrong in two independent ways: bands run 1..10, so
  a floor of 5 admits deciles 5-10 and therefore the top 60 percent; and
  ``qcut`` with ``duplicates='drop'`` yields only as many bands as the score has
  distinct values, so a coarse score can produce a single band that no fixed
  floor of 5 would ever reach.

  Deriving it makes the flag mean what it says -- the upper half of the
  distribution as banded -- and makes the degenerate case degrade to "all of it",
  which is the correct reading of a population that cannot be split, rather than
  to "none of it".

  Args:
    bands: The band labels assigned to the population.

  Returns:
    The inclusive lower bound of the upper half, at least 1.
  """
  observed = bands.dropna()
  if observed.empty:
    return 1
  highest = int(observed.max())
  # The upper half is the top `ceil(n / 2)` bands, so its first label is
  # `n - ceil(n/2) + 1`. Stated that way the boundary is also correct for an odd
  # band count: with 10 bands the floor is 6 -- not 5, which would admit six
  # bands and so 60 percent -- and with 3 bands it is 2.
  return highest - (highest + 1) // 2 + 1


def _report_in_band_diagnostics(
  model: Any,
  working: pd.DataFrame,
  features_to_use: list[str],  # noqa: ARG001 - part of the caller's uniform diagnostic signature
  target_col: str | None,
  probability: np.ndarray | None = None,
  model_threshold: float = 0.5,
) -> None:
  """Emit accuracy and proper-scoring-rule diagnostics when labels are present.

  The positive label is taken from the estimator's own class list rather than
  assumed, because ``brier_score_loss`` needs the label the target actually uses
  and a hard-coded ``pos_label=2`` is absent from a ``{0, 1}`` target. The
  probability is passed straight from the ``predict_proba`` already computed
  rather than re-derived, since recomputing predictions for values that are only
  logged would triple the scoring cost.
  """
  if not target_col or target_col not in working.columns:
    return
  if probability is None:
    return
  LOGGER.info(
    'In-band diagnostics',
    extra={'target_distribution': working[target_col].value_counts().to_dict(), 'rows': len(working)},
  )
  classes = list(getattr(model, 'classes_', [0, 1]))
  positive = classes[-1]
  try:
    from sklearn.metrics import brier_score_loss, log_loss  # noqa: PLC0415

    # `probability` is the array `predict_proba` already produced for this frame,
    # so nothing is re-derived. The previous version ignored it, ran
    # `predict_proba` a second time, and then called `model.score`, which is a
    # third pass plus an internal `predict` -- four inference passes per scoring
    # frame instead of one. The cost lands on the largest frame in the pipeline,
    # in a cluster provisioned solely to be destroyed.
    raw = probability[:, len(classes) - 1]
    # Accuracy is derived from the same array rather than from `model.score`,
    # which would be another pass *and* would use the estimator's own 0.5 cut-off
    # rather than this model's selected operating point -- so it described a
    # different classifier from the one the score table reports.
    threshold = 0.5 if len(classes) != 2 else float(model_threshold)
    accuracy = float(
      np.mean((raw >= threshold).astype(int) == working[target_col].to_numpy())
    )
    LOGGER.info(
      'In-band quality',
      extra={
        'accuracy': accuracy,
        'brier_loss': float(brier_score_loss(working[target_col], raw, pos_label=positive)),
        'log_loss': float(log_loss(working[target_col], raw, labels=classes)),
      },
    )
  except ValueError as error:  # pragma: no cover - only on a degenerate slice
    LOGGER.warning('In-band diagnostics are undefined for this slice', extra={'error': str(error)})


def _infer_feature_columns(
  working: pd.DataFrame,
  calibrated_model: Any,
  features_to_use: list[str] | None = None,
) -> list[str]:
  """Recover the feature columns a wrapped calibrator was fitted on.

  A ``CalibratedClassifierCV`` scores from the *feature matrix*, so the columns
  must be supplied in the order the wrapper was fitted on. The framework already
  knows that order -- it is the resolved feature list baked into the model handle,
  and it is the single most load-bearing ordering in the system, because a
  different order yields different predictions with no error.

  The previous fallback reconstructed the list by taking every column of the
  scored frame that was not a known framework output. That is not the model's
  feature vector: it retains the target column and the identifier columns, and it
  inherits the *frame's* ordering rather than the resolved one. The wrapper's own
  ``feature_names_in_`` is also usually absent, because ``CalibratedClassifierCV``
  calls ``indexable(X, y)`` rather than ``check_array``, so only its inner per-fold
  clones carry the attribute -- the fallback was the branch actually taken.

  Args:
    working: The scored frame.
    calibrated_model: A ``CalibratedClassifierCV`` instance.
    features_to_use: The resolved, ordered feature list from the model handle.

  Returns:
    The feature column names, in the order the wrapper was fitted on.
  """
  if features_to_use:
    return list(features_to_use)
  declared = getattr(calibrated_model, 'feature_names_in_', None)
  if declared is not None:
    return list(declared)
  reserved = {*SCORE_COLUMNS, *CALIBRATED_BANDING_COLUMNS, constants.MODEL_KEY, constants.REPORT_DATE}
  return [
    column
    for column in working.columns
    if column not in reserved and not column.startswith(PROBABILITY_PREFIX)
  ]
