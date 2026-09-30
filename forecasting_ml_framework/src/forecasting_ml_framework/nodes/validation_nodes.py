#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Champion/challenger validation.

This is the most consequential business logic in the framework: it decides
whether a retrained model may replace the one serving production traffic. Two
gates apply, in this order.

**Gate 1 — the non-regression tolerance.** The challenger is scored on the same
held-out slice as the incumbent, at each model's own operating point, using each
model's own feature list, and is accepted if its relative deviation is within
tolerance. A challenger that is *worse but by less than the tolerance* is
promoted. That is deliberate and it is correct for this problem: retrained
retention models on rolling windows are routinely statistically
indistinguishable, so an improvement gate would reject near-ties on sampling
noise and stall the model lifecycle indefinitely.

**Gate 2 — temporal validity.** A challenger may only supersede an incumbent if
it was evaluated against a strictly later observation window. This is the key
insight of the design: *a model may only supersede a better-informed one if it
was itself better informed.* A model trained on stale evidence cannot displace a
model that saw more outcomes, regardless of its score.

**
raising a ``ValueError`` and catching it in the same block, printing the message
and continuing on the default. Because ``eval_metric`` is absent from the
configured model program, the default is always in force, and a typo in a run
parameter would silently change the metric a production promotion is decided on,
with the only evidence a line in a cluster log. An unrecognised metric now fails
the promotion; the default applies only when no metric was configured at all.

**
reported the top band's value under a different band's metric name. That changes
a *reported number*, not merely whether a task runs, so it is never silent: the
substitution is either rejected or, where an operator explicitly opts in,
recorded in the audit trail.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from forecasting_ml_framework import constants
from forecasting_ml_framework.exceptions import MetricBandEmptyError, RegistryEntryMissingError
from forecasting_ml_framework.modeling.metrics import (
  band_metrics,
  evaluate_acceptance,
  parse_metric,
  regression_band_metrics,
  select_band,
)
from forecasting_ml_framework.modeling.promotion import PromotionDecision, apply_promotion, decide_promotion
from forecasting_ml_framework.observability.logging import get_logger

LOGGER = get_logger(__name__)

#: Default tolerance for feature-set drift, as a fraction of the feature vector.
#: When the incumbent requires a feature the current pipeline no longer produces,
#: the availability-over-fidelity trade is explicit and configurable rather than
#: hardcoded.
DEFAULT_DRIFT_FILL = 'zero'


def validate_and_promote(
  challenger: Any,
  challenger_model: Any,
  challenger_calibrator: Any,
  holdout: pd.DataFrame,
  parameters: dict[str, Any],
  incumbent: Any = None,
  incumbent_model: Any = None,
  incumbent_calibrator: Any = None,
  strict_metric: bool = True,
  allow_band_fallback: bool = False,
) -> tuple[Any, Any, Any]:
  """Score the challenger, compare it to the incumbent, and promote if accepted.

  Args:
    challenger: The challenger's trainer instance.
    challenger_model: The challenger's fitted estimator.
    challenger_calibrator: The challenger's fitted calibrator.
    holdout: The held-out slice both models are scored on.
    parameters: The full parameters document.
    incumbent: The incumbent's trainer instance, or ``None`` for a first
      training.
    incumbent_model: The incumbent's fitted estimator.
    incumbent_calibrator: The incumbent's fitted calibrator.
    strict_metric: When ``True``, an unrecognised configured metric fails the
      promotion rather than being silently substituted.
    allow_band_fallback: When ``True``, an empty requested band falls back to the
      top band, and the substitution is recorded in the audit trail.

  Returns:
    The winning triple. The challenger is returned when there is no incumbent,
    when the gate accepts it, or when the mode does not compare. The incumbent
    is returned only when the gate rejects the challenger.

  Raises:
    ValueError: If no holdout exists.
  """
  if holdout.empty:
    raise ValueError(
      'Model validation requires a held-out slice. Set eval_size (for example eval_size: 0.2) under '
      'the modeling block in parameters.yml.'
    )

  block = _modelling_block(parameters)
  spec = parse_metric(block.get('eval_metric'), kind='classification', strict=strict_metric)

  challenger_frame, challenger_score, band_used = _score_model(
    challenger, challenger_model, holdout, spec, allow_band_fallback
  )

  if incumbent is None:
    LOGGER.info('No incumbent; recording the first model', extra={'model_key': parameters.get('model_key')})
    apply_promotion(parameters, decide_promotion_from_parameters(parameters), spec.name, challenger_score, 0.0)
    # The node declares three outputs, so it must return three values. Returning
    # `None` here -- the "no incumbent" case, which is the *first* training of
    # every new model -- made Kedro raise while saving outputs, i.e. after the
    # model had already been trained and the registry already written. A first
    # training therefore could not complete in compare mode.
    return challenger, challenger_model, challenger_calibrator

  holdout = _reconcile_features(holdout, incumbent, challenger_frame)
  incumbent_frame, incumbent_score, _ = _score_model(
    incumbent, incumbent_model, holdout, spec, allow_band_fallback
  )
  del challenger_frame, incumbent_frame

  accepted, deviation = evaluate_acceptance(challenger_score, incumbent_score, spec)
  decision = _gate(accepted, deviation, spec, band_used, decide_promotion_from_parameters(parameters))

  apply_promotion(
    parameters, decision, spec.name, challenger_score if accepted else incumbent_score,
    incumbent_score, new_base_date=None,
  )
  if decision.action == 'PROMOTE':
    LOGGER.info('Challenger promoted', extra={'metric': spec.name, 'deviation': deviation})
    return challenger, challenger_model, challenger_calibrator
  LOGGER.info('Incumbent retained', extra={'metric': spec.name, 'deviation': deviation})
  return incumbent, incumbent_model, incumbent_calibrator


def validate_and_promote_regression(
  challenger: Any,
  challenger_model: Any,
  challenger_calibrator: Any,
  holdout: pd.DataFrame,
  parameters: dict[str, Any],
  incumbent: Any = None,
  incumbent_model: Any = None,
  incumbent_calibrator: Any = None,
  strict_metric: bool = True,
  allow_band_fallback: bool = False,
) -> tuple[Any, Any, Any]:
  """Score and compare a continuous-target challenger.

  The sign of the acceptance rule is derived from the metric's declared
  direction, so it is correct for losses as well as for scores. This is the
  issue the framework had: the regression gate tested
  ``deviation < 5``, which for a loss metric accepts a challenger up to five
  percent *worse* and rejects one four percent *better*. Since the reachable
  regression vocabulary is losses only, the configured regression model
  ran the inverted gate on every run — a silent wrong decision on
    a production promotion, which is the most dangerous class of issue there is.

  Args:
    challenger: The challenger's trainer instance.
    challenger_model: The challenger's fitted estimator.
    challenger_calibrator: The challenger's fitted calibrator.
    holdout: The held-out slice.
    parameters: The full parameters document.
    incumbent: The incumbent's trainer instance, or ``None``.
    incumbent_model: The incumbent's fitted estimator.
    incumbent_calibrator: The incumbent's fitted calibrator.
    strict_metric: Whether an unrecognised metric fails the promotion.
    allow_band_fallback: Whether an empty band falls back to the top band.

  Returns:
    The winning triple. With no incumbent the challenger wins by default, so the
    node's arity is the three declared outputs in every mode.
  """
  if holdout.empty:
    raise ValueError(
      'Model validation requires a held-out slice. Set eval_size under the modeling_reg block in '
      'parameters.yml.'
    )

  block = _modelling_block(parameters, regression=True)
  spec = parse_metric(block.get('eval_metric'), kind='regression', strict=strict_metric)

  (challenger_score, band_used), challenger_frame = _score_regression_model(
    challenger, challenger_model, holdout, spec, allow_band_fallback
  )
  if incumbent is None:
    apply_promotion(
      parameters, decide_promotion_from_parameters(parameters), spec.name, challenger_score, 0.0
    )
    return challenger, challenger_model, challenger_calibrator

  # The challenger's scored frame is the reference for a non-zero drift fill. The
  # regression path passed `None` unconditionally, which restricted the incumbent
  # to a fabricated zero for every feature the current pipeline no longer produces.
  holdout = _reconcile_features(holdout, incumbent, challenger_frame)
  (incumbent_score, _), _ = _score_regression_model(incumbent, incumbent_model, holdout, spec, allow_band_fallback)

  accepted, deviation = evaluate_acceptance(challenger_score, incumbent_score, spec)
  decision = _gate(accepted, deviation, spec, band_used, decide_promotion_from_parameters(parameters))
  apply_promotion(
    parameters, decision, spec.name, challenger_score if accepted else incumbent_score, incumbent_score
  )
  if decision.action == 'PROMOTE':
    return challenger, challenger_model, challenger_calibrator
  return incumbent, incumbent_model, incumbent_calibrator


def promote_without_comparison(
  challenger: Any,
  challenger_model: Any,
  challenger_calibrator: Any,
  holdout: pd.DataFrame,
  parameters: dict[str, Any],
) -> tuple[Any, Any, Any]:
  """Promote a freshly trained model without comparing it to an incumbent.

  This node exists so that the production-named model datasets are written by
  exactly one node in every training mode. Without it, the graph would have to
  make ``model_training`` itself write the production names in the common case
  and ``model_validation`` write them in the comparison case — two nodes with the
  same output contract but different arity, which the catalog cannot describe and
  which makes the graph's shape depend on the run configuration.

  The distinction between the two remaining non-comparison modes is preserved and
  is the only thing this node decides:

  ``train_updateMETADATA``
    A first training, or a retrain with no incumbent. The registry is updated so
    the new model becomes the current one.
  ``train_only``
    A plain training run. Nothing is written to the registry, so a retrain that
    has not been evaluated cannot silently replace the production model.

  Args:
    challenger: The new trainer instance.
    challenger_model: The new fitted estimator.
    challenger_calibrator: The new fitted calibrator.
    holdout: The held-out slice. Unused here, but accepted so the node has the
      same signature as :func:`validate_and_promote` and can be substituted for
      it.
    parameters: The full parameters document.

  Returns:
    The promoted triple, which is always the challenger.
  """
  del holdout
  train_mode = str(parameters.get(constants.ENV_TRAIN_MODE, 'train_only'))
  if train_mode == constants.TrainMode.UPDATE_METADATA.value:
    try:
      apply_promotion(
        parameters,
        decide_promotion_from_parameters(parameters),
        'TRAIN_ONLY',
        0.0,
        0.0,
      )
    except Exception as error:  # noqa: BLE001 - metadata write is best-effort here
      LOGGER.warning(
        'Could not update the model registry. The model is still promoted for this run, but the '
        'registry will not reflect it -- an unrecorded promotion is a governance gap.',
        extra={'error': str(error)},
      )
  else:
    LOGGER.info(
      'Train-only mode: the registry is not updated, so this model cannot replace the production '
      'model until a comparison run promotes it',
      extra={'train_mode': train_mode},
    )
  return challenger, challenger_model, challenger_calibrator


def decide_promotion_from_parameters(parameters: dict[str, Any]) -> PromotionDecision:
  """Derive the promotion decision from the temporal gate alone.

  A **first training** has no incumbent, so there is no incumbent window to be
  behind. Reading the registry to compare against a row that does not exist would
  make the very first training of every model impossible, which is the one case
  this gate must not block: with nothing to supersede, there is nothing to be
  better informed than. That case is therefore resolved here rather than by
  letting a missing row propagate as an error.

  Args:
    parameters: The full parameters document, which must carry
    ``perf_base_date``.

  Returns:
    The decision.
  """
  from datetime import date  # noqa: PLC0415

  from forecasting_ml_framework.modeling.promotion import read_current_entry  # noqa: PLC0415

  challenger_evidence = _parse_date(parameters.get('perf_base_date'))
  if challenger_evidence is None:
    challenger_evidence = date.min
  try:
    entry = read_current_entry(parameters)
  except RegistryEntryMissingError:
    LOGGER.info(
      'No current registry row; the temporal gate cannot block a first promotion',
      extra={'model_key': parameters.get('model_key')},
    )
    return PromotionDecision(
      'PROMOTE',
      'There is no incumbent model, so there is no observation window to be behind.',
    )
  return decide_promotion(challenger_evidence, entry['evidence_date'])


# --------------------------------------------------------------------------- #
# Internals
# --------------------------------------------------------------------------- #
def _score_model(
  trainer: Any,
  fitted: Any,
  holdout: pd.DataFrame,
  spec: Any,
  allow_band_fallback: bool,
) -> tuple[pd.DataFrame, float, int | None]:
  """Score one model on the holdout at its own operating point.

  The challenger's score is measured at *its own* threshold, using *its own*
  feature list, on the holdout it was split from — which is the correct
  comparison, and also the reason the two models' feature vectors may differ.

  Args:
    trainer: The trainer instance.
    fitted: The fitted estimator.
    holdout: The held-out slice.
    spec: The parsed metric specification.
    allow_band_fallback: Whether an empty band falls back to the top band.

  Returns:
    A three-tuple of the scored frame, the scalar metric and the band actually
    read.
  """
  working = holdout.reset_index(drop=True)
  features = list(trainer.features_to_use)
  target = trainer.target_col

  probabilities = fitted.predict_proba(working[features])
  positive = probabilities[:, -1]
  # The feature columns are retained in the returned frame, not just the target.
  # `_reconcile_features` reads a mean or median fill from this frame when the
  # incumbent needs a feature this pipeline no longer produces; returning a
  # target-only projection made that documented strategy unreachable, so every
  # drifted incumbent was silently scored against a fabricated zero.
  scored = working.copy()
  scored[constants.ORIG_SCORE_VALUE] = positive
  scored['__prediction'] = np.where(positive >= trainer.optimal_model_threshold, 1, 0)

  bands = band_metrics(
    scored, target_col=target, prediction_col='__prediction', probability_col=constants.ORIG_SCORE_VALUE, spec=spec
  )
  value, band_used = _read_metric(bands, spec, allow_band_fallback)
  return scored, value, band_used


def _score_regression_model(
  trainer: Any,
  fitted: Any,
  holdout: pd.DataFrame,
  spec: Any,
  allow_band_fallback: bool,
) -> tuple[tuple[float, int | None], pd.DataFrame]:
  """Score one continuous-target model on the holdout.

  Returns the scored frame alongside the metric. The frame is the only available
  source for a non-zero drift fill, so the promotion path needs it: returning a
  bare metric made the mean and median strategies unreachable on the regression
  side as well as the classification side.

  Args:
    trainer: The trainer instance.
    fitted: The fitted estimator.
    holdout: The held-out slice.
    spec: The parsed metric specification.
    allow_band_fallback: Whether an empty band falls back to the top band.

  Returns:
    A two-tuple of the metric pair and the scored frame.
  """
  working = holdout.reset_index(drop=True)
  features = list(trainer.features_to_use)
  target = trainer.target_col
  scored = working[[target]].copy()
  scored['__prediction'] = fitted.predict(working[features])

  bands, unused_profile = regression_band_metrics(
    scored,
    target_col=target,
    prediction_col='__prediction',
    probability_col='__prediction',
    n_features=len(features),
    spec=spec,
  )
  # The second return of `regression_band_metrics` is the lift-shaped companion
  # frame, produced by the same pass and not part of this node's output.
  del unused_profile
  return _read_metric(bands, spec, allow_band_fallback), scored


def _gate(
  accepted: bool,
  deviation: float,
  spec: Any,
  band_used: int | None,
  temporal: PromotionDecision,
) -> PromotionDecision:
  """Compose the acceptance gate and the temporal gate into one decision.

  The framework applies two independent gates, and both must pass for a
  promotion. ``accepted`` is the **acceptance** gate: the challenger scored no
  worse than the incumbent, within the non-regression tolerance. ``temporal`` is
  the **temporal** gate: the challenger was evaluated against an observation
  window at least as late as the incumbent's, which is what stops a run over old
  data from superseding a model that knows more.

  The gates are ordered so that the *narrower* one reports the failure. A
  challenger that fails acceptance is ``RECORD_ONLY`` regardless of its window,
  because a better-informed but worse model is still a worse model. A challenger
  that passes acceptance but fails the temporal gate is also ``RECORD_ONLY``,
  because passing acceptance against an older window is not evidence at all.

  Collapsing the two into one variable is the failure this exists to prevent:
  assigning the acceptance verdict and then discarding the temporal verdict makes
  the temporal gate dead code while leaving every call site looking as though both
  were enforced.

  Args:
    accepted: Whether the challenger met the non-regression tolerance.
    deviation: The relative deviation, as a percentage, for the rationale.
    spec: The parsed metric specification.
    band_used: The band the metric was read from, recorded in the audit trail.
    temporal: The temporal gate's own decision.

  Returns:
    The composed decision, carrying whichever gate's rationale explains it.
  """
  direction = 'higher-is-better' if spec.higher_is_better else 'lower-is-better'
  if not accepted:
    return PromotionDecision(
      'RECORD_ONLY',
      f'{spec.name} deviation {deviation:.2f}% exceeds the non-regression tolerance, so the '
      'incumbent is retained.',
      band_used=band_used,
    )
  if temporal.action == 'PROMOTE':
    return PromotionDecision(
      'PROMOTE',
      f'{spec.name} deviation {deviation:.2f}% is within the {abs(-5.0)}% non-regression tolerance for a '
      f'{direction} metric, and {temporal.rationale}',
      band_used=band_used,
    )
  return PromotionDecision(
    'RECORD_ONLY',
    f'{spec.name} deviation {deviation:.2f}% is within tolerance, but the promotion is blocked on '
    f'temporal grounds: {temporal.rationale}',
    band_used=band_used,
  )


def _read_metric(bands: pd.DataFrame, spec: Any, allow_band_fallback: bool) -> tuple[float, int | None]:
  """Read a single metric from a banded frame, with an audited fallback.

  Args:
    bands: The banded metric frame.
    spec: The parsed metric specification.
    allow_band_fallback: Whether an empty band falls back to the top band.

  Returns:
    A two-tuple of the value and the band actually read.

  Raises:
    MetricBandEmptyError: If the band is empty and the fallback is not permitted.
  """
  try:
    selected = select_band(bands, spec)
    key = 'ADJ_R2' if spec.metric == 'ADJ_R2' else spec.metric
    return float(selected.iloc[0][key]), int(selected.iloc[0]['score_ntile'])
  except MetricBandEmptyError:
    if not allow_band_fallback:
      raise
    top = bands[bands['score_ntile'] == bands['score_ntile'].max()]
    key = 'ADJ_R2' if spec.metric == 'ADJ_R2' else spec.metric
    band_used = int(top.iloc[0]['score_ntile'])
    LOGGER.warning(
      'Falling back to the top band. The substitution is recorded in the promotion audit log, '
      'because reporting one band\'s value under another band\'s metric name changes a reported '
      'number rather than merely whether a task runs.',
      extra={'requested': spec.name, 'band_used': band_used, 'value': float(top.iloc[0][key])},
    )
    return float(top.iloc[0][key]), band_used


def _reconcile_features(
  holdout: pd.DataFrame, incumbent: Any, challenger_frame: pd.DataFrame | None
) -> pd.DataFrame:
  """Reconcile the incumbent's feature vector with the current frame.

  The incumbent may have been trained on features the current pipeline no longer
  produces, because the column list was edited between training and now, or a
  routine was removed, or the challenger added features. Rather than failing, the
  missing features are filled so the comparison can always be made.

  The fill strategy is configurable and defaults to zero, which is a behaviour.
  That default is an explicit *availability over fidelity* trade and it is named
  as such: zero is a plausible value for a standardised or a tree model but may be
  far outside the training range for a log-transformed or a domain-normalised
  feature, so a score computed this way can be silently wrong. Configuring
  ``drift_fill: mean`` (or ``median``) uses the incumbent's own recorded training
  statistics, which the trainer captures at fit time and which travel with the
  incumbent inside the pickled handle.

  Args:
    holdout: The held-out slice.
    incumbent: The incumbent's trainer instance.
    challenger_frame: The challenger's scored frame, used to read a mean fill
      value when one is requested.

  Returns:
    The reconciled frame.
  """
  required = set(incumbent.features_to_use)
  missing = sorted(required - set(holdout.columns))
  if not missing:
    return holdout

  strategy = getattr(incumbent, 'drift_fill', DEFAULT_DRIFT_FILL)
  working = holdout.copy()
  # The reference for a non-zero fill is the *incumbent's own recorded training
  # statistic*, and it has to be: a column the current pipeline no longer produces
  # is by definition absent from the holdout and from the challenger's scored
  # frame, so no statistic can be read from either. The previous guard was
  # `column in challenger_frame.columns`, which is therefore False for every
  # member of `missing` -- the mean and median strategies were unreachable, every
  # drifted incumbent was scored against a fabricated zero, and the warning named
  # the 'zero' strategy that had never actually been selected.
  recorded = getattr(incumbent, '_feature_fill_stats', None) or {}
  for column in missing:
    value = _fill_value(strategy, recorded.get(column), challenger_frame, column)
    if value is None:
      LOGGER.warning(
        'The incumbent recorded no usable %r statistic for the missing feature %r; filling with '
        'zero, which is an assumption about the feature rather than a neutral value',
        strategy,
        column,
      )
      value = 0.0
    working[column] = value
  LOGGER.warning(
    'The incumbent requires %d feature(s) this pipeline no longer produces; filled using the %r '
    'strategy. The comparison is therefore between two differently-specified models, not two '
    'alternatives of the same model, and the incumbent\'s score may be optimistically biased.',
    len(missing),
    strategy,
    extra={'missing_features': missing},
  )
  return working


def _fill_value(
  strategy: str,
  recorded: dict[str, float] | None,
  reference: Any,
  column: str,
) -> float | None:
  """Compute a single feature's fill value, or ``None`` when none is usable.

  The incumbent's recorded statistic is preferred because it describes the
  distribution the incumbent was actually fitted on. The reference frame is a
  fallback for a column that happens to be present in the challenger's scored
  frame, which occurs when the drift is a *renaming* rather than a removal.

  Args:
    strategy: ``'mean'``, ``'median'`` or anything else, which yields ``None`` and
      therefore the zero fill.
    recorded: The incumbent's recorded statistic for this column, or ``None``.
    reference: A frame to read the statistic from, or ``None``.
    column: The column name.

  Returns:
    A finite float, or ``None`` when the strategy does not apply or no finite
    statistic is available.
  """
  if strategy not in ('mean', 'median'):
    return None
  if recorded and strategy in recorded:
    value = float(recorded[strategy])
    return value if math.isfinite(value) else None
  if reference is not None and column in getattr(reference, 'columns', []):
    try:
      value = float(reference[column].mean() if strategy == 'mean' else reference[column].median())
    except (TypeError, ValueError):
      return None
    return value if math.isfinite(value) else None
  return None


def _modelling_block(parameters: dict[str, Any], regression: bool = False) -> dict[str, Any]:
  """Locate the modelling configuration block.

  A thin, typed wrapper over
  :func:`~forecasting_ml_framework.nodes.model_nodes._resolve_model_block`, which
  owns the prefix vocabulary and the strict single-match rule. This function
  previously restated the whole lookup with its own prefixes and raised a bare
  ``ValueError``, so the same concept had two implementations that could drift --
  and the copy here is the one the promotion path uses.

  Args:
    parameters: The full parameters document.
    regression: Whether to resolve the regression block.

  Returns:
    The configuration block.

  Raises:
    DataContractError: If no block, or more than one block, matches.
  """
  from forecasting_ml_framework.nodes.model_nodes import (  # noqa: PLC0415
    CLASSIFICATION_BLOCK_TEMPLATE,
    REGRESSION_BLOCK_TEMPLATE,
    _resolve_model_block,
    _run_suffix,
  )

  return _resolve_model_block(
    parameters,
    REGRESSION_BLOCK_TEMPLATE if regression else CLASSIFICATION_BLOCK_TEMPLATE,
    _run_suffix(),
  )


def _parse_date(value: Any) -> Any:
  """Parse a configuration date.

  Args:
    value: The raw value.

  Returns:
    A ``date``, or ``None`` when the value is absent.
  """
  from datetime import datetime  # noqa: PLC0415

  if value is None or value == '':
    return None
  if hasattr(value, 'date'):
    return value.date()
  return datetime.strptime(str(value), '%Y-%m-%d').date()
