#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""The metric engines.

Two shapes, two consumers, correctly separated:

* :func:`band_metrics` and :func:`regression_band_metrics` produce a *wide*
  frame — one row per score band, dozens of columns — for the metric and lift
  tables that land in the warehouse.
* :func:`evaluate_band` and :func:`evaluate_regression_band` produce a *scalar*
  from a specific band, for the promotion gate, which needs one number to compare
  a challenger against an incumbent.

A system produced the wide frame in both cases and then indexed a
single row out of it, which is a wide intermediate for a scalar requirement. The
scalar path here is a separate, documented function.

**The metric vocabulary is defined once.** A defined the promotion
metric's allow-list twice, in the two validators, and used three different
spellings for adjusted R-squared across three sites: the frame's key
(``Adj_R2``), the allow-list entry (``ADJR2``) and the gate's direction test
(``ADJ_R2``). The consequence was that the only higher-is-better regression
metrics were unreachable, leaving only losses — and the gate's sign was inverted
for losses, so it reliably accepted a challenger that was *worse*. Both problems
are structural here: one registry, one spelling, one sign rule.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
  accuracy_score,
  brier_score_loss,
  f1_score,
  mean_absolute_error,
  mean_squared_error,
  precision_score,
  r2_score,
  recall_score,
  roc_auc_score,
)

from forecasting_ml_framework import constants
from forecasting_ml_framework.exceptions import (
  MetricBandEmptyError,
  MetricConfigurationError,
)
from forecasting_ml_framework.observability.logging import get_logger

LOGGER = get_logger(__name__)

#: Classification metrics for which a *larger* value is better.
#:
#: ``KSI`` is deliberately absent. The rank-separation statistic the architecture
#: describes is computed by the framework as a *threshold selection criterion*
#: (``ks_statistics`` in :data:`THRESHOLD_METHODS`), not as a reportable metric
#: column. It was previously listed here and in ``METRIC_GRAMMAR_HELP`` while
#: being subtracted again when the registry was built, so the parser rejected a
#: name the help text told operators to use -- an error message recommending the
#: very thing it refuses. A name joins this set only once an engine emits a
#: column of that name, so the vocabulary and the parser cannot disagree.
HIGHER_IS_BETTER = frozenset({'LIFT', 'PRECISION', 'RECALL', 'ACCURACY', 'F1', 'AUC'})
#: Classification metrics for which a *smaller* value is better.
LOWER_IS_BETTER = frozenset({'BRIER_SCORE'})
#: Regression metrics for which a smaller value is better.
REGRESSION_LOSSES = frozenset({'MAE', 'MAPE', 'MSE', 'MSLE', 'RMSE', 'RMSLE'})
#: Regression metrics for which a larger value is better.
REGRESSION_SCORES = frozenset({'R2', 'ADJ_R2'})

#: The single source of truth for metric names, directions and defaults. One
#: registry, one spelling per metric, and one direction each -- so a metric can
#: always be selected, and the acceptance rule can never disagree with the
#: quantity it is judging.
METRIC_REGISTRY: dict[str, dict[str, Any]] = {
  name: {'higher_is_better': True, 'kind': 'classification'} for name in HIGHER_IS_BETTER
}
METRIC_REGISTRY.update({name: {'higher_is_better': False, 'kind': 'classification'} for name in LOWER_IS_BETTER})
METRIC_REGISTRY.update({name: {'higher_is_better': False, 'kind': 'regression'} for name in REGRESSION_LOSSES})
METRIC_REGISTRY.update({name: {'higher_is_better': True, 'kind': 'regression'} for name in REGRESSION_SCORES})

#: The default promotion metric. Decile 10 is the metric retention campaigns are
#: actually managed on: how much lift did I get in the customers I was going to
#: target?
DEFAULT_PROMOTION_METRIC = 'LIFT_DECILE_10'
DEFAULT_REGRESSION_PROMOTION_METRIC = 'RMSE'

#: The grammar help shown when a metric name is rejected, rendered from
#: :data:`METRIC_REGISTRY` so it cannot name a metric the parser refuses.
METRIC_GRAMMAR_HELP = constants.render_metric_grammar_help(
  sorted(name for name, spec in METRIC_REGISTRY.items() if spec['kind'] == 'classification')
  + sorted(name for name, spec in METRIC_REGISTRY.items() if spec['kind'] == 'regression')
)


@dataclass(frozen=True)
class MetricSpec:
  """A parsed promotion-metric name.

  The grammar is ``<METRIC>[_DECILE|_CENTILE][_n]``. An unbanded name is
  evaluated over the whole population; a banded name is evaluated within a
  single band.

  Attributes:
    name: The configured name, upper-cased.
    metric: The base metric key, e.g. ``'LIFT'``.
    band_kind: ``'DECILE'``, ``'CENTILE'`` or ``None``.
    band: The band number, or ``0`` for an unbanded metric.
    kind: ``'classification'`` or ``'regression'``.
    higher_is_better: The direction of improvement.
  """

  name: str
  metric: str
  band_kind: str | None
  band: int
  kind: str
  higher_is_better: bool

  @property
  def is_banded(self) -> bool:
    """Report whether the metric is scoped to a single band.

    Returns:
      ``True`` when a band was requested.
    """
    return self.band_kind is not None


def parse_metric(metric_name: str | None, kind: str = 'classification', strict: bool = True) -> MetricSpec:
  """Parse and validate a promotion metric name.

  An unrecognised name fails the promotion rather than falling back. A typo in
  ``eval_metric`` would otherwise mean a production promotion was decided on a
  metric the operator did not choose, with the only evidence a line in a cluster
  log. Silently substituting a metric that drives a promotion is a governance
  failure rather than a usability convenience, so the default applies only when
  no metric was configured at all.

  Args:
    metric_name: The configured name, or ``None``.
    kind: ``'classification'`` or ``'regression'``.
    strict: When ``True`` and a metric *was* configured but is unrecognised, the
      promotion fails. When ``False`` the default is substituted and logged at
      warning level.

  Returns:
    The parsed specification.

  Raises:
    MetricConfigurationError: If a configured name is unrecognised and
      ``strict`` is set.
  """
  fallback = (
    DEFAULT_PROMOTION_METRIC if kind == 'classification' else DEFAULT_REGRESSION_PROMOTION_METRIC
  )
  if metric_name is None or not str(metric_name).strip():
    spec = parse_metric(fallback, kind, strict=True)
    LOGGER.info('No promotion metric configured; using the default', extra={'metric': spec.name})
    return spec

  candidate = str(metric_name).upper().strip()

  # The band qualifier is stripped first: the registry holds base metric names, so
  # a banded name must be decomposed before it can be validated. Decomposing
  # before validating is also what makes the *base* name the thing that is
  # checked, which is where a's three-spellings-of-ADJ_R2 issue lived.
  band_kind: str | None = None
  band = 0
  base = candidate
  parts = candidate.split('_')
  if len(parts) > 2 and parts[-2] in ('DECILE', 'CENTILE'):
    band_kind = parts[-2]
    try:
      band = int(parts[-1])
    except ValueError as error:
      raise MetricConfigurationError(
        f'Malformed band in promotion metric {metric_name!r}; expected '
        f'"<METRIC>_{band_kind}_<n>".\n{METRIC_GRAMMAR_HELP}',
        metric=metric_name,
      ) from error
    # Everything ahead of the band qualifier is the base name. Joining rather than
    # taking the first token is what lets a multi-word metric be banded at all:
    # `ADJ_R2_DECILE_10` must resolve to the base `ADJ_R2`, and `BRIER_SCORE_
    # DECILE_10` to `BRIER_SCORE`. Taking parts[0] made both resolve to a single
    # token, which put the registered multi-word metrics out of reach of banding.
    base = '_'.join(parts[:-2])
  elif len(parts) > 1 and parts[0] in ('DECILE', 'CENTILE'):
    raise MetricConfigurationError(
      f'Malformed promotion metric {metric_name!r}; a band qualifier must follow a metric name.\n'
      f'{METRIC_GRAMMAR_HELP}',
      metric=metric_name,
    )

  if base not in METRIC_REGISTRY:
    if strict:
      raise MetricConfigurationError(
        f'Unsupported promotion metric: {metric_name!r}.\n{METRIC_GRAMMAR_HELP}',
        metric=metric_name,
        kind=kind,
        supported=sorted(METRIC_REGISTRY),
      )
    LOGGER.warning(
      'Unsupported promotion metric; substituting the default. A promotion decided on an '
      'unintended metric is a governance failure, so set strict=True in production.',
      extra={'metric': metric_name, 'fallback': fallback},
    )
    return parse_metric(fallback, kind, strict=True)

  if kind == 'classification' and METRIC_REGISTRY[base]['kind'] != 'classification':
    raise MetricConfigurationError(
      f'{metric_name!r} is a regression metric but was requested for a classification model. '
      f'Class metrics: {sorted(k for k, v in METRIC_REGISTRY.items() if v["kind"] == "classification")}',
      metric=metric_name,
      kind=kind,
    )

  return MetricSpec(
    name=candidate,
    metric=base,
    band_kind=band_kind,
    band=band,
    kind=METRIC_REGISTRY[base]['kind'],
    higher_is_better=bool(METRIC_REGISTRY[base]['higher_is_better']),
  )


def acceptance_threshold() -> float:
  """Return the relative non-regression tolerance, as a percentage.

  A challenger that is *worse than the incumbent but by less than this margin* is
  promoted. That is a non-regression gate, not an improvement gate, and it is the
  right gate for this problem: retrained retention models on rolling windows are
  routinely statistically indistinguishable, so an improvement gate would reject
  near-ties on sampling noise and stall the model lifecycle indefinitely.

  Returns:
    The tolerance percentage.
  """
  return -5.0


def evaluate_acceptance(new_score: float, old_score: float, spec: MetricSpec) -> tuple[bool, float]:
  """Apply the non-regression gate with the correct sign for the metric.

  the classification gate was ``diff >= -5`` and the regression gate
  was ``diff < 5``. For a loss metric the latter accepts a challenger up to five
  percent *worse* and rejects one four percent *better* — the sign is inverted.
  Because the reachable regression vocabulary happened to be losses only, the
  configured regression model hit the inverted gate on every run, so a worse
  model was promoted with a log line reading ``score deviation: 3.7``.

  The sign is derived from the metric's declared direction, so it is correct
  for every metric rather than correct only for the higher-is-better ones.

  Args:
    new_score: The challenger's score.
    old_score: The incumbent's score.
    spec: The parsed metric specification.

  Returns:
    A two-tuple of the acceptance decision and the relative deviation in percent.
  """
  if not old_score:
    LOGGER.warning(
      'Incumbent score is zero; the relative deviation is undefined and the challenger is accepted',
      extra={'metric': spec.name, 'new_score': new_score},
    )
    return True, 0.0

  deviation = ((new_score - old_score) / abs(old_score)) * 100
  tolerance = acceptance_threshold()
  # A relative deviation computed in binary floating point lands a hair below the
  # tolerance at the boundary -- 0.95 against 1.0 gives -5.000000000000004 -- so a
  # challenger exactly at tolerance would be rejected by a strict comparison. The
  # epsilon is scaled to the magnitude of the comparison, not fixed, so it cannot
  # mask a real breach.
  epsilon = abs(tolerance) * 1e-9
  accepted = (
    deviation >= tolerance - epsilon if spec.higher_is_better else deviation <= -tolerance + epsilon
  )
  LOGGER.info(
    'Promotion gate evaluated',
    extra={
      'metric': spec.name,
      'higher_is_better': spec.higher_is_better,
      'new_score': new_score,
      'old_score': old_score,
      'deviation_percent': deviation,
      'accepted': accepted,
    },
  )
  return accepted, deviation


def select_band(frame: pd.DataFrame, spec: MetricSpec, band_col: str = 'score_ntile') -> pd.DataFrame:
  """Select the band a metric is evaluated on.

  a fell back to the highest band when the requested one
  was empty, so ``LIFT_DECILE_7`` reported decile 10's lift — a materially
  different quantity presented as a successful measurement. That is the
  difference between changing whether a *task runs* and changing a *reported
  number*, and the second must never be silent.

  Args:
    frame: The banded metric frame.
    spec: The parsed metric specification.
    band_col: The band column.

  Returns:
    A single-row frame.

  Raises:
    MetricBandEmptyError: If the requested band is absent and ``strict`` was not
      set on the parse. Callers that prefer availability may catch this and
      substitute the top band, recording the substitution in the audit trail.
  """
  if not spec.is_banded:
    return frame
  selected = frame[frame[band_col] == spec.band]
  if selected.empty:
    available = sorted(frame[band_col].unique().tolist()) if band_col in frame else []
    raise MetricBandEmptyError(
      f'The requested band {spec.name!r} is empty. Available bands: {available}. Falling back to '
      'the top band would report a materially different quantity under the requested metric '
      'name, so this is reported rather than absorbed.',
      metric=spec.name,
      available_bands=available,
    )
  return selected.reset_index(drop=True)


def band_metrics(
  frame: pd.DataFrame,
  target_col: str,
  prediction_col: str,
  probability_col: str,
  spec: MetricSpec,
) -> pd.DataFrame:
  """Compute the per-band classification metric table.

  Args:
    frame: A labelled frame carrying the target, the hard prediction and the
      score.
    target_col: The label column.
    prediction_col: The binarised prediction column.
    probability_col: The score column the bands are derived from.
    spec: The parsed metric specification, whose ``band_kind`` determines
      whether ten deciles or one hundred centiles are produced.

  Returns:
    A frame with one row per band, carrying counts, lift, and six metrics.
  """
  working = frame.copy()
  working = _assign_bands(working, probability_col, spec)
  working['tp'] = ((working[prediction_col] == 1) & (working[target_col] == 1)).astype(int)
  working['fp'] = ((working[prediction_col] == 1) & (working[target_col] == 0)).astype(int)
  working['tn'] = ((working[prediction_col] == 0) & (working[target_col] == 0)).astype(int)
  working['fn'] = ((working[prediction_col] == 0) & (working[target_col] == 1)).astype(int)

  aggregated = working.groupby('score_ntile').agg(
    ntile_count=(target_col, 'count'),
    target_count=(target_col, 'sum'),
    sum_tp=('tp', 'sum'),
    sum_fp=('fp', 'sum'),
    sum_tn=('tn', 'sum'),
    sum_fn=('fn', 'sum'),
  )
  # Lift is the band's positive RATE over the overall positive RATE, so an
  # uninformative model reports ~1.0 in every band and the top band is the
  # multiple by which it beats the base rate. It is deliberately *not* computed
  # as `band_positives / mean(band_positives)`: that is the band's share of all
  # positives scaled by the band count, which reads 10.0 rather than 1.0 for a
  # coin flip and therefore makes the number meaningless as a ratio.
  band_rate = aggregated['target_count'] / aggregated['ntile_count'].replace(0, np.nan)
  overall_rate = aggregated['target_count'].sum() / aggregated['ntile_count'].sum()
  aggregated['LIFT'] = band_rate / overall_rate if overall_rate else np.nan
  aggregated['OVERALL_RATE'] = overall_rate
  aggregated = aggregated.reset_index().rename(columns={'index': 'score_ntile'})

  per_band = _apply_per_band(
    working, _classification_band_row, target_col=target_col, prediction_col=prediction_col,
    probability_col=probability_col,
  )
  return aggregated.merge(per_band, on='score_ntile', how='inner').sort_values('score_ntile', ascending=False)


def regression_band_metrics(
  frame: pd.DataFrame,
  target_col: str,
  prediction_col: str,
  probability_col: str,
  n_features: int,
  spec: MetricSpec,
) -> tuple[pd.DataFrame, pd.DataFrame]:
  """Compute the per-band regression error table and the lift-shaped companion.

  With a continuous target there is no positive class to lift over, so the decile
  error profile is the analogue of a lift table. That is why the regression
  pipelines have no separate lift node.

  Args:
    frame: A labelled frame carrying the target and the prediction.
    target_col: The target column.
    prediction_col: The predicted value column.
    probability_col: The score column the bands are derived from.
    n_features: The feature count, used for adjusted R-squared.
    spec: The parsed metric specification.

  Returns:
    A two-tuple of the error-profile frame and the lift-shaped companion frame.
  """
  working = frame.copy()
  # The band's granularity is taken from the *specification*, not forced to
  # DECILE. Forcing it made two things unreachable. A centile-banded regression
  # metric -- ``RMSE_CENTILE_90`` -- parsed to a spec naming band 90 while the
  # table only ever contained bands 1..10, so selecting it raised
  # ``MetricBandEmptyError`` on a perfectly valid configuration. And an *unbanded*
  # metric, which ``MetricSpec`` documents as "evaluated over the whole
  # population", was banded into ten ascending rows of which the reader took the
  # first -- so ``RMSE`` read *decile 1*'s error rather than the population's, and
  # ``RMSE`` is the default regression promotion metric. A challenger 40% worse
  # overall but 10% better in the bottom decile would have been promoted.
  working = _assign_bands(working, probability_col, spec)
  working['pred_error'] = working[target_col] - working[prediction_col]
  working['abs_error'] = working['pred_error'].abs()

  per_band = _apply_per_band(
    working, _regression_band_row, target_col=target_col, prediction_col=prediction_col, n_features=n_features
  )

  profile = working.groupby('score_ntile').agg(
    ntile_count=(target_col, 'count'),
    target_sum=(target_col, 'sum'),
    target_min=(target_col, 'min'),
    target_max=(target_col, 'max'),
    target_mean=(target_col, 'mean'),
    pred_sum=(prediction_col, 'sum'),
    pred_min=(prediction_col, 'min'),
    pred_max=(prediction_col, 'max'),
    pred_mean=(prediction_col, 'mean'),
    max_abs_error=('abs_error', 'max'),
  ).reset_index()
  return profile.merge(per_band, on='score_ntile', how='inner').sort_values('score_ntile'), profile


def evaluate_regression_band(frame: pd.DataFrame, target_col: str, prediction_col: str, n_features: int) -> float:
  """Compute adjusted R-squared over a whole population.

  Adjusted R-squared uses the feature count as the parameter count, which is
  correct in form for an unregularised estimator and over-pessimistic for a
  heavily implicit one such as a gradient-boosted ensemble. It is retained
  because it is a legitimate comparison statistic across candidate models, and
  the assumption is documented rather than hidden.

  Args:
    frame: A labelled frame.
    target_col: The target column.
    prediction_col: The predicted value column.
    n_features: The feature count.

  Returns:
    The adjusted R-squared value.
  """
  rows = len(frame)
  if rows <= n_features + 1:
    raise MetricBandEmptyError(
      f'Adjusted R^2 is undefined for {rows} rows and {n_features} features; the denominator '
      'would be zero or negative.',
      rows=rows,
      n_features=n_features,
    )
  r2 = float(r2_score(frame[target_col], frame[prediction_col]))
  return 1.0 - ((1.0 - r2) * (rows - 1) / (rows - n_features - 1))


def evaluate_band(frame: pd.DataFrame, spec: MetricSpec, band_col: str = 'score_ntile') -> float:
  """Extract a single scalar metric value from a banded metric frame.

  Args:
    frame: The banded metric frame.
    spec: The parsed metric specification.
    band_col: The band column.

  Returns:
    The metric value for the selected band.
  """
  selected = select_band(frame, spec, band_col)
  row = selected.iloc[0]
  if spec.metric == 'ADJ_R2':
    return float(row['ADJ_R2'])
  return float(row[spec.metric])


def _assign_bands(
  frame: pd.DataFrame,
  probability_col: str,
  spec: MetricSpec,
  band_kind: str | None = None,
) -> pd.DataFrame:
  """Assign rank-based, equal-volume bands to a frame.

  ``pd.qcut`` is rank-based, so bands are equal-volume by construction and the
  top band always holds the same share of the population. ``duplicates='drop'``
  is essential: on a skewed or low-cardinality score it prevents ``qcut`` from
  raising on duplicate bin edges.

  For an unbanded metric every row is assigned to band ``0``, so the
  whole-population case flows through exactly the same code path as a banded one.

  Args:
    frame: The frame to band.
    probability_col: The score column.
    spec: The parsed metric specification.
    band_kind: An override for the band's granularity.

  Returns:
    The frame with a ``score_ntile`` column, 1-indexed.
  """
  kind = band_kind or spec.band_kind
  if kind is None:
    frame['score_ntile'] = 0
    return frame
  count = constants.DECILE_COUNT if kind == 'DECILE' else constants.CENTILE_COUNT
  # The degeneracy checks precede `qcut` deliberately. `qcut` on an all-NaN
  # column reaches into numpy's internals and raises `IndexError: index -1 is out
  # of bounds for axis 0 with size 0`, so the guard written after the call never
  # runs on the case it was written for. One usable value has no ordering to
  # band either, and neither does one distinct value.
  usable = frame[probability_col].dropna()
  if len(usable) < 2 or usable.nunique() < 2:
    # A constant or unusable score has no ordering to band.
    frame['score_ntile'] = 1
    return frame
  raw = pd.qcut(frame[probability_col], count, labels=False, duplicates='drop')
  # `duplicates='drop` avoids a raise on duplicate edges but leaves gaps in the
  # label sequence, so a requested band number could match no rows at all. Dense
  # re-ranking guarantees the bands are contiguous from 1.
  frame['score_ntile'] = raw.fillna(0).astype(int).rank(method='dense').astype(int)
  return frame


def _apply_per_band(
  frame: pd.DataFrame,
  row_builder: Any,
  **kwargs: Any,
) -> pd.DataFrame:
  """Apply a per-band row builder across every score band.

  ``DataFrameGroupBy.apply`` grew an ``include_groups`` keyword in pandas 2.2,
  while the framework pins 2.1. Passing it unconditionally sends it through to the
  applied function as an unexpected keyword argument. Rather than branch on the
  installed version, the band column is selected out of the group before the call
  and the grouping key is carried explicitly, so the same code works on both and
  on either side of the deprecation.

  Args:
    frame: The banded frame.
    row_builder: The per-band callable.
    **kwargs: Keyword arguments forwarded to the row builder.

  Returns:
    One row per band, with the band id as a column.
  """
  bands = frame['score_ntile'].to_numpy()
  rows: list[pd.Series] = []
  for band_id in sorted(set(bands.tolist())):
    rows.append(row_builder(frame[bands == band_id], **kwargs))
  result = pd.DataFrame(rows)
  result.insert(0, 'score_ntile', sorted(set(bands.tolist())))
  return result


def _classification_band_row(
  band: pd.DataFrame,
  target_col: str,
  prediction_col: str,
  probability_col: str,
) -> pd.Series:
  """Compute the six band metrics for one score band.

  Args:
    band: The rows in this band.
    target_col: The label column.
    prediction_col: The binarised prediction column.
    probability_col: The score column.

  Returns:
    A series of the band's metrics. AUC and F1 are undefined for a single-class
    band, so they degrade to ``NaN`` rather than raising — a degenerate band is
    an expected outcome of a skewed score, not an error.
  """
  labels = band[target_col]
  predictions = band[prediction_col]
  return pd.Series(
    {
      'AUC': _safe(roc_auc_score, labels, band[probability_col]),
      'PRECISION': _safe(precision_score, labels, predictions, zero_division=0),
      'RECALL': _safe(recall_score, labels, predictions, zero_division=0),
      'ACCURACY': _safe(accuracy_score, labels, predictions),
      'F1': _safe(f1_score, labels, predictions, zero_division=0),
      'BRIER_SCORE': _safe(brier_score_loss, labels, band[probability_col]),
    }
  )


def _regression_band_row(
  band: pd.DataFrame,
  target_col: str,
  prediction_col: str,
  n_features: int,
) -> pd.Series:
  """Compute the error metrics for one score band.

  ``MSLE`` is computed by hand as the mean of ``(log(y+1) - log(yhat+1))^2``
  rather than by calling scikit-learn, because the ``+1`` offset is what keeps
  ``log`` defined for a zero target. That is a deliberate, documented difference
  from ``mean_squared_log_error``, whose edge handling differs.

  Args:
    band: The rows in this band.
    target_col: The target column.
    prediction_col: The predicted value column.
    n_features: The feature count, used for adjusted R-squared.

  Returns:
    A series of the band's error metrics.
  """
  actual = band[target_col].astype(float)
  predicted = band[prediction_col].astype(float)
  mse = float(mean_squared_error(actual, predicted))
  msle = _mean_squared_log_error(actual, predicted)
  row = {
    'MAE': float(mean_absolute_error(actual, predicted)),
    'MAPE': mean_absolute_percentage_error(actual, predicted),
    'MSE': mse,
    'RMSE': float(np.sqrt(mse)),
    'MSLE': msle,
    'RMSLE': float(np.sqrt(msle)),
    'R2': _safe(r2_score, actual, predicted),
  }
  if len(band) > n_features + 1:
    row['ADJ_R2'] = 1.0 - ((1.0 - row['R2']) * (len(band) - 1) / (len(band) - n_features - 1))
  else:
    row['ADJ_R2'] = np.nan
  return pd.Series(row)


def _mean_squared_log_error(actual: pd.Series, predicted: pd.Series) -> float:
  """Return the mean squared logarithmic error over every row in the band.

  The subtlety is the domain. ``log(y + 1)`` is undefined for any ``y < -1``, so
  a regressor that predicts a negative amount produces ``NaN`` -- and
  ``Series.mean()`` *silently skips* ``NaN``. The band mean was therefore taken
  over a reduced, unpredictable subset: the worst error in the band was discarded
  and the denominator was wrong, which is how a badly-fitting band could report a
  small MSLE. A regressor predicting a negative amount is routine, not exotic.

  The prediction is clipped at zero before the transform. That is the standard
  treatment for a log-scale error on a quantity that cannot be negative, and it
  is applied here rather than left to produce an undefined result. A negative
  *target* is not clippable -- it means the data is wrong -- so those rows
  contribute ``NaN`` and are reported through the returned value rather than
  being silently averaged away.

  Args:
    actual: The target values.
    predicted: The predicted values.

  Returns:
    The mean squared logarithmic error, or ``NaN`` when no row is defined.
  """
  terms = (np.log(np.clip(actual, 0, None) + 1) - np.log(np.clip(predicted, 0, None) + 1)) ** 2
  terms = pd.Series(terms, index=actual.index).to_numpy(dtype=float)
  if not np.isfinite(terms).any():
    return float('nan')
  return float(np.nanmean(terms))


def mean_absolute_percentage_error(
  actual: pd.Series | np.ndarray,
  predicted: pd.Series | np.ndarray,
) -> float:
  """Compute the mean absolute percentage error, excluding zero targets.

  A percentage error is undefined where the target is zero, and the unguarded
  expression divides a generally non-zero numerator by a zero denominator, which
  yields ``inf`` rather than ``NaN``. The previous guard tested
  ``(actual != 0).any()`` -- that *at least one* target is non-zero -- which is
  true for essentially every dataset, so the guard passed and the surviving
  infinities were averaged straight into the published metric, because
  ``np.nanmean`` discards ``NaN`` and not ``inf``.

  This is not a cosmetic issue. ``inf`` is the worst possible input to the
  promotion gate: the relative deviation computed from it is also ``inf``, so the
  gate can never be satisfied, and if the *incumbent* carries the ``inf`` the
  deviation collapses to zero and the gate accepts unconditionally. Zero is a
  routine target value for count- and amount-flavoured models, so this is a common
  case. The masked division below is the correct computation, and it is the one
  :func:`regression_band_metrics` already builds in its ``abs_perc_error`` column
  before discarding it.

  Args:
    actual: The observed target.
    predicted: The predicted target.

  Returns:
    The mean absolute percentage error as a percentage, or ``nan`` when every
    target is zero and the quantity is therefore undefined.
  """
  actual_series = pd.Series(actual, dtype=float).reset_index(drop=True)
  predicted_series = pd.Series(predicted, dtype=float).reset_index(drop=True)
  non_zero = actual_series != 0
  if not bool(non_zero.any()):
    return float('nan')
  ratio = (actual_series - predicted_series)[non_zero].abs() / actual_series[non_zero].abs()
  return float(np.nanmean(ratio) * 100)


def _safe(function: Any, *args: Any, **kwargs: Any) -> float:
  """Call a metric, degrading an undefined result to ``NaN``.

  Args:
    function: The metric callable.
    *args: Positional arguments.
    **kwargs: Keyword arguments.

  Returns:
    The metric value, or ``NaN`` when it is undefined.
  """
  try:
    return float(function(*args, **kwargs))
  except ValueError as error:
    LOGGER.debug('Metric is undefined for this slice', extra={'metric': function.__name__, 'error': str(error)})
    return float('nan')
