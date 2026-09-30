#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Probability calibration and decision-threshold selection.

**Calibration.** The framework computes two rates: ``real_rate``, the population
positive rate measured *before* any reweighting, and ``sample_rate``, the
positive rate of the effective training distribution *after* reweighting. Their
ratio is the exact prior-shift factor, so the correction below is exact Bayes
scaling rather than a refitted calibrator.

Writing ``p`` for the raw model probability, ``pi_sample`` for the training rate
and ``pi_true`` for the population rate, the Saerens-Latinne-Decaestecker
correction scales the posterior odds by the ratio of the two prior odds::

    p_adj = ( (p/(1-p)) * (pi_true/(1-pi_true)) / (pi_sample/(1-pi_sample)) ) / (1 + ... )

Equivalently, in the closed form actually implemented::

    p_adj = (p * pi_true/pi_sample)
            / ( (1-p) * (1-pi_true)/(1-pi_sample) + p * pi_true/pi_sample )

**The direction of the correction is the whole point, and it is easy to invert.**
A model trained on a sample enriched with positives overstates the true
probability, so ``pi_true < pi_sample`` must *reduce* the odds. An implementation
that multiplies by ``pi_sample/pi_true`` instead produces a score moving in the
wrong direction — which is worse than no calibration at all, because the error is
invisible: the output still looks like a probability.

The correction is valid under the assumption that the model's *ranking* is
unaffected by the reweighting — which is exactly what a loss weight such as
``scale_pos_weight`` preserves, because it reweights the loss rather than the
data. It is also exact under stratified resampling when the sampling fractions
are known. When no correction was applied, ``sample_rate == real_rate`` and the
formula reduces to the identity, which is why the framework can ship the
correction unconditionally.

**The dual-score contract.** Two score columns are produced simultaneously:
``orig_score_value`` carries the raw model probability, which is what
discrimination, model comparison and threshold semantics are defined on; and
``score_value`` carries the calibrated probability, which is what is published
as the expected event probability. Keeping both lets a consumer verify that
calibration has not degraded ranking, and lets a monitor watch the two
distributions independently.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef, precision_recall_curve, roc_curve

from forecasting_ml_framework.observability.logging import get_logger

LOGGER = get_logger(__name__)

#: Supported threshold-selection criteria.
THRESHOLD_METHODS = ('pr_curve', 'roc_curve', 'mcc_curve', 'ks_statistics')

#: The escalating grids used by the ``mcc_curve`` progressive zoom. For a
#: retention model with a one-percent positive rate, the MCC optimum is routinely
#: below 0.01, and a single coarse grid reports ``0`` because its resolution
#: cannot express the answer. Each grid is entered only if the previous one
#: selected its own leftmost point, so the zoom costs nothing when the coarse
#: answer is already adequate.
MCC_ZOOM_GRIDS = (
  np.arange(0.0, 1.0, 0.01),
  np.arange(0.0, 0.01, 0.0001),
  np.arange(0.0, 0.0001, 0.000001),
)

#: Fractional precision used when rounding a threshold, chosen adaptively so the
#: value stays usable as both a float comparison and a readable configuration
#: entry.
_ROUNDING_BANDS = ((1e-4, 6), (1e-2, 4))


def calibrate_probability(
  probabilities: pd.Series | np.ndarray,
  real_rate: float,
  sample_rate: float,
) -> pd.Series:
  """Apply exact prior-shift correction to a probability column.

  Args:
    probabilities: The raw model probability.
    real_rate: ``P(y = 1)`` in the population, measured before reweighting.
    sample_rate: ``P(y = 1)`` in the effective training distribution.

  Returns:
    The calibrated probability. The raw score is returned unchanged when either
    rate is degenerate, which is what prevents a division by zero from silently
    disabling calibration on a single-class frame.

  Raises:
    Never. A degenerate prior degrades to the raw score rather than raising.
  """
  # Both *endpoints* of the unit interval are degenerate, not just zero. The
  # prior odds are ``rate / (1 - rate)``, which is undefined at 0 and at 1
  # equally, and the previous guard tested only ``not rate`` -- so it caught 0.0
  # and passed 1.0 straight into a scalar division by zero. A rate of exactly
  # 1.0 is reachable from an ordinary two-class frame: the framework rounds the
  # observed rate to two decimals, so 9 996 positives out of 10 000 rows rounds
  # to 1.0, and the run then died inside threshold selection with a bare
  # ``ZeroDivisionError`` after the model had already been fitted.
  if not _is_proportion(real_rate) or not _is_proportion(sample_rate):
    LOGGER.warning(
      'Prior-shift calibration is degenerate and was skipped; the raw score is published instead',
      extra={'real_rate': real_rate, 'sample_rate': sample_rate},
    )
    return probabilities

  raw = np.asarray(probabilities, dtype=float)
  safe = np.clip(raw, 1e-12, 1 - 1e-12)
  # Saerens-Latinne-Decaestecker: scale the *posterior odds* by the ratio of the
  # true prior's odds to the training prior's odds. The direction matters and is
  # easy to invert: a model trained on a sample that was enriched with positives
  # overstates the true probability, so the correction must *reduce* the odds.
  odds_train = safe / (1.0 - safe)
  prior_odds = (real_rate / (1.0 - real_rate)) / (sample_rate / (1.0 - sample_rate))
  adjusted_odds = odds_train * prior_odds
  calibrated = adjusted_odds / (adjusted_odds + 1.0)
  return pd.Series(calibrated, index=getattr(probabilities, 'index', None), name=getattr(probabilities, 'name', None))


def _is_proportion(value: Any) -> bool:
  """Report whether a value is a usable proportion, strictly inside ``(0, 1)``.

  The prior-odds term ``rate / (1 - rate)`` is defined only for a rate strictly
  between zero and one. Both endpoints are excluded deliberately: a rate of zero
  is a frame with no positives and a rate of one is a frame with no negatives,
  and neither carries the prior information the correction consumes.

  Args:
    value: The candidate rate.

  Returns:
    ``True`` when the value is a finite, non-zero, non-unit proportion.
  """
  try:
    rate = float(value)
  except (TypeError, ValueError):
    return False
  if not np.isfinite(rate):
    return False
  return 0.0 < rate < 1.0


def select_threshold(
  frame: pd.DataFrame,
  method: str,
  target_col: str,
  probability_col: str,
) -> float:
  """Select the decision threshold that maximises a chosen criterion.

  Args:
    frame: A labelled frame containing the target and the probability.
    method: One of :data:`THRESHOLD_METHODS`. An unrecognised name degrades to
      ``0.5`` and is logged at warning level.
    target_col: The label column.
    probability_col: The probability column being thresholded.

  Returns:
    The selected threshold, rounded adaptively.

  Raises:
    ValueError: If the frame is empty or the required columns are absent.
  """
  if frame is None or frame.empty:
    raise ValueError('Cannot select a threshold from an empty frame')
  missing = [column for column in (target_col, probability_col) if column not in frame.columns]
  if missing:
    raise ValueError(f'Cannot select a threshold; missing column(s): {missing}')

  y_true = frame[target_col].to_numpy()
  y_score = frame[probability_col].to_numpy(dtype=float)

  if method == 'pr_curve':
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    denominator = precision + recall
    fscore = np.divide(2 * precision * recall, denominator, out=np.zeros_like(precision), where=denominator > 0)
    selected = float(thresholds[int(np.argmax(fscore))])
  elif method == 'roc_curve':
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    gmeans = np.sqrt(tpr * (1 - fpr))
    selected = float(thresholds[int(np.argmax(gmeans))])
  elif method == 'mcc_curve':
    selected = _select_by_mcc(y_true, y_score)
  elif method == 'ks_statistics':
    selected = _select_by_ks(frame, target_col, probability_col)
  else:
    LOGGER.warning(
      'Unrecognised threshold_selection_method; falling back to 0.5',
      extra={'method': method, 'supported': list(THRESHOLD_METHODS)},
    )
    selected = 0.5

  selected = _guard_threshold(selected, y_score, method, probability_col)
  rounded = _round_threshold(selected)
  LOGGER.info(
    'Selected decision threshold',
    extra={'method': method, 'column': probability_col, 'threshold': rounded},
  )
  return rounded


def _guard_threshold(selected: float, y_score: np.ndarray, method: str, column: str) -> float:
  """Refuse a threshold that is not a usable probability.

  Both scikit-learn threshold curves lead with a sentinel that is not a
  probability: ``roc_curve`` opens with ``inf`` and ``precision_recall_curve``
  with the *lowest* observed score. A criterion that is undefined on a degenerate
  slice -- a single class, or scores with no separation -- makes every candidate
  equal, ``argmax`` returns index 0, and the sentinel is selected. It is then
  pickled into the model handle and applied to every future scoring run, so the
  defect is durable and silent: ``inf`` predicts nothing positive, and the
  minimum score predicts everything positive.

  A threshold is only meaningful if it is finite and inside the range of scores
  it was selected from. Outside that range the selection carries no information,
  so the maximum observed score is substituted: the most permissive *valid*
  cut-off, which is the conservative choice for a degenerate slice because it
  defers to the existing threshold rather than inventing one.

  Args:
    selected: The threshold the criterion chose.
    y_score: The scores the criterion was evaluated on.
    method: The criterion, for the log record.
    column: The score column, for the log record.

  Returns:
    A finite threshold within the observed score range.
  """
  if y_score.size == 0:
    return 0.5
  lowest, highest = float(np.nanmin(y_score)), float(np.nanmax(y_score))
  if np.isfinite(selected) and lowest <= selected <= highest:
    return selected
  replacement = highest
  LOGGER.warning(
    'The threshold criterion produced a threshold outside the observed score range; substituting '
    'the maximum observed score. This happens when the criterion is undefined for the slice, '
    'typically a single-class holdout or scores with no separation. The sentinel was not '
    'persisted, because it would otherwise be applied unchanged to every future scoring run.',
    extra={
      'method': method,
      'column': column,
      'selected': selected,
      'substituted': replacement,
      'score_min': lowest,
      'score_max': highest,
    },
  )
  return replacement


def _select_by_mcc(y_true: np.ndarray, y_score: np.ndarray) -> float:
  """Select the threshold maximising the Matthews correlation coefficient.

  MCC is the only criterion in the set that stays informative under extreme
  imbalance, which is the entire reason the progressive zoom exists: the first
  grid's resolution often cannot express the answer at all, and each subsequent
  grid is entered only when the previous one selected its leftmost point.

  Args:
    y_true: The labels.
    y_score: The probabilities.

  Returns:
    The selected threshold.
  """
  best_threshold = 0.0
  for index, grid in enumerate(MCC_ZOOM_GRIDS):
    binarised = (y_score[None, :] >= grid[:, None]).astype(int)
    scores = np.empty(len(grid), dtype=float)
    for position, prediction in enumerate(binarised):
      scores[position] = _safe_mcc(y_true, prediction)
    best_index = int(np.nanargmax(scores))
    best_threshold = float(grid[best_index])
    # The zoom is entered only when this grid's answer sits on its own leftmost
    # point, which is the signal that its resolution could not express a better
    # answer. Otherwise the coarse result is already the answer.
    #
    # The terminal test is the grid's POSITION and not its length. The grids are
    # deliberately equal in length -- each covers a decade at a fixed hundred
    # points -- so a length comparison reported "last grid" on the very first pass
    # and made the finer grids unreachable. The zoom is the entire reason a
    # sub-percent retention model gets a usable operating point at all, so this is
    # worth stating rather than leaving as an apparent simplification.
    if best_index > 0 or index == len(MCC_ZOOM_GRIDS) - 1:
      break
  return best_threshold


def _select_by_ks(frame: pd.DataFrame, target_col: str, probability_col: str) -> float:
  """Select the threshold maximising the Kolmogorov-Smirnov separation.

  KS optimises ranking quality directly rather than optimising a point metric,
  which is why it is the natural criterion when the operational question is
  "how well does the score separate the population?".

  Args:
    frame: A labelled frame.
    target_col: The label column.
    probability_col: The probability column.

  Returns:
    The selected threshold.
  """
  ordered = frame.sort_values(by=probability_col, ascending=False)
  labels = ordered[target_col]
  positives = float(labels.sum())
  negatives = float(len(labels) - positives)
  tpr = labels.cumsum() / positives if positives else labels.cumsum() * 0
  fpr = (1 - labels).cumsum() / negatives if negatives else (1 - labels).cumsum() * 0
  separation = tpr - fpr
  LOGGER.debug('ks separation', extra={'ks_value': float(separation.max())})
  # Positional, not label-based. The frame is built on the caller's index, which
  # is routinely non-unique after a concat, a groupby or a filter, and
  # `.loc[separation.idxmax()]` on a duplicated label returns a *Series* rather
  # than a scalar -- raising `TypeError: cannot convert the series to float`. That
  # happens after the model has been fitted, so it aborted the run rather than
  # the selection. Positions are unambiguous whatever the index looks like.
  scores = ordered[probability_col].to_numpy()
  return float(scores[int(np.argmax(separation.to_numpy()))])


def _safe_mcc(y_true: np.ndarray, prediction: np.ndarray) -> float:
  """Compute the Matthews correlation coefficient, degrading gracefully.

  Args:
    y_true: The labels.
    prediction: The binarised prediction.

  Returns:
    The coefficient, or ``0.0`` when it is undefined.
  """
  try:
    return float(matthews_corrcoef(y_true, prediction))
  except ValueError:
    return 0.0


def _round_threshold(value: float) -> float:
  """Round a threshold with precision adapted to its magnitude.

  Args:
    value: The raw threshold.

  Returns:
    The rounded threshold.
  """
  for ceiling, digits in _ROUNDING_BANDS:
    if value < ceiling:
      return float(np.round(value, digits))
  return float(np.round(value, 2))
