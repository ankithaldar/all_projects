#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Promotion gate: pure deviation check for model promotion.

Parity: a challenger is promoted when its deviation from
the incumbent score is within tolerance -- higher-better
metrics promote at deviation >= -5%, lower-better metrics
promote at deviation < +5%. The direction comes from a
per-metric map (``metrics.LOWER_BETTER``) for BOTH tasks,
so lower-better classification metrics such as ``log_loss``
promote on improvement (regression previously flipped but
classification did not). Regression metrics default to
lower-better unless the metric is higher-better (the
``adj_r2``/``r2`` family or an explicit signed family), so
an unknown or empty regression metric cannot silently
invert the gate; classification metrics default to
higher-better unless listed in ``LOWER_BETTER``.

Signed metrics (kappa / MCC / r2 families) may legitimately
have a non-positive incumbent. For those, a non-positive
old score uses an absolute-tolerance branch: deviation is
reported in percentage points, ``(new - old) * 100``, and
promotion requires staying within 5 percentage points of
the incumbent for higher-better metrics. Non-signed metrics
with ``old <= 0`` still never promote (no division).
Non-finite new/old scores (nan/inf) never promote.
'''

import math

from forecast_ml_framework.pipelines.modeling.ntile import _task_kind

_HIGHER_BETTER_REG = {'adj_r2', 'adjr2', 'r2'}

#: Signed metrics whose incumbent may legitimately be <= 0.
_SIGNED_METRICS = {
    'kappa', 'kappa_linear', 'kappa_quadratic', 'cohen_kappa',
    'cohen_kappa_linear', 'cohen_kappa_quadratic',
    'matthews_corrcoef', 'mcc', 'r2', 'adj_r2', 'adjr2',
}

#: Absolute tolerance (percentage points) for signed baselines.
_SIGNED_TOLERANCE_PTS = 5.0


def _is_lower_better(task_name, metric):
  '''Return True when the metric is lower-better for the task.

  Args:
    task_name: Normalised task kind.
    metric: Metric name (any case, may be empty).

  Returns:
    True for lower-better metrics. Regression defaults to
    lower-better (except the r2 family); classification
    defaults to higher-better unless the metric is in
    ``metrics.LOWER_BETTER``.
  '''
  name = str(metric or '').strip().lower()
  if task_name == 'regression':
    return name not in _HIGHER_BETTER_REG
  from forecast_ml_framework.pipelines.modeling.metrics import (
    LOWER_BETTER,
  )
  return name in LOWER_BETTER


def should_promote(new_score, old_score, task, metric=''):
  '''Decide whether a challenger score should be promoted.

  Args:
    new_score: Challenger score.
    old_score: Incumbent score (0/None never promotes for
      non-signed metrics; signed metrics use the absolute
      branch below).
    task: 'classification' or 'regression'.
    metric: Metric name driving the direction map for both
      tasks (``LOWER_BETTER`` metrics are lower-better;
      regression defaults lower-better except r2 family,
      classification defaults higher-better).

  Returns:
    Tuple ``(promote, deviation_pct)`` where deviation is
    ``(new - old) / old * 100`` for positive incumbents, or
    ``(new - old) * 100`` percentage points for signed
    metrics with a non-positive incumbent.
  '''
  if old_score is None:
    return (False, float('nan'))
  try:
    old = float(old_score)
    new = float(new_score)
  except (TypeError, ValueError):
    return (False, float('nan'))
  if not math.isfinite(old) or not math.isfinite(new):
    return (False, float('nan'))
  task_name = _task_kind(task)
  lower_better = _is_lower_better(task_name, metric)
  name = str(metric or '').strip().lower()
  if old <= 0:
    if name not in _SIGNED_METRICS:
      return (False, float('nan'))
    deviation = round((new - old) * 100.0, 10)
    if not math.isfinite(deviation):
      return (False, float('nan'))
    if lower_better:
      return (bool(deviation <= _SIGNED_TOLERANCE_PTS),
              float(deviation))
    return (bool(deviation >= -_SIGNED_TOLERANCE_PTS),
            float(deviation))
  # Round to 10dp so float noise (0.95 vs 1.0) cannot push
  # an exact -5% boundary just past the gate.
  deviation = round((new - old) / old * 100.0, 10)
  if not math.isfinite(deviation):
    # Extreme magnitudes overflow the deviation to inf/nan
    # (e.g. 1e308 vs 1e-308); such a gate must never promote.
    return (False, float('nan'))
  if lower_better:
    return (bool(deviation < 5.0), float(deviation))
  return (bool(deviation >= -5.0), float(deviation))
