#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Promotion gate: pure deviation check for model promotion.

Parity: a challenger is promoted when its deviation from
the incumbent score is within tolerance -- classification
(and higher-better regression metrics such as ``adj_r2``)
promotes at deviation >= -5%, lower-better regression
metrics promote at deviation < +5%. An old score of 0 or
None never promotes (no division). This function only
returns the decision; calling the registry stays elsewhere.

Regression metrics default to lower-better; only the
``adj_r2``/``r2`` family is treated as higher-better, so
an unknown or empty regression metric cannot silently
invert the gate. A non-positive old score (``old <= 0``)
never promotes: no signed-metric exception is documented,
and a negative incumbent would invert the deviation sign.
Non-finite new/old scores (nan/inf) never promote.
'''

import math

from forecast_ml_framework.pipelines.modeling.ntile import _task_kind

_HIGHER_BETTER_REG = {'adj_r2', 'adjr2', 'r2'}


def should_promote(new_score, old_score, task, metric=''):
  '''Decide whether a challenger score should be promoted.

  Args:
    new_score: Challenger score.
    old_score: Incumbent score (0/None never promotes).
    task: 'classification' or 'regression'.
    metric: Metric name; regression defaults to
      lower-better unless the metric is higher-better
      (``adj_r2``/``r2``). Unknown/empty regression
      metrics are lower-better. ``old <= 0`` or any
      non-finite score never promotes.

  Returns:
    Tuple ``(promote, deviation_pct)`` where deviation is
    ``(new - old) / old * 100``.
  '''
  if old_score is None:
    return (False, float('nan'))
  try:
    old = float(old_score)
    new = float(new_score)
  except (TypeError, ValueError):
    return (False, float('nan'))
  if old <= 0 or not math.isfinite(old) or not math.isfinite(new):
    return (False, float('nan'))
  # Round to 10dp so float noise (0.95 vs 1.0) cannot push
  # an exact -5% boundary just past the gate.
  deviation = round((new - old) / old * 100.0, 10)
  if not math.isfinite(deviation):
    # Extreme magnitudes overflow the deviation to inf/nan
    # (e.g. 1e308 vs 1e-308); such a gate must never promote.
    return (False, float('nan'))
  task_name = _task_kind(task)
  if task_name == 'regression':
    name = str(metric or '').strip().lower()
    # Only the r2 family is higher-better; every other
    # regression metric (known lower-better, unknown, or
    # empty) defaults to lower-better so an unrecognised
    # metric name cannot invert the promotion gate.
    lower_better = name not in _HIGHER_BETTER_REG
    if lower_better:
      return (bool(deviation < 5.0), float(deviation))
    return (bool(deviation >= -5.0), float(deviation))
  return (bool(deviation >= -5.0), float(deviation))
