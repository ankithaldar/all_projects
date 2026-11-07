#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Retraining trigger nodes (Phase 3).

Plan only: :func:`build_retrain_plan` returns a plain dict
describing a would-be run. It never executes Kedro, never
scores, and never writes to a catalog -- the caller decides
whether to launch the ``modeling`` pipeline for the
requested env. Decision logic is deterministic: tests inject
``now`` via ``params['now']`` (a datetime or ISO string);
when no ``now`` is given the schedule check alone falls back
to the current UTC time. No other wall-clock reads happen.

Inputs (all tolerant of missing/``None`` -- a missing signal
simply does not fire its trigger, never crashes):

- ``monitoring_report``: dict with any of ``drift_alerts``
  (int count or list of alerts; aliases ``drift_alert_count``,
  ``alerts``), ``performance_decay`` (float; aliases
  ``metric_decay``; or ``baseline_metric`` / ``current_metric``
  whose difference is the decay), ``data_volume_growth``
  (float; alias ``volume_growth``; or ``current_rows`` compared
  against the last-train row count below).
- ``last_train_info``: dict with ``last_train_at`` (datetime,
  date, or ISO string) and the row count at last train
  (``last_train_rows``; aliases ``train_rows``, ``row_count``,
  ``n_rows``).
- ``params``: thresholds plus ``now`` and ``env``. Threshold
  keys (a value of ``None`` disables that trigger):
  ``drift_alert_threshold`` (alias ``drift_threshold``,
  default 0.2, aligned to ``platform.yml``/schema), 
  ``performance_decay_threshold`` (alias
  ``decay_threshold``, default 0.05),
  ``data_volume_growth_threshold`` (alias
  ``volume_growth_threshold``, default 0.20), and
  ``schedule_days`` (default 7, aligned to ``platform.yml``).

Triggers: drift (see below); performance
decay ``>`` threshold; schedule elapsed (``now`` minus
``last_train_at``) ``>=`` ``schedule_days`` days; data volume
growth ``>=`` threshold.

Drift semantics (conf-aligned): the shipped
``drift_alert_threshold`` is a 0--1 fraction/value
(``platform.yml`` 0.2), never a raw count. When the
threshold is < 1 it compares against (a) an explicit drift
value in the report (``drift_value`` / ``drift_score`` /
``drift_metric`` / ``drift_psi`` / ``max_psi`` / ``psi`` /
``ks``, or the max PSI/KS across ``features``), or (b) the
alert fraction: alert count divided by the number of
features checked (explicit denominator keys such as
``n_features``, else ``len(features)``). A bare alert
count with no value or denominator does not fire under a
fraction threshold (it cannot be interpreted as a
fraction). Explicit count thresholds >= 1 keep legacy
count semantics (``count >= threshold``).
'''

import logging
import math
from datetime import datetime, timezone

from forecast_ml_framework.core.params import (
  get_first,
  parse_datetime,
  require_dict as _require_dict,
)

LOG = logging.getLogger(__name__)

DEFAULT_DRIFT_ALERT_THRESHOLD = 0.2
DEFAULT_PERFORMANCE_DECAY_THRESHOLD = 0.05
DEFAULT_SCHEDULE_DAYS = 7
DEFAULT_DATA_VOLUME_GROWTH_THRESHOLD = 0.20
DEFAULT_PIPELINE_NAME = 'modeling'
DEFAULT_ENV = 'local'


def _threshold(params, *keys, default):
  '''Resolve a threshold; explicit None disables the trigger.

  Args:
    params: Retraining params dict.
    *keys: Accepted key names, first present wins.
    default: Default when no key is present.

  Returns:
    Float threshold, or None when disabled.
  '''
  params = params or {}
  for key in keys:
    if key in params:
      value = params[key]
      if value is None:
        return None
      try:
        number = float(value)
      except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('threshold %r must be numeric, got %r'
                         % (key, value)) from exc
      if not math.isfinite(number):
        raise ValueError('threshold %r must be numeric, got %r'
                         % (key, value))
      return number
  return float(default)


def _drift_alert_count(report):
  '''Return the drift alert count from a report, or 0.'''
  report = report or {}
  alerts = get_first(report, 'drift_alerts', 'drift_alert_count',
                'drift_alerts_count', 'alerts')
  if alerts is None:
    drift = report.get('drift')
    if isinstance(drift, dict):
      alerts = get_first(drift, 'alerts', 'alert_count', 'count')
  if alerts is None:
    return 0
  if isinstance(alerts, (list, tuple, set)):
    return len(alerts)
  if isinstance(alerts, bool):
    return int(alerts)
  if isinstance(alerts, float) and not math.isfinite(alerts):
    raise ValueError('drift_alerts must be a count or list of'
                     ' alerts, got %r' % (alerts,))
  try:
    return int(alerts)
  except (TypeError, ValueError, OverflowError) as exc:
    raise ValueError('drift_alerts must be a count or list of'
                     ' alerts, got %r' % (alerts,)) from exc


def _finite_value(raw):
  '''Return (value, unparseable) for a raw numeric value.'''
  if raw is None:
    return None, False
  try:
    value = float(raw)
  except (TypeError, ValueError, OverflowError):
    return None, True
  if not math.isfinite(value):
    return None, True
  return value, False


def _drift_value(report):
  '''Return (value, unparseable) drift value from a report.

  Looks for an explicit drift value, then the max PSI/KS
  across ``features`` stats. Returns (None, False) when no
  value is present.
  '''
  report = report or {}
  raw = get_first(report, 'drift_value', 'drift_score',
             'drift_metric', 'drift_psi', 'max_psi', 'psi',
             'ks')
  if raw is not None:
    return _finite_value(raw)
  drift = report.get('drift')
  if isinstance(drift, dict):
    raw = get_first(drift, 'value', 'score', 'metric', 'psi',
               'ks', 'drift_value')
    if raw is not None:
      return _finite_value(raw)
  features = report.get('features')
  if isinstance(features, dict) and features:
    values = []
    bad = False
    for stats in features.values():
      if not isinstance(stats, dict):
        continue
      for key in ('psi', 'ks'):
        if stats.get(key) is not None:
          value, is_bad = _finite_value(stats[key])
          if is_bad:
            bad = True
          elif value is not None:
            values.append(value)
    if values:
      return max(values), False
    if bad:
      return None, True
  return None, False


def _drift_denominator(report):
  '''Return (denominator, unparseable) features-checked count.'''
  report = report or {}
  raw = get_first(report, 'n_features', 'total_features',
             'feature_count', 'n_checked', 'features_checked',
             'drift_alerts_total', 'total', 'denominator')
  if raw is not None:
    value, bad = _finite_value(raw)
    if bad or value is None or value <= 0:
      return None, bad
    return value, False
  features = report.get('features')
  if isinstance(features, dict) and features:
    return float(len(features)), False
  drift = report.get('drift')
  if isinstance(drift, dict):
    nested = drift.get('features')
    if isinstance(nested, dict) and nested:
      return float(len(nested)), False
  return None, False


def _drift_fraction(report, count):
  '''Return (fraction, unparseable) alert fraction.

  Fraction is ``count / denominator``; (None, False) when
  no denominator is available.
  '''
  denominator, bad = _drift_denominator(report)
  if bad:
    return None, True
  if denominator is None:
    return None, False
  return count / denominator, False


def _performance_decay(report):
  '''Return (decay, unparseable) from a report.'''
  report = report or {}
  decay = get_first(report, 'performance_decay', 'metric_decay',
               'decay')
  if decay is not None:
    return _finite_value(decay)
  for holder in (report, report.get('performance'),
                 report.get('metrics')):
    if not isinstance(holder, dict):
      continue
    baseline = get_first(holder, 'baseline_metric', 'baseline',
                    'train_metric', 'reference_metric')
    current = get_first(holder, 'current_metric', 'current',
                   'test_metric', 'latest_metric')
    if baseline is not None and current is not None:
      base_value, base_bad = _finite_value(baseline)
      cur_value, cur_bad = _finite_value(current)
      if base_bad or cur_bad:
        return None, True
      return base_value - cur_value, False
  return None, False


def _data_volume_growth(report, last_train_info):
  '''Return (growth, unparseable) from a report.'''
  report = report or {}
  growth = get_first(report, 'data_volume_growth', 'volume_growth',
                'new_data_ratio', 'row_count_growth')
  if growth is not None:
    return _finite_value(growth)
  volume = report.get('data_volume')
  if isinstance(volume, dict):
    growth = get_first(volume, 'growth', 'ratio')
    if growth is not None:
      return _finite_value(growth)
  current = get_first(report, 'current_rows', 'current_row_count',
                 'row_count', 'n_rows',
                 'current_data_volume')
  trained = get_first(last_train_info, 'last_train_rows', 'train_rows',
                 'row_count', 'n_rows', 'data_volume')
  if current is None or trained is None:
    return None, False
  cur_value, cur_bad = _finite_value(current)
  trained_value, trained_bad = _finite_value(trained)
  if cur_bad or trained_bad:
    return None, True
  if trained_value <= 0:
    return None, False
  return (cur_value - trained_value) / trained_value, False


def should_retrain(monitoring_report, last_train_info, params=None):
  '''Decide whether retraining should run.

  Args:
    monitoring_report: Monitoring report dict (see module
      docstring); ``None`` is treated as empty.
    last_train_info: Last-train info dict; ``None`` is
      treated as empty.
    params: Retraining params dict with thresholds,
      ``schedule_days``, and an injectable ``now``.

  Returns:
    ``{'retrain': bool, 'reasons': [...]}`` where reasons
    holds one stable, human-readable entry per trigger
    that fired (empty when none fired).
  '''
  params = _require_dict(params, 'params')
  report = _require_dict(monitoring_report, 'monitoring_report')
  info = _require_dict(last_train_info, 'last_train_info')
  reasons = []
  warnings = []

  drift_threshold = _threshold(
      params, 'drift_alert_threshold', 'drift_threshold',
      default=DEFAULT_DRIFT_ALERT_THRESHOLD)
  if drift_threshold is not None:
    count = _drift_alert_count(report)
    if drift_threshold >= 1:
      # Legacy count semantics for explicit thresholds >= 1.
      if count >= drift_threshold:
        reasons.append(
            'drift_alerts: %d >= threshold %g'
            % (count, drift_threshold))
    else:
      # Conf-aligned fraction/value semantics (default 0.2):
      # a bare count is never compared against a fraction.
      value, value_bad = _drift_value(report)
      if value_bad:
        LOG.warning('should_retrain: unparseable metric:'
                    ' drift_value')
        warnings.append('unparseable metric: drift_value'
                        ' (severity=warning)')
      elif value is not None:
        if value >= drift_threshold:
          reasons.append(
              'drift_value: %g >= threshold %g'
              % (value, drift_threshold))
      else:
        fraction, fraction_bad = _drift_fraction(report, count)
        if fraction_bad:
          LOG.warning('should_retrain: unparseable metric:'
                      ' drift denominator')
          warnings.append('unparseable metric: drift_denominator'
                          ' (severity=warning)')
        elif fraction is not None:
          if fraction >= drift_threshold:
            reasons.append(
                'drift_alert_fraction: %g >= threshold %g'
                ' (%d alerts)' % (fraction, drift_threshold,
                                   count))
        elif count:
          LOG.warning('should_retrain: drift_alert_threshold %g'
                      ' is a fraction but the report has no drift'
                      ' value or feature denominator; bare count'
                      ' %d does not fire', drift_threshold, count)

  decay_threshold = _threshold(
      params, 'performance_decay_threshold', 'decay_threshold',
      default=DEFAULT_PERFORMANCE_DECAY_THRESHOLD)
  if decay_threshold is not None:
    decay, decay_bad = _performance_decay(report)
    if decay_bad:
      LOG.warning('should_retrain: unparseable metric:'
                  ' performance_decay')
      warnings.append('unparseable metric: performance_decay'
                      ' (severity=warning)')
    elif decay is not None and decay > decay_threshold:
      reasons.append(
          'performance_decay: %g > threshold %g'
          % (decay, decay_threshold))

  schedule_days = _threshold(
      params, 'schedule_days',
      default=DEFAULT_SCHEDULE_DAYS)
  if schedule_days is not None:
    last_at = get_first(info, 'last_train_at', 'last_trained_at',
                   'trained_at')
    if last_at is not None:
      raw_now = get_first(params, 'now', 'current_time')
      if raw_now is None:
        raw_now = get_first(info, 'now', 'current_time')
      now = (parse_datetime(raw_now) if raw_now is not None
             else datetime.now(timezone.utc))
      elapsed_days = (parse_datetime(now)
                      - parse_datetime(last_at)).days
      if elapsed_days >= schedule_days:
        reasons.append(
            'schedule_elapsed: %d days >= schedule_days %g'
            % (elapsed_days, schedule_days))

  growth_threshold = _threshold(
      params, 'data_volume_growth_threshold',
      'volume_growth_threshold', 'growth_threshold',
      default=DEFAULT_DATA_VOLUME_GROWTH_THRESHOLD)
  if growth_threshold is not None:
    growth, growth_bad = _data_volume_growth(report, info)
    if growth_bad:
      LOG.warning('should_retrain: unparseable metric:'
                  ' data_volume_growth')
      warnings.append('unparseable metric: data_volume_growth'
                      ' (severity=warning)')
    elif growth is not None and growth >= growth_threshold:
      reasons.append(
          'data_volume_growth: %g >= threshold %g'
          % (growth, growth_threshold))

  return {'retrain': bool(reasons), 'reasons': reasons + warnings}


def build_retrain_plan(decision, params=None):
  '''Build a retraining plan dict (plan only, no execution).

  This never runs Kedro and never touches a catalog: the
  returned dict merely names the pipeline and env a caller
  would launch, plus the reason(s) from ``decision``.

  Args:
    decision: Dict from :func:`should_retrain` (or any
      mapping with ``retrain`` / ``reasons``); a bare bool
      is also accepted.
    params: Retraining params dict; ``env`` (default
      ``'local'``) and ``pipeline_name`` (alias
      ``pipeline`` / ``target_pipeline``, default
      ``'modeling'``) are honoured.

  Returns:
    Dict with ``pipeline`` / ``pipeline_name``, ``env``,
    ``retrain``, ``reasons``, and ``reason`` (the reasons
    joined by ``'; '``, or ``'no_retrain'`` when clean).
  '''
  params = _require_dict(params, 'params')
  if isinstance(decision, bool):
    decision = {'retrain': decision, 'reasons': []}
  if decision is not None and not isinstance(decision, dict):
    raise ValueError('decision must be a dict, got %r' % (decision,))
  decision = decision or {}
  reasons = list(decision.get('reasons') or [])
  retrain = bool(decision.get('retrain', bool(reasons)))
  pipeline_name = get_first(params, 'pipeline_name', 'pipeline',
                       'target_pipeline') or DEFAULT_PIPELINE_NAME
  env = get_first(params, 'env', 'environment') or DEFAULT_ENV
  return {
    'pipeline': pipeline_name,
    'pipeline_name': pipeline_name,
    'env': env,
    'retrain': retrain,
    'reasons': reasons,
    'reason': '; '.join(reasons) if reasons else 'no_retrain',
  }
