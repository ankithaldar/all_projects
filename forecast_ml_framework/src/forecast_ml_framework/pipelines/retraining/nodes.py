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
  default 1), ``performance_decay_threshold`` (alias
  ``decay_threshold``, default 0.05),
  ``data_volume_growth_threshold`` (alias
  ``volume_growth_threshold``, default 0.20), and
  ``schedule_days`` (default 30).

Triggers: drift alerts count ``>=`` threshold; performance
decay ``>`` threshold; schedule elapsed (``now`` minus
``last_train_at``) ``>=`` ``schedule_days`` days; data volume
growth ``>=`` threshold.
'''

import logging
import math
from datetime import date, datetime, timezone

LOG = logging.getLogger(__name__)

DEFAULT_DRIFT_ALERT_THRESHOLD = 1
DEFAULT_PERFORMANCE_DECAY_THRESHOLD = 0.05
DEFAULT_SCHEDULE_DAYS = 30
DEFAULT_DATA_VOLUME_GROWTH_THRESHOLD = 0.20
DEFAULT_PIPELINE_NAME = 'modeling'
DEFAULT_ENV = 'local'


def _require_dict(value, label):
  '''Return value as a dict, raising an informative ValueError.'''
  if value is None:
    return {}
  if not isinstance(value, dict):
    raise ValueError('%s must be a dict, got %r' % (label, value))
  return value


def _get(mapping, *keys):
  '''Return the first present, non-None value among keys.'''
  mapping = mapping or {}
  for key in keys:
    if key in mapping and mapping[key] is not None:
      return mapping[key]
  return None


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


def _parse_datetime(value):
  '''Parse a datetime/date/ISO-string into an aware datetime.

  Args:
    value: Datetime, date, or ISO-format string. Naive
      values are treated as UTC.

  Returns:
    Aware datetime.

  Raises:
    ValueError: If the value cannot be parsed.
  '''
  if isinstance(value, datetime):
    parsed = value
  elif isinstance(value, date):
    parsed = datetime(value.year, value.month, value.day)
  elif isinstance(value, str):
    text = value.strip().replace('Z', '+00:00')
    try:
      parsed = datetime.fromisoformat(text)
    except ValueError as exc:
      raise ValueError(
        'cannot parse datetime value %r' % (value,)) from exc
  else:
    raise ValueError('cannot parse datetime value %r' % (value,))
  if parsed.tzinfo is None:
    parsed = parsed.replace(tzinfo=timezone.utc)
  return parsed


def _drift_alert_count(report):
  '''Return the drift alert count from a report, or 0.'''
  report = report or {}
  alerts = _get(report, 'drift_alerts', 'drift_alert_count',
                'drift_alerts_count', 'alerts')
  if alerts is None:
    drift = report.get('drift')
    if isinstance(drift, dict):
      alerts = _get(drift, 'alerts', 'alert_count', 'count')
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


def _performance_decay(report):
  '''Return (decay, unparseable) from a report.'''
  report = report or {}
  decay = _get(report, 'performance_decay', 'metric_decay',
               'decay')
  if decay is not None:
    return _finite_value(decay)
  for holder in (report, report.get('performance'),
                 report.get('metrics')):
    if not isinstance(holder, dict):
      continue
    baseline = _get(holder, 'baseline_metric', 'baseline',
                    'train_metric', 'reference_metric')
    current = _get(holder, 'current_metric', 'current',
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
  growth = _get(report, 'data_volume_growth', 'volume_growth',
                'new_data_ratio', 'row_count_growth')
  if growth is not None:
    return _finite_value(growth)
  volume = report.get('data_volume')
  if isinstance(volume, dict):
    growth = _get(volume, 'growth', 'ratio')
    if growth is not None:
      return _finite_value(growth)
  current = _get(report, 'current_rows', 'current_row_count',
                 'row_count', 'n_rows',
                 'current_data_volume')
  trained = _get(last_train_info, 'last_train_rows', 'train_rows',
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
    if count >= drift_threshold:
      reasons.append(
          'drift_alerts: %d >= threshold %g'
          % (count, drift_threshold))

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
    last_at = _get(info, 'last_train_at', 'last_trained_at',
                   'trained_at')
    if last_at is not None:
      raw_now = _get(params, 'now', 'current_time')
      if raw_now is None:
        raw_now = _get(info, 'now', 'current_time')
      now = (_parse_datetime(raw_now) if raw_now is not None
             else datetime.now(timezone.utc))
      elapsed_days = (_parse_datetime(now)
                      - _parse_datetime(last_at)).days
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
  pipeline_name = _get(params, 'pipeline_name', 'pipeline',
                       'target_pipeline') or DEFAULT_PIPELINE_NAME
  env = _get(params, 'env', 'environment') or DEFAULT_ENV
  return {
    'pipeline': pipeline_name,
    'pipeline_name': pipeline_name,
    'env': env,
    'retrain': retrain,
    'reasons': reasons,
    'reason': '; '.join(reasons) if reasons else 'no_retrain',
  }
