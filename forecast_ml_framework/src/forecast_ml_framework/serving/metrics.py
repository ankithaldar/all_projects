#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Prometheus /metrics producer for the serving app.

Single responsibility: collect HTTP timings, /predict
counters, and drift gauges in process, and render them
in Prometheus text exposition format (version 0.0.4).
No third-party metrics client is used, so the producer
adds no dependency.

Drift source (documented choice): drift gauges are
updated by an explicit monitoring-pipeline hook --
:meth:`MetricsRegistry.update_from_monitoring_report`
called with the dict returned by
``build_monitoring_report`` -- not by reading the
``monitoring_report`` dataset on every scrape.
Reading a dataset file per scrape would add file I/O
to the request path and couple scrape latency to
artifact storage; the hook keeps /metrics a pure
in-process render of the last reported drift, and
the gauges hold their last value (0.0 before the
first report) between pipeline runs.
'''

import copy
import math
import threading

#: Histogram bucket upper bounds in seconds.
HTTP_DURATION_BUCKETS = (
  0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)

#: Prometheus text exposition content type.
CONTENT_TYPE = 'text/plain; version=0.0.4; charset=utf-8'


def _escape_label(value):
  '''Escape a label value for Prometheus text format.'''
  text = str(value)
  text = text.replace('\\', '\\\\')
  text = text.replace('"', '\\"')
  return text.replace('\n', '\\n')


def _labels(pairs):
  '''Render label pairs as ``{k="v",...}`` ('' if none).'''
  if not pairs:
    return ''
  inner = ','.join(
    f'{name}="{_escape_label(value)}"' for name, value in pairs)
  return '{' + inner + '}'


def _number(value):
  '''Render a float Prometheus-style (no exponent noise).'''
  return f'{float(value):g}'


def _check_drift(value, name):
  '''Return value as float, rejecting bad drift input.

  Args:
    value: Candidate drift statistic.
    name: Statistic name for error messages.

  Returns:
    Drift statistic as a non-negative finite float.

  Raises:
    ValueError: When value is non-numeric, negative,
      or non-finite.
  '''
  if isinstance(value, bool):
    raise ValueError(f'{name} must be a finite number')
  try:
    number = float(value)
  except (TypeError, ValueError, OverflowError):
    raise ValueError(f'{name} must be a finite number') from None
  if not math.isfinite(number) or number < 0:
    raise ValueError(f'{name} must be a finite number')
  return number


def _finite_stat(stats, key):
  '''Return a non-negative finite stat, else None.

  Drift statistics are non-negative by definition, so
  a negative value is treated as missing rather than
  returned to the caller: the monitoring hook then
  leaves the gauges unchanged instead of raising via
  ``set_drift`` and crashing the pipeline.
  '''
  if not isinstance(stats, dict):
    return None
  value = stats.get(key)
  if isinstance(value, bool):
    return None
  try:
    number = float(value)
  except (TypeError, ValueError, OverflowError):
    return None
  if math.isfinite(number) and number >= 0:
    return number
  return None


class MetricsRegistry:
  '''In-process registry of the emitted metric series.

  Thread-safe: ``observe_http``, ``record_predict``,
  ``set_drift``, and ``render`` may be called from
  request and pipeline threads concurrently. Each
  app instance owns one registry, so tests and
  processes never share counters accidentally.

  Args:
    buckets: Histogram bucket upper bounds in
      seconds; defaults to HTTP_DURATION_BUCKETS.
  '''

  def __init__(self, buckets=HTTP_DURATION_BUCKETS):
    '''Initialise empty series and zeroed gauges.'''
    self._buckets = tuple(float(bucket) for bucket in buckets)
    self._lock = threading.Lock()
    self._histograms = {}
    self._predict_requests = 0
    self._predict_errors = 0
    self._drift_psi = 0.0
    self._drift_ks = 0.0

  def observe_http(self, duration_seconds, route, code):
    '''Record one HTTP request duration.

    Args:
      duration_seconds: Request duration in seconds.
      route: Request path, e.g. ``/predict``.
      code: HTTP status code.

    Raises:
      ValueError: When duration is negative or
        non-finite.
    '''
    duration = _check_drift(duration_seconds, 'duration_seconds')
    key = (str(route), str(int(code)))
    with self._lock:
      entry = self._histograms.setdefault(
        key, {'buckets': [0] * len(self._buckets),
              'sum': 0.0, 'count': 0})
      for index, bound in enumerate(self._buckets):
        if duration <= bound:
          entry['buckets'][index] += 1
      entry['sum'] += duration
      entry['count'] += 1

  def record_predict(self, code):
    '''Count one /predict attempt by status code.

    Args:
      code: HTTP status code of the attempt; codes
        400 and above also count as errors.
    '''
    with self._lock:
      self._predict_requests += 1
      if int(code) >= 400:
        self._predict_errors += 1

  def set_drift(self, psi, ks):
    '''Set the drift gauges directly.

    Args:
      psi: Prediction PSI statistic.
      ks: Prediction KS statistic.

    Raises:
      ValueError: When either value is negative or
        non-finite.
    '''
    psi_value = _check_drift(psi, 'psi')
    ks_value = _check_drift(ks, 'ks')
    with self._lock:
      self._drift_psi = psi_value
      self._drift_ks = ks_value

  def update_from_monitoring_report(self, report):
    '''Refresh drift gauges from a monitoring report.

    Uses the ``prediction`` section when present,
    otherwise the maximum PSI/KS across ``features``
    entries, so dashboards keep a single series.
    Reports without any non-negative finite drift
    (negative values count as missing) leave the
    gauges unchanged and never raise.

    Args:
      report: Dict from ``build_monitoring_report``
        (or ``compute_drift`` output).

    Returns:
      True when the gauges were updated, False when
      the report carried no non-negative finite drift
      values.
    '''
    if not isinstance(report, dict) or not report:
      return False
    prediction = report.get('prediction')
    psi = _finite_stat(prediction, 'psi')
    ks = _finite_stat(prediction, 'ks')
    if psi is None or ks is None:
      features = report.get('features')
      stats = features if isinstance(features, dict) else report
      psi_values = []
      ks_values = []
      for entry in stats.values():
        psi_value = _finite_stat(entry, 'psi')
        ks_value = _finite_stat(entry, 'ks')
        if psi_value is not None:
          psi_values.append(psi_value)
        if ks_value is not None:
          ks_values.append(ks_value)
      psi = max(psi_values) if psi_values else None
      ks = max(ks_values) if ks_values else None
    if psi is None or ks is None:
      return False
    self.set_drift(psi, ks)
    return True

  def render(self):
    '''Render all series in Prometheus text format.

    Histogram entries are deep-copied under the lock
    and rendered from the copy, so concurrent
    requests cannot tear a single render.

    Returns:
      Exposition text ending with a newline. The
      counter and gauge series are always present
      (zeroed before any traffic or drift report);
      histogram series appear per observed
      (route, code) pair.
    '''
    lines = [
      '# HELP http_request_duration_seconds '
      'Duration of HTTP requests in seconds.',
      '# TYPE http_request_duration_seconds histogram',
    ]
    with self._lock:
      histograms = copy.deepcopy(self._histograms)
      requests = self._predict_requests
      errors = self._predict_errors
      psi = self._drift_psi
      ks = self._drift_ks
    for (route, code), entry in sorted(histograms.items()):
      base = [('code', code), ('route', route)]
      total = entry['count']
      summed = _number(entry['sum'])
      for bound, count in zip(self._buckets, entry['buckets']):
        lines.append(
          'http_request_duration_seconds_bucket'
          + _labels(base + [('le', _number(bound))])
          + f' {count}')
      lines.append(
        'http_request_duration_seconds_bucket'
        + _labels(base + [('le', '+Inf')])
        + f' {total}')
      lines.append(
        'http_request_duration_seconds_sum'
        + _labels(base) + f' {summed}')
      lines.append(
        'http_request_duration_seconds_count'
        + _labels(base) + f' {total}')
    lines.extend([
      '# HELP predict_requests_total Total /predict requests.',
      '# TYPE predict_requests_total counter',
      f'predict_requests_total {requests}',
      '# HELP predict_errors_total Failed /predict requests.',
      '# TYPE predict_errors_total counter',
      f'predict_errors_total {errors}',
      '# HELP drift_psi Population stability index drift.',
      '# TYPE drift_psi gauge',
      'drift_psi' + _labels([('feature', 'prediction')])
      + f' {_number(psi)}',
      '# HELP drift_ks Kolmogorov-Smirnov drift statistic.',
      '# TYPE drift_ks gauge',
      'drift_ks' + _labels([('feature', 'prediction')])
      + f' {_number(ks)}',
    ])
    return '\n'.join(lines) + '\n'
