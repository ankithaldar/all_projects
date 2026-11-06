#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Cost / performance tracking nodes (Phase 3).

Offline estimation only: every input comes from the ``run_info``
and ``params`` dicts. No cloud API calls are made, so the same
code runs against any environment (dev/prod, GCP/AWS) with
identical arithmetic; only the configured unit prices differ.

Unit prices (config-driven, per-key documentation). Prices may
be flat in ``params`` or nested under ``unit_prices`` (or
``cost_tracking_params`` / ``cost_params``); all are read here,
never hardcoded beyond the documented defaults below.

- ``compute_price_per_hour`` -- cost per cluster runtime hour
  (USD/hour). Multiplies ``runtime_hours`` (aliases:
  ``cluster_runtime_hours``, ``compute_hours``). Aliases:
  ``price_per_compute_hour``, ``unit_price_per_hour``.
- ``data_scan_price_per_gb`` -- cost per GB of BigQuery bytes
  scanned (USD/GB). Multiplies scanned GB, computed as
  ``bytes_scanned / 1e9`` (aliases: ``bq_bytes_scanned``,
  ``data_scan_bytes``, ``bigquery_bytes_scanned``), or a direct
  ``gb_scanned`` / ``scanned_gb`` quantity when given. Aliases:
  ``price_per_gb_scanned``, ``scan_price_per_gb``.
- ``storage_price_per_gb_month`` -- cost per GB-month of saved
  artifacts (USD/GB-month). Multiplies ``artifact_gb`` (aliases:
  ``artifact_size_gb``, ``storage_gb``), times ``storage_months``
  (default 1). Aliases: ``price_per_gb_month``,
  ``storage_price_per_gb``.
- ``currency`` -- currency label for the totals
  (default ``'USD'``).

Missing quantities or prices default to 0 with a warning
(``LOG.warning``), never a crash; non-numeric or negative
values are treated the same way.

Records and summaries: :func:`track_run_cost` builds one record
per run (model / env / month + the cost estimate) and appends
it to a caller-supplied cost-log list when one is given.
:func:`summarize_costs` rolls a list of records up into totals
by model, env, and month; an empty list yields a zero summary.
'''

import logging
import math

from forecast_ml_framework.core.params import (
  require_dict as _require_dict,
)

LOG = logging.getLogger(__name__)

_BYTES_PER_GB = 1e9

_RUNTIME_KEYS = ('runtime_hours', 'cluster_runtime_hours',
                 'compute_hours')
_SCAN_BYTES_KEYS = ('bytes_scanned', 'bq_bytes_scanned',
                    'data_scan_bytes', 'bigquery_bytes_scanned')
_SCAN_GB_KEYS = ('gb_scanned', 'scanned_gb')
_STORAGE_KEYS = ('artifact_gb', 'artifact_size_gb', 'storage_gb')
_COMPUTE_PRICE_KEYS = ('compute_price_per_hour',
                       'price_per_compute_hour',
                       'unit_price_per_hour')
_SCAN_PRICE_KEYS = ('data_scan_price_per_gb', 'price_per_gb_scanned',
                    'scan_price_per_gb')
_STORAGE_PRICE_KEYS = ('storage_price_per_gb_month',
                       'price_per_gb_month', 'storage_price_per_gb')
_NESTED_KEYS = ('unit_prices', 'cost_tracking_params',
                'cost_params')


def _nested_params(params):
  '''Yield nested param dicts (unit_prices etc.), if any.'''
  for key in _NESTED_KEYS:
    nested = params.get(key)
    if isinstance(nested, dict):
      yield nested


def _lookup(sources, keys):
  '''Return (value, found) for the first key present in sources.'''
  for source in sources:
    if not isinstance(source, dict):
      continue
    for key in keys:
      if key in source and source[key] is not None:
        return source[key], True
  return None, False


def _num(value, label):
  '''Coerce value to a non-negative float, warning on failure.'''
  try:
    number = float(value)
  except (TypeError, ValueError, OverflowError):
    LOG.warning('cost_tracking: invalid numeric value for %s: %r',
                label, value)
    return 0.0
  if not math.isfinite(number):
    LOG.warning('cost_tracking: non-finite value for %s: %r',
                label, value)
    return 0.0
  if number < 0:
    LOG.warning('cost_tracking: negative value for %s: %r',
                label, value)
    return 0.0
  return number


def _quantity(run_info, params, keys, label):
  '''Resolve a quantity from run_info, falling back to params.'''
  value, found = _lookup([run_info, params], keys)
  if not found:
    LOG.warning('cost_tracking: missing field %s, defaulting to 0',
                label)
    return 0.0
  return _num(value, label)


def _price(params, keys, label):
  '''Resolve a unit price from params (flat or nested).'''
  sources = [params] + list(_nested_params(params))
  value, found = _lookup(sources, keys)
  if not found:
    LOG.warning('cost_tracking: missing unit price %s, defaulting '
                'to 0', label)
    return 0.0
  return _num(value, label)


def _currency(run_info, params):
  '''Resolve the currency label (default USD).'''
  value, found = _lookup([run_info, params], ('currency',))
  return str(value) if found else 'USD'


def _month_of(info):
  '''Derive a YYYY-MM month from a run/record dict.'''
  month = info.get('month')
  if month:
    return str(month)
  for key in ('date', 'timestamp', 'run_date', 'run_at'):
    value = info.get(key)
    if value:
      return str(value)[:7]
  return 'unknown'


def _is_estimate(params):
  '''Return True when params is a precomputed cost estimate.'''
  return isinstance(params, dict) and 'total' in params and (
    'compute_cost' in params or 'data_scan_cost' in params)


def estimate_cost(run_info, params=None):
  '''Estimate the cost of one run from dict inputs only.

  Args:
    run_info: Dict with runtime hours, BigQuery bytes scanned,
      and artifact GB (see module docstring for key aliases),
      plus optional run/model/env/month metadata.
    params: Dict with config-driven unit prices (see module
      docstring) and, as a fallback, the same quantities.

  Returns:
    Dict with ``compute_cost``, ``data_scan_cost``,
    ``storage_cost``, ``total``, and ``currency``. Missing
    fields default to 0 with a warning.
  '''
  run_info = _require_dict(run_info, 'run_info')
  params = _require_dict(params, 'params')
  runtime_hours = _quantity(run_info, params, _RUNTIME_KEYS,
                             'runtime_hours')
  compute_price = _price(params, _COMPUTE_PRICE_KEYS,
                         'compute_price_per_hour')
  scan_gb, scan_found = _lookup(
    [run_info, params], _SCAN_GB_KEYS)
  if scan_found:
    scan_gb = _num(scan_gb, 'gb_scanned')
  else:
    scan_bytes = _quantity(run_info, params, _SCAN_BYTES_KEYS,
                           'bytes_scanned')
    scan_gb = scan_bytes / _BYTES_PER_GB
  scan_price = _price(params, _SCAN_PRICE_KEYS,
                      'data_scan_price_per_gb')
  artifact_gb = _quantity(run_info, params, _STORAGE_KEYS,
                          'artifact_gb')
  months, found = _lookup([run_info, params],
                          ('storage_months', 'months'))
  months = _num(months, 'storage_months') if found else 1.0
  storage_price = _price(params, _STORAGE_PRICE_KEYS,
                         'storage_price_per_gb_month')
  compute_cost = round(runtime_hours * compute_price, 6)
  data_scan_cost = round(scan_gb * scan_price, 6)
  storage_cost = round(artifact_gb * months * storage_price, 6)
  return {
    'compute_cost': compute_cost,
    'data_scan_cost': data_scan_cost,
    'storage_cost': storage_cost,
    'total': round(compute_cost + data_scan_cost + storage_cost, 6),
    'currency': _currency(run_info, params),
  }


def track_run_cost(run_info, params=None, cost_log=None):
  '''Build one cost record for a run, optionally logging it.

  Args:
    run_info: Run metadata + usage dict (see
      :func:`estimate_cost`).
    params: Unit-price params, or a precomputed cost estimate
      (as produced by :func:`estimate_cost`, e.g. wired in by
      the pipeline) which is reused instead of recomputed.
    cost_log: Optional list; when given, the record is appended
      to it (the caller-owned cost log).

  Returns:
    Record dict with run/model/env/month metadata plus the
    cost estimate fields. Missing fields default to 0 /
    ``'unknown'`` with a warning.
  '''
  run_info = _require_dict(run_info, 'run_info')
  params = _require_dict(params, 'params')
  if _is_estimate(params):
    estimate = {key: _num(params[key], key) for key in (
      'compute_cost', 'data_scan_cost', 'storage_cost',
      'total') if key in params}
    if 'currency' in params:
      estimate['currency'] = str(params['currency'])
    estimate.setdefault('currency', _currency(run_info, {}))
  else:
    estimate = estimate_cost(run_info, params)
  record = {
    'run_id': str(run_info.get('run_id') or 'unknown'),
    'model': str(run_info.get('model')
                 or run_info.get('model_name') or 'unknown'),
    'env': str(run_info.get('env')
               or run_info.get('environment') or 'unknown'),
    'month': _month_of(run_info),
    'compute_cost': estimate.get('compute_cost', 0.0),
    'data_scan_cost': estimate.get('data_scan_cost', 0.0),
    'storage_cost': estimate.get('storage_cost', 0.0),
    'total': estimate.get('total', 0.0),
    'currency': estimate.get('currency', 'USD'),
  }
  if isinstance(cost_log, list):
    cost_log.append(record)
  return record


def summarize_costs(records):
  '''Roll cost records up into totals by model, env, and month.

  Args:
    records: Iterable of record dicts as produced by
      :func:`track_run_cost`. ``None`` is treated as empty.

  Returns:
    Dict with ``total``, ``count``, ``currency``, ``by_model``,
    ``by_env``, and ``by_month``. Empty input yields a zero
    summary. Records missing ``total`` count as 0 with a
    warning; missing model/env/month group under ``'unknown'``.
  '''
  records = list(records) if records else []
  summary = {
    'total': 0.0,
    'count': 0,
    'currency': 'USD',
    'by_model': {},
    'by_env': {},
    'by_month': {},
  }
  for record in records:
    if not isinstance(record, dict):
      LOG.warning('cost_tracking: skipping non-dict record %r',
                  record)
      continue
    if 'total' in record:
      total = _num(record['total'], 'total')
    else:
      LOG.warning('cost_tracking: missing field total, defaulting '
                  'to 0')
      total = 0.0
    summary['total'] = round(summary['total'] + total, 6)
    summary['count'] += 1
    if summary['count'] == 1 and record.get('currency'):
      summary['currency'] = str(record['currency'])
    model = str(record.get('model')
                or record.get('model_name') or 'unknown')
    env = str(record.get('env')
              or record.get('environment') or 'unknown')
    month = _month_of(record)
    for bucket, key in ((summary['by_model'], model),
                        (summary['by_env'], env),
                        (summary['by_month'], month)):
      bucket[key] = round(bucket.get(key, 0.0) + total, 6)
  return summary
