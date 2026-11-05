#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Tests for cost / performance tracking (Phase 3).'''

import logging

import pytest

from forecast_ml_framework.pipelines.cost_tracking.nodes import (
  estimate_cost,
  summarize_costs,
  track_run_cost,
)
from forecast_ml_framework.pipelines.cost_tracking.pipeline import (
  create_pipeline,
)


def _prices(**overrides):
  '''Return unit-price params for tests.'''
  base = {
    'compute_price_per_hour': 2.0,
    'data_scan_price_per_gb': 0.5,
    'storage_price_per_gb_month': 0.1,
    'currency': 'USD',
  }
  base.update(overrides)
  return base


def _run_info(**overrides):
  '''Return a run_info dict for tests.'''
  base = {
    'run_id': 'run-1',
    'model': 'forecast-v1',
    'env': 'prod',
    'month': '2026-11',
    'runtime_hours': 10.0,
    'bytes_scanned': 2_000_000_000,
    'artifact_gb': 50.0,
  }
  base.update(overrides)
  return base


def test_estimate_cost_arithmetic_correct():
  '''Compute, scan, and storage costs use the unit prices.'''
  out = estimate_cost(_run_info(), _prices())
  # 10h * $2 = $20; 2 GB * $0.5 = $1; 50 GB-mo * $0.1 = $5.
  assert out['compute_cost'] == pytest.approx(20.0)
  assert out['data_scan_cost'] == pytest.approx(1.0)
  assert out['storage_cost'] == pytest.approx(5.0)
  assert out['total'] == pytest.approx(26.0)
  assert out['currency'] == 'USD'
  assert set(out) == {
    'compute_cost', 'data_scan_cost', 'storage_cost',
    'total', 'currency'}


@pytest.mark.parametrize('env', ['dev', 'prod'])
def test_estimate_cost_both_envs(env):
  '''Cost arithmetic is identical across environments.'''
  out = estimate_cost(_run_info(env=env), _prices())
  assert out['total'] == pytest.approx(26.0)
  assert out['currency'] == 'USD'


def test_estimate_cost_quantities_from_params_fallback():
  '''Quantities may live in params when absent from run_info.'''
  params = _prices(runtime_hours=4.0, bytes_scanned=1_000_000_000,
                   artifact_gb=10.0)
  out = estimate_cost({'model': 'm'}, params)
  assert out['compute_cost'] == pytest.approx(8.0)
  assert out['data_scan_cost'] == pytest.approx(0.5)
  assert out['storage_cost'] == pytest.approx(1.0)
  assert out['total'] == pytest.approx(9.5)


def test_estimate_cost_nested_unit_prices():
  '''Unit prices may be nested under ``unit_prices``.'''
  params = {'unit_prices': _prices()}
  out = estimate_cost(_run_info(), params)
  assert out['total'] == pytest.approx(26.0)


def test_estimate_cost_missing_fields_default_zero_with_warning(caplog):
  '''Missing quantities and prices default to 0 with a warning.'''
  with caplog.at_level(logging.WARNING):
    out = estimate_cost({}, {})
  assert out == {
    'compute_cost': 0.0, 'data_scan_cost': 0.0,
    'storage_cost': 0.0, 'total': 0.0, 'currency': 'USD'}
  assert any(r.levelno >= logging.WARNING for r in caplog.records)


def test_estimate_cost_missing_one_field_warns_and_zeroes_it(caplog):
  '''One missing quantity zeroes only its own component.'''
  info = _run_info()
  del info['bytes_scanned']
  with caplog.at_level(logging.WARNING):
    out = estimate_cost(info, _prices())
  assert out['data_scan_cost'] == 0.0
  assert out['compute_cost'] == pytest.approx(20.0)
  assert out['total'] == pytest.approx(25.0)
  assert any('bytes_scanned' in r.message for r in caplog.records)


def test_track_run_cost_returns_record():
  '''track_run_cost returns a record carrying the estimate.'''
  record = track_run_cost(_run_info(), _prices())
  assert record['run_id'] == 'run-1'
  assert record['model'] == 'forecast-v1'
  assert record['env'] == 'prod'
  assert record['month'] == '2026-11'
  assert record['total'] == pytest.approx(26.0)
  assert record['compute_cost'] == pytest.approx(20.0)
  assert record['currency'] == 'USD'


def test_track_run_cost_appends_to_log_list():
  '''track_run_cost appends its record to a supplied log list.'''
  log = []
  record = track_run_cost(_run_info(), _prices(), log)
  assert log == [record]
  assert record['total'] == pytest.approx(26.0)


def test_track_run_cost_missing_fields_warns(caplog):
  '''track_run_cost with sparse run_info warns, defaults 0.'''
  with caplog.at_level(logging.WARNING):
    record = track_run_cost({'model': 'm'}, _prices())
  assert record['total'] == 0.0
  assert record['env'] == 'unknown'
  assert any(r.levelno >= logging.WARNING for r in caplog.records)


def test_summarize_costs_empty_records_zero_summary():
  '''Empty (or None) records produce a zero summary.'''
  for empty in ([], None):
    out = summarize_costs(empty)
    assert out['total'] == 0.0
    assert out['count'] == 0
    assert out['currency'] == 'USD'
    assert out['by_model'] == {}
    assert out['by_env'] == {}
    assert out['by_month'] == {}


def test_summarize_costs_totals_by_model_env_month():
  '''Totals roll up by model, env, and month (both envs).'''
  records = [
    track_run_cost(_run_info(run_id='r1', model='a', env='dev',
                             month='2026-11'), _prices()),
    track_run_cost(_run_info(run_id='r2', model='a', env='prod',
                             month='2026-11'), _prices()),
    track_run_cost(_run_info(run_id='r3', model='b', env='prod',
                             month='2026-12'), _prices()),
  ]
  out = summarize_costs(records)
  assert out['total'] == pytest.approx(78.0)
  assert out['count'] == 3
  assert out['currency'] == 'USD'
  assert out['by_model'] == {'a': pytest.approx(52.0),
                             'b': pytest.approx(26.0)}
  assert out['by_env'] == {'dev': pytest.approx(26.0),
                           'prod': pytest.approx(52.0)}
  assert out['by_month'] == {'2026-11': pytest.approx(52.0),
                             '2026-12': pytest.approx(26.0)}


def test_summarize_costs_missing_total_defaults_zero_with_warning(caplog):
  '''Records missing ``total`` count as 0 with a warning.'''
  with caplog.at_level(logging.WARNING):
    out = summarize_costs([{'model': 'a', 'env': 'prod',
                            'month': '2026-11'}])
  assert out['total'] == 0.0
  assert out['count'] == 1
  assert any('total' in r.message for r in caplog.records)


def test_summarize_costs_derives_month_from_date():
  '''Month is derived as YYYY-MM from a date/timestamp field.'''
  out = summarize_costs([{'total': 3.0, 'model': 'a', 'env': 'dev',
                          'date': '2026-11-05'}])
  assert out['by_month'] == {'2026-11': pytest.approx(3.0)}


def test_pipeline_has_three_named_nodes():
  '''Pipeline wires estimate -> track -> summarize.'''
  pipe = create_pipeline()
  names = [n.name for n in pipe.nodes]
  # summarize is an independent branch, so Kedro may reorder it.
  assert set(names) == {
    'estimate_cost', 'track_run_cost', 'summarize_costs'}
  assert len(pipe.nodes) == 3
  nodes = {n.name: n for n in pipe.nodes}
  assert nodes['estimate_cost'].inputs == [
    'run_info', 'params:cost_tracking_params']
  assert nodes['track_run_cost'].inputs == ['run_info', 'cost_estimate']
  assert nodes['summarize_costs'].inputs == ['cost_log']


def test_pipeline_end_to_end_via_node_funcs():
  '''Node funcs chain estimate -> track -> summarize.'''
  nodes = {n.name: n for n in create_pipeline().nodes}
  estimate = nodes['estimate_cost'].func(_run_info(), _prices())
  record = nodes['track_run_cost'].func(_run_info(), estimate)
  summary = nodes['summarize_costs'].func([record])
  assert estimate['total'] == pytest.approx(26.0)
  assert record['total'] == pytest.approx(26.0)
  assert summary['total'] == pytest.approx(26.0)
  assert summary['count'] == 1


def test_track_run_cost_precomputed_negative_clamped():
  '''Precomputed negative costs are clamped to 0.'''
  record = track_run_cost(
    {'model': 'm'},
    {'total': -10.0, 'compute_cost': -5.0, 'data_scan_cost': 1.0,
     'storage_cost': 1.0, 'currency': 'USD'})
  assert record['total'] == 0.0
  assert record['compute_cost'] == 0.0


def test_track_run_cost_precomputed_inf_nan_rejected(caplog):
  '''Precomputed inf/NaN costs become 0 with a warning.'''
  with caplog.at_level(logging.WARNING):
    record = track_run_cost(
      {'model': 'm'},
      {'total': float('inf'), 'compute_cost': float('nan')})
  assert record['total'] == 0.0
  assert record['compute_cost'] == 0.0
  assert any(r.levelno >= logging.WARNING for r in caplog.records)


def test_estimate_cost_inf_nan_quantities_rejected(caplog):
  '''Inf/NaN quantities are treated as 0, never poison totals.'''
  with caplog.at_level(logging.WARNING):
    out = estimate_cost(
      _run_info(runtime_hours=float('inf')), _prices())
  assert out['compute_cost'] == 0.0
  assert out['total'] == pytest.approx(6.0)


def test_track_run_cost_non_dict_run_info_raises_value_error():
  '''Non-dict run_info raises an informative ValueError.'''
  with pytest.raises(ValueError, match='run_info must be a dict'):
    track_run_cost('bad', _prices())


def test_estimate_cost_non_dict_params_raises_value_error():
  '''Non-dict params raises an informative ValueError.'''
  with pytest.raises(ValueError, match='params must be a dict'):
    estimate_cost(_run_info(), 'bad')
