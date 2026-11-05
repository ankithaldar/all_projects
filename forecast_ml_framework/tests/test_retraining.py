#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Tests for retraining triggers (Phase 3).'''

from datetime import datetime, timezone

import pytest

from forecast_ml_framework.pipelines.retraining.nodes import (
  build_retrain_plan,
  should_retrain,
)
from forecast_ml_framework.pipelines.retraining.pipeline import (
  create_pipeline,
)

NOW = datetime(2026, 11, 5, tzinfo=timezone.utc)


def _params(**overrides):
  '''Return retraining params with injected now.'''
  base = {
    'now': NOW,
    'drift_alert_threshold': 2,
    'performance_decay_threshold': 0.05,
    'schedule_days': 30,
    'data_volume_growth_threshold': 0.20,
    'env': 'local',
  }
  base.update(overrides)
  return base


def _clean_report():
  '''Return a monitoring report triggering nothing.'''
  return {
    'drift_alerts': 0,
    'performance_decay': 0.01,
    'current_rows': 1000,
  }


def _clean_last_train():
  '''Return last-train info triggering nothing.'''
  return {
    'last_train_at': datetime(2026, 11, 1, tzinfo=timezone.utc),
    'last_train_rows': 1000,
  }


def test_no_trigger_is_clean():
  '''No trigger fires on a clean report.'''
  out = should_retrain(_clean_report(), _clean_last_train(), _params())
  assert out == {'retrain': False, 'reasons': []}


def test_drift_alerts_fire_alone():
  '''Drift alert count at threshold fires alone.'''
  report = _clean_report()
  report['drift_alerts'] = 2
  out = should_retrain(report, _clean_last_train(), _params())
  assert out['retrain'] is True
  assert len(out['reasons']) == 1
  assert 'drift' in out['reasons'][0]


def test_drift_alert_list_counts_entries():
  '''A list of drift alerts is counted by length.'''
  report = _clean_report()
  report['drift_alerts'] = [{'feature': 'a'}, {'feature': 'b'}]
  out = should_retrain(report, _clean_last_train(), _params())
  assert out['retrain'] is True
  assert 'drift' in out['reasons'][0]


def test_performance_decay_fires_alone():
  '''Performance decay above threshold fires alone.'''
  report = _clean_report()
  report['performance_decay'] = 0.10
  out = should_retrain(report, _clean_last_train(), _params())
  assert out['retrain'] is True
  assert len(out['reasons']) == 1
  assert 'performance_decay' in out['reasons'][0]


def test_performance_decay_at_threshold_does_not_fire():
  '''Decay equal to threshold does not fire (strict >).'''
  report = _clean_report()
  report['performance_decay'] = 0.05
  out = should_retrain(report, _clean_last_train(), _params())
  assert out == {'retrain': False, 'reasons': []}


def test_performance_decay_from_baseline_and_current():
  '''Decay is baseline minus current when no explicit decay.'''
  report = {'drift_alerts': 0, 'baseline_metric': 0.90,
            'current_metric': 0.80, 'current_rows': 1000}
  out = should_retrain(report, _clean_last_train(), _params())
  assert out['retrain'] is True
  assert 'performance_decay' in out['reasons'][0]


def test_schedule_elapsed_fires_alone():
  '''Elapsed schedule at threshold fires alone.'''
  info = _clean_last_train()
  info['last_train_at'] = datetime(2026, 10, 6, tzinfo=timezone.utc)
  out = should_retrain(_clean_report(), info, _params())
  assert out['retrain'] is True
  assert len(out['reasons']) == 1
  assert 'schedule' in out['reasons'][0]


def test_schedule_iso_string_last_train_at():
  '''ISO-string last_train_at parses for the schedule check.'''
  info = _clean_last_train()
  info['last_train_at'] = '2026-10-06T00:00:00+00:00'
  out = should_retrain(_clean_report(), info, _params())
  assert out['retrain'] is True
  assert 'schedule' in out['reasons'][0]


def test_data_volume_growth_fires_alone():
  '''Row growth at threshold fires alone.'''
  report = _clean_report()
  report['current_rows'] = 1200
  out = should_retrain(report, _clean_last_train(), _params())
  assert out['retrain'] is True
  assert len(out['reasons']) == 1
  assert 'data_volume_growth' in out['reasons'][0]


def test_data_volume_growth_explicit_value():
  '''An explicit growth value is used when present.'''
  report = _clean_report()
  report['data_volume_growth'] = 0.50
  out = should_retrain(report, _clean_last_train(), _params())
  assert out['retrain'] is True
  assert 'data_volume_growth' in out['reasons'][0]


def test_missing_inputs_are_clean_not_crashes():
  '''None / empty inputs yield a clean decision.'''
  out = should_retrain(None, None, _params())
  assert out == {'retrain': False, 'reasons': []}


def test_multiple_triggers_accumulate_reasons():
  '''All firing triggers appear in reasons.'''
  report = _clean_report()
  report['drift_alerts'] = 3
  report['performance_decay'] = 0.20
  out = should_retrain(report, _clean_last_train(), _params())
  assert out['retrain'] is True
  assert len(out['reasons']) == 2


@pytest.mark.parametrize('env', ['local', 'cloud'])
def test_build_retrain_plan_contains_reason(env):
  '''Plan names pipeline, env, and reason for both envs.'''
  decision = {'retrain': True, 'reasons': ['schedule_elapsed']}
  plan = build_retrain_plan(decision, _params(env=env))
  assert plan['env'] == env
  assert plan['reason'] == 'schedule_elapsed'
  assert plan['reasons'] == ['schedule_elapsed']
  assert plan['retrain'] is True
  assert plan['pipeline_name']
  assert plan['pipeline'] == plan['pipeline_name']


def test_build_retrain_plan_no_retrain_reason():
  '''Plan for a clean decision says no retrain.'''
  decision = {'retrain': False, 'reasons': []}
  plan = build_retrain_plan(decision, _params())
  assert plan['retrain'] is False
  assert plan['reason'] == 'no_retrain'


def test_pipeline_has_two_named_nodes():
  '''Pipeline wires should_retrain -> build_retrain_plan.'''
  pipe = create_pipeline()
  names = [n.name for n in pipe.nodes]
  assert names == ['should_retrain', 'build_retrain_plan']
  assert len(pipe.nodes) == 2
  nodes = {n.name: n for n in pipe.nodes}
  assert 'monitoring_report' in nodes['should_retrain'].inputs
  assert 'last_train_info' in nodes['should_retrain'].inputs
  assert 'params:retraining_params' in nodes['should_retrain'].inputs
  assert 'retrain_decision' in nodes['build_retrain_plan'].inputs
  assert 'retrain_plan' in nodes['build_retrain_plan'].outputs


def test_pipeline_end_to_end_via_node_funcs():
  '''Node funcs chain decision -> plan without execution.'''
  nodes = {n.name: n for n in create_pipeline().nodes}
  report = _clean_report()
  report['drift_alerts'] = 5
  decision = nodes['should_retrain'].func(
      report, _clean_last_train(), _params())
  plan = nodes['build_retrain_plan'].func(decision, _params())
  assert decision['retrain'] is True
  assert 'drift' in plan['reason']


def test_nan_decay_warns_unparseable_without_trigger():
  '''NaN performance decay warns but does not trigger retrain.'''
  report = _clean_report()
  report['performance_decay'] = float('nan')
  out = should_retrain(report, _clean_last_train(), _params())
  assert out['retrain'] is False
  assert any('unparseable metric' in r for r in out['reasons'])
  assert any('warning' in r for r in out['reasons'])


def test_nan_growth_warns_unparseable_without_trigger():
  '''NaN data volume growth warns but does not trigger.'''
  report = _clean_report()
  report['data_volume_growth'] = float('nan')
  out = should_retrain(report, _clean_last_train(), _params())
  assert out['retrain'] is False
  assert any('unparseable metric' in r for r in out['reasons'])
  assert any('warning' in r for r in out['reasons'])


def test_invalid_date_raises_value_error():
  '''Invalid last_train_at raises an informative ValueError.'''
  info = _clean_last_train()
  info['last_train_at'] = 123
  with pytest.raises(ValueError, match='cannot parse datetime'):
    should_retrain(_clean_report(), info, _params())


def test_invalid_threshold_raises_value_error():
  '''Non-numeric thresholds raise an informative ValueError.'''
  with pytest.raises(ValueError, match='threshold.*must be numeric'):
    should_retrain(_clean_report(), _clean_last_train(),
                   _params(drift_alert_threshold='x'))


def test_invalid_drift_alerts_shape_raises_value_error():
  '''Non-count drift_alerts shapes raise an informative ValueError.'''
  report = _clean_report()
  report['drift_alerts'] = {'a': 1}
  with pytest.raises(ValueError, match='drift_alerts'):
    should_retrain(report, _clean_last_train(), _params())


def test_non_dict_report_raises_value_error():
  '''Non-dict monitoring reports raise an informative ValueError.'''
  with pytest.raises(ValueError, match='monitoring_report must be'):
    should_retrain('bad', _clean_last_train(), _params())
