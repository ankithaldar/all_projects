#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Tests for A/B and shadow serving router.'''


class FakeModel:
  '''Fake model returning a fixed prediction.'''

  def __init__(self, value):
    self.value = value

  def predict(self, features):
    '''Return fixed prediction regardless of features.'''
    del features
    return [self.value]


def _make_router(mode='ab', split=0.1):
  from forecast_ml_framework.serving.ab_testing import ABRouter

  champion = FakeModel(10.0)
  challenger = FakeModel(20.0)
  return ABRouter(
    lambda: champion, lambda: challenger, traffic_split=split, mode=mode
  )


def test_route_is_deterministic_for_same_request_id():
  router = _make_router(mode='ab')
  first = router.route('req-1', [[1.0]])
  second = router.route('req-1', [[1.0]])
  assert first['variant'] == second['variant']
  assert first['prediction'] == second['prediction']
  assert first['variant'] in ('champion', 'challenger')


def test_traffic_split_approx_ten_percent_challenger():
  router = _make_router(mode='ab', split=0.1)
  challenger = sum(
    1
    for i in range(1000)
    if router.route(f'req-{i}', [[1.0]])['variant'] == 'challenger'
  )
  fraction = challenger / 1000
  assert 0.07 <= fraction <= 0.13


def test_ab_mode_returns_selected_variant_prediction():
  router = _make_router(mode='ab', split=0.5)
  seen = {}
  for i in range(200):
    result = router.route(f'req-{i}', [[1.0]])
    seen.setdefault(result['variant'], result['prediction'])
  assert seen['champion'] == 10.0
  assert seen['challenger'] == 20.0


def test_shadow_mode_returns_champion_and_logs_both():
  router = _make_router(mode='shadow')
  result = router.route('req-shadow', [[1.0]])
  assert result['prediction'] == 10.0
  assert result['variant'] == 'champion'
  assert len(router.logs) == 1
  log = router.logs[0]
  assert log['champion_prediction'] == 10.0
  assert log['challenger_prediction'] == 20.0
  assert log['champion_latency'] >= 0
  assert log['challenger_latency'] >= 0


def test_compare_variants_aggregates_correctly():
  from forecast_ml_framework.serving.ab_testing import compare_variants

  logs = [
    {'variant': 'champion', 'prediction': 10.0, 'latency': 0.01},
    {'variant': 'champion', 'prediction': 20.0, 'latency': 0.03},
    {'variant': 'challenger', 'prediction': 30.0, 'latency': 0.05},
    {'variant': 'challenger', 'prediction': 50.0, 'latency': 0.07},
  ]
  result = compare_variants(logs)
  assert result['champion']['count'] == 2
  assert result['challenger']['count'] == 2
  assert result['champion']['mean_prediction'] == 15.0
  assert result['challenger']['mean_prediction'] == 40.0
  assert result['champion']['latency_p50'] == 0.01
  assert result['champion']['latency_p95'] == 0.03
  assert result['challenger']['latency_p50'] == 0.05
  assert result['challenger']['latency_p95'] == 0.07


def test_compare_variants_from_shadow_logs_counts_both():
  from forecast_ml_framework.serving.ab_testing import compare_variants

  router = _make_router(mode='shadow')
  router.route('a', [[1.0]])
  router.route('b', [[1.0]])
  result = compare_variants(router.logs)
  assert result['champion']['count'] == 2
  assert result['challenger']['count'] == 2
  assert result['champion']['mean_prediction'] == 10.0
  assert result['challenger']['mean_prediction'] == 20.0


def test_compare_variants_empty_logs():
  from forecast_ml_framework.serving.ab_testing import compare_variants

  result = compare_variants([])
  assert result['champion']['count'] == 0
  assert result['challenger']['count'] == 0


def test_invalid_mode_rejected():
  import pytest
  from forecast_ml_framework.serving.ab_testing import ABRouter

  with pytest.raises(ValueError, match='mode'):
    ABRouter(lambda: FakeModel(1.0), lambda: FakeModel(2.0), mode='bogus')


def test_invalid_traffic_split_rejected():
  import pytest
  from forecast_ml_framework.serving.ab_testing import ABRouter

  with pytest.raises(ValueError, match='traffic_split'):
    ABRouter(lambda: FakeModel(1.0), lambda: FakeModel(2.0), traffic_split=1.5)


def test_salt_changes_bucket():
  from forecast_ml_framework.serving.ab_testing import ABRouter, _bucket

  left = ABRouter(
    lambda: FakeModel(1.0), lambda: FakeModel(2.0),
    traffic_split=0.5, mode='ab', salt='salt-a')
  right = ABRouter(
    lambda: FakeModel(1.0), lambda: FakeModel(2.0),
    traffic_split=0.5, mode='ab', salt='salt-b')
  ids = [f'req-{i}' for i in range(200)]
  assert any(
    _bucket(i, 'salt-a') != _bucket(i, 'salt-b') for i in ids)
  assert any(
    left._variant(i) != right._variant(i) for i in ids)
  # same salt is deterministic
  assert left._variant('req-1') == left._variant('req-1')


def test_compare_variants_skips_none_and_malformed():
  from forecast_ml_framework.serving.ab_testing import compare_variants

  logs = [
    {'variant': 'champion', 'prediction': None, 'latency': 0.01},
    {'variant': 'champion', 'prediction': 'bad', 'latency': 0.01},
    {'variant': 'champion', 'prediction': [1, 2], 'latency': 0.01},
    {'variant': 'champion', 'prediction': 10.0, 'latency': 0.01},
    {'variant': 'challenger', 'prediction': None, 'latency': 0.02},
    {'variant': 'challenger', 'prediction': 20.0, 'latency': 'bad'},
    {'champion_prediction': None, 'challenger_prediction': 'bad'},
    None,
    'malformed',
    {},
    {'variant': 'unknown', 'prediction': 5.0, 'latency': 0.01},
  ]
  result = compare_variants(logs)
  assert result['champion']['count'] == 1
  assert result['champion']['mean_prediction'] == 10.0
  assert result['challenger']['count'] == 1
  assert result['challenger']['mean_prediction'] == 20.0


def test_shadow_challenger_exception_returns_champion():
  from forecast_ml_framework.serving.ab_testing import ABRouter

  class BadModel:
    def predict(self, features):
      raise RuntimeError('challenger boom')

  router = ABRouter(
    lambda: FakeModel(10.0), lambda: BadModel(), mode='shadow')
  result = router.route('req-shadow-bad', [[1.0]])
  assert result == {'prediction': 10.0, 'variant': 'champion'}
  assert len(router.logs) == 1
  assert router.logs[0]['champion_prediction'] == 10.0
  assert router.logs[0]['challenger_prediction'] is None
