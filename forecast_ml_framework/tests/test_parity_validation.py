#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Parity Task 7: should_promote promotion gate.'''

import math

from forecast_ml_framework.pipelines.modeling.validation import (
  should_promote,
)


def test_cls_within_minus_five_promotes():
  '''Classification candidate at -3% deviation promotes.'''
  promote, dev = should_promote(0.97, 1.0, 'classification')
  assert promote is True
  assert dev == -3.0


def test_cls_beyond_minus_five_does_not_promote():
  '''Classification candidate at -10% deviation does not promote.'''
  promote, dev = should_promote(0.9, 1.0, 'classification')
  assert promote is False
  assert dev == -10.0


def test_cls_boundary_minus_five_promotes():
  '''Classification candidate at exactly -5% still promotes (>= -5).'''
  promote, dev = should_promote(0.95, 1.0, 'classification')
  assert promote is True
  assert dev == -5.0


def test_reg_lower_better_boundary():
  '''Regression rmse: +4.9% promotes, +5% does not (< +5).'''
  promote, dev = should_promote(104.9, 100.0, 'regression', 'rmse')
  assert promote is True
  assert dev < 5.0
  promote, dev = should_promote(105.0, 100.0, 'regression', 'rmse')
  assert promote is False
  assert dev == 5.0


def test_reg_adj_r2_higher_better():
  '''adj_r2 is higher-better: -5% promotes, -10% does not.'''
  promote, _ = should_promote(0.95, 1.0, 'regression', 'adj_r2')
  assert promote is True
  promote, _ = should_promote(0.9, 1.0, 'regression', 'ADJ_R2')
  assert promote is False


def test_old_zero_or_none_never_promotes():
  '''Old score 0/None returns (False, nan), never a division.'''
  promote, dev = should_promote(0.97, 0, 'classification')
  assert promote is False
  assert math.isnan(dev)
  promote, dev = should_promote(0.97, None, 'classification')
  assert promote is False
  assert math.isnan(dev)


def test_reg_default_metric_is_lower_better():
  '''Regression with no metric name defaults to lower-better.'''
  promote, _ = should_promote(101.0, 100.0, 'regression')
  assert promote is True
  promote, _ = should_promote(106.0, 100.0, 'regression')
  assert promote is False


def test_reg_unknown_metric_defaults_lower_better():
  '''Unknown regression metric defaults to lower-better (safe gate).'''
  promote, dev = should_promote(
    104.0, 100.0, 'regression', 'some_unknown_metric')
  assert promote is True
  assert dev == 4.0
  promote, dev = should_promote(
    106.0, 100.0, 'regression', 'some_unknown_metric')
  assert promote is False
  assert dev == 6.0


def test_reg_only_r2_family_is_higher_better():
  '''Only adj_r2/r2 family is higher-better among regression metrics.'''
  for name in ('r2', 'R2', 'adj_r2', 'adjr2'):
    promote, _ = should_promote(106.0, 100.0, 'regression', name)
    assert promote is True, name
  promote, _ = should_promote(106.0, 100.0, 'regression', '')
  assert promote is False


def test_old_negative_never_promotes():
  '''Negative old score never promotes (no signed-metric exception).'''
  promote, dev = should_promote(0.97, -1.0, 'classification')
  assert promote is False
  assert math.isnan(dev)
  promote, dev = should_promote(90.0, -100.0, 'regression', 'rmse')
  assert promote is False
  assert math.isnan(dev)


def test_non_finite_scores_never_promote():
  '''Non-finite new/old scores never promote.'''
  for new, old in (
    (float('inf'), 100.0),
    (100.0, float('inf')),
    (float('inf'), float('inf')),
    (float('nan'), 100.0),
    (100.0, float('nan')),
    (float('-inf'), 100.0),
    (100.0, float('-inf')),
  ):
    promote, dev = should_promote(new, old, 'regression', 'rmse')
    assert promote is False, (new, old)
    assert math.isnan(dev), (new, old)


def test_task_alias_reg_is_regression():
  '''Task alias reg is treated as regression (lower-better).'''
  promote, _ = should_promote(104.0, 100.0, 'reg', 'rmse')
  assert promote is True
  promote, _ = should_promote(106.0, 100.0, 'reg', 'rmse')
  assert promote is False


def test_task_alias_cls_is_classification():
  '''Task alias cls/binary is classification (higher-better).'''
  for alias in ('cls', 'binary', 'class'):
    promote, _ = should_promote(0.97, 1.0, alias)
    assert promote is True, alias
    promote, _ = should_promote(0.90, 1.0, alias)
    assert promote is False, alias


def test_task_alias_reg_higher_better_metric():
  '''reg alias with adj_r2 uses higher-better regression gate.'''
  promote, _ = should_promote(0.95, 1.0, 'reg', 'adj_r2')
  assert promote is True
  promote, _ = should_promote(0.90, 1.0, 'reg', 'adj_r2')
  assert promote is False


def test_promotion_deviation_overflow_never_promotes():
  '''Extreme scores whose deviation overflows to inf never promote.'''
  promote, dev = should_promote(1e308, 1e-308, 'classification')
  assert promote is False
  assert math.isnan(dev)
  promote, dev = should_promote(1e308, 1e-308, 'regression', 'rmse')
  assert promote is False
  assert math.isnan(dev)
