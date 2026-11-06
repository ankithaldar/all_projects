#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Smoke defaults for local train (clean-ux Task 10).'''

import datetime
from pathlib import Path

import yaml


def _root():
  '''Return project root.'''
  return Path(__file__).resolve().parents[1]


def test_local_models_to_use_is_sklearn_only():
  '''Local overlay trains sklearn only for smoke.'''
  local = yaml.safe_load(
    (_root() / 'conf/local/parameters.yml').read_text())
  assert local['models_to_use'] == ['sklearn']


def test_base_models_to_use_unchanged():
  '''Base models_to_use keeps all five models.'''
  base = yaml.safe_load(
    (_root() / 'conf/base/parameters/modeling.yml').read_text())
  assert set(base['models_to_use']) == {
    'h2o_automl', 'xgboost', 'lightgbm', 'catboost', 'sklearn'}


def test_run_date_is_recent():
  '''global run_date is recent relative to today (no hardcoded date).

  TIME-BOMB NOTE: the previous version hardcoded
  datetime.date(2026, 11, 6) as "today", so it passed on that
  date and silently became wrong every day after (and would
  fail once run_date aged >30 days). This test now uses
  datetime.date.today() and asserts recency + not-future.
  global.yml run_date must be refreshed to the run date when
  committing (currently 2026-11-06); the 30-day window gives
  slack but is not a substitute for updating it.
  '''
  g = yaml.safe_load(
    (_root() / 'conf/base/parameters/global.yml').read_text())
  run_date = datetime.date.fromisoformat(str(g['global_params']['run_date']))
  today = datetime.date.today()
  assert (today - run_date).days <= 30
  assert run_date <= today
