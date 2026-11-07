#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Direct tests for core params helpers.

Covers the valid-dict passthrough, the key unwrap, nested
merging, and non-dict conversions of ``core.params`` so a
removed branch turns a test red.
'''

from datetime import date, datetime, timezone

import pytest

from forecast_ml_framework.core.params import (
  flatten_params,
  get_first,
  parse_datetime,
  parse_float,
  parse_int,
  require_dict,
)


def test_require_dict_valid_returns_same():
  '''Valid dict returns the identical object.

  Fails if require_dict copies the dict instead of returning
  the same object it was given.
  '''
  value = {'a': 1}
  assert require_dict(value, 'params') is value


def test_require_dict_none_returns_empty():
  '''None returns an empty dict.

  Fails if require_dict raises on None instead of treating
  it as empty params.
  '''
  assert require_dict(None, 'params') == {}


def test_require_dict_invalid_raises_with_label():
  '''Non-dict raises ValueError naming the label.

  Fails if the error message omits the label or value.
  '''
  with pytest.raises(ValueError, match='params must be a dict'):
    require_dict([1, 2], 'params')


def test_flatten_params_key_none_no_unwrap():
  '''Key None means no unwrapping.

  Fails if flatten unwraps a nested key when key is None.
  '''
  params = {'eda_params': {'a': 1}, 'b': 2}
  assert flatten_params(params, None) == {
    'eda_params': {'a': 1}, 'b': 2}


def test_flatten_params_list_of_pairs():
  '''List of pairs converts through dict().

  Fails if flatten_params only accepts literal dicts and
  rejects dict-convertible sequences.
  '''
  assert flatten_params([('a', 1), ('b', 2)], 'k') == {
    'a': 1, 'b': 2}


def test_flatten_params_nested_nondict_ignored():
  '''Nested non-dict under key is ignored, not merged.

  Fails if flatten crashes or merges non-dict nested values.
  '''
  params = {'eda_params': 'not-a-dict', 'x': 1}
  assert flatten_params(params, 'eda_params') == {
    'eda_params': 'not-a-dict', 'x': 1}


def test_flatten_params_nested_dict_merges_top_wins():
  '''Nested dict merges with top-level keys winning.

  Fails if the nested merge overwrites top-level keys.
  '''
  params = {'eda_params': {'a': 1, 'b': 1}, 'b': 2}
  assert flatten_params(params, 'eda_params') == {'a': 1, 'b': 2}


def test_flatten_params_empty():
  '''Empty params yield an empty dict.

  Fails if empty input raises instead of returning {}.
  '''
  assert flatten_params({}, 'k') == {}


def test_flatten_params_none():
  '''None params yield an empty dict.

  Fails if None raises instead of returning {}.
  '''
  assert flatten_params(None, 'k') == {}


def test_flatten_params_nested_empty_dict():
  '''Nested empty dict merges to the top level only.

  Fails if an empty nested dict drops the top-level keys.
  '''
  params = {'eda_params': {}, 'b': 2}
  assert flatten_params(params, 'eda_params') == {'b': 2}


def test_flatten_params_invalid_raises():
  '''Non-convertible params raise ValueError.

  Fails if an unconvertible value leaks a raw TypeError.
  '''
  with pytest.raises(ValueError, match='params must be a dict'):
    flatten_params('not-a-mapping', 'k')


def test_parse_datetime_datetime_instance_kept_aware():
  '''Naive datetime is assumed UTC; aware datetime is kept.

  Fails if the datetime-instance branch is removed or naive
  datetimes stop being normalised to UTC.
  '''
  naive = datetime(2026, 1, 1, 12, 0)
  parsed = parse_datetime(naive)
  assert parsed == datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
  aware = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
  assert parse_datetime(aware) == aware


def test_parse_datetime_date_instance():
  '''A date instance becomes midnight UTC.

  Fails if the date-instance branch is removed and dates raise
  instead of parsing to a UTC midnight datetime.
  '''
  assert parse_datetime(date(2026, 1, 2)) == datetime(
    2026, 1, 2, tzinfo=timezone.utc)


def test_parse_datetime_naive_string_assumed_utc():
  '''A naive ISO string is assumed UTC.

  Fails if naive strings stop being normalised to aware UTC.
  '''
  assert parse_datetime('2026-01-01') == datetime(
    2026, 1, 1, tzinfo=timezone.utc)


def test_parse_datetime_nonstring_raises():
  '''Non-string values raise ValueError.

  Fails if non-string values are coerced instead of raising.
  '''
  for bad in (123, None, ['2026-01-01']):
    with pytest.raises(ValueError):
      parse_datetime(bad)
  with pytest.raises(ValueError):
    parse_datetime('not-a-date')


def test_get_first_none_and_falsy():
  '''None mapping yields None; falsy values are not absent.

  Fails if get_first treats a falsy stored value such as 0 as
  missing, or raises on a None mapping.
  '''
  assert get_first(None, 'a') is None
  assert get_first({}, 'a') is None
  assert get_first({'b': 1}, 'a') is None
  assert get_first({'a': 0}, 'a') == 0
  assert get_first({'a': 0, 'b': 1}, 'a', 'b') == 0


def test_parse_int_float_truncation():
  '''Float input truncates through int; strings parse.

  Fails if parse_int stops truncating floats toward zero.
  '''
  assert parse_int(3.9, 0) == 3
  assert parse_int('7', 0) == 7
  assert parse_int('bad', 5) == 5
  assert parse_float('2.5', 0.0) == 2.5
  assert parse_float('bad', 1.5) == 1.5


def test_pipeline_nodes_use_canonical_params_helpers():
  '''Pipeline/governance nodes do not redefine param helpers.

  Fails if a module redefines a canonical helper instead of
  importing it from core.params: verbatim copies of parse_int,
  parse_float, get_first, or parse_datetime reintroduce drift.
  '''
  from pathlib import Path
  root = Path(__file__).resolve().parents[1]
  src = root / 'src' / 'forecast_ml_framework'
  eda = (src / 'pipelines' / 'eda' / 'nodes.py').read_text()
  interp = (src / 'pipelines' / 'interpretability'
            / 'nodes.py').read_text()
  retraining = (src / 'pipelines' / 'retraining'
                / 'nodes.py').read_text()
  lineage = (src / 'governance' / 'lineage.py').read_text()
  retention = (src / 'governance' / 'retention.py').read_text()
  assert 'def _float(' not in eda
  assert 'def _int(' not in eda
  assert 'def _int(' not in interp
  assert 'def _get(' not in retraining
  assert 'def _parse_datetime(' not in retraining
  assert 'from forecast_ml_framework.core.params import' in eda
  assert 'from forecast_ml_framework.core.params import' in interp
  assert 'from forecast_ml_framework.core.params import' in retraining
  assert 'from forecast_ml_framework.core.params import' in lineage
  assert 'from forecast_ml_framework.core.params import' in retention
