#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Column lineage: record then get (FIX-STUB-2, full impl).

TDD RED->GREEN: record_column_lineage / get_column_lineage
were DEFER stubs raising NotImplementedError pending
D-LINEAGE-COL-1. These tests fail with NotImplementedError
until lineage.py GREEN lands.

Spec (documented here and in lineage.py):

- ``record_column_lineage(record)`` is a pure function:
  it validates and returns a canonical record dict with
  exactly the keys ``source_dataset``, ``source_column``,
  ``target_dataset``, ``target_column``, ``transform``,
  ``run_id``, ``config_hash``. It does not persist; the
  caller appends the returned dict to its own store
  (list or dict), like ``build_lineage``. Legacy keyword
  form is also accepted: ``source_column``,
  ``target_column``, ``transformation`` (alias for
  ``transform``) and ``run_id``; omitted datasets and
  ``config_hash`` default to ``None``.
- ``get_column_lineage(dataset, column, store)`` is a
  pure query over a caller-supplied store (a list of
  records or a dict whose values are records or lists
  of records). It returns a list of deep-copied records
  in store order whose ``target_dataset`` equals
  ``dataset`` (``None`` matches any dataset) and whose
  ``target_column`` equals ``column`` (``None`` or the
  legacy single-argument form matches any dataset).
  Unknown dataset/column, an empty store, or no store
  returns ``[]`` (empty list).
'''

import pytest


def _record():
  '''Return a fresh canonical column lineage record.'''
  return {
    'source_dataset': 'raw_data',
    'source_column': 'price',
    'target_dataset': 'features',
    'target_column': 'price_lag1',
    'transform': 'lag(1)',
    'run_id': 'run-1',
    'config_hash': 'abc123',
  }


def test_record_returns_exact_canonical_mapping():
  '''Test record returns exact canonical mapping.'''
  from forecast_ml_framework.governance.lineage import (
    record_column_lineage)
  record = _record()
  result = record_column_lineage(record)
  assert result == record
  assert set(result) == {
    'source_dataset', 'source_column', 'target_dataset',
    'target_column', 'transform', 'run_id', 'config_hash',
  }


def test_record_then_get_returns_exact_mapping():
  '''Test record then get returns exact mapping.'''
  from forecast_ml_framework.governance.lineage import (
    get_column_lineage, record_column_lineage)
  store = []
  store.append(record_column_lineage(_record()))
  result = get_column_lineage('features', 'price_lag1', store)
  assert result == [_record()]


def test_get_unknown_returns_empty_list():
  '''Test get unknown returns empty list.'''
  from forecast_ml_framework.governance.lineage import (
    get_column_lineage, record_column_lineage)
  store = [record_column_lineage(_record())]
  assert get_column_lineage('features', 'missing', store) == []
  assert get_column_lineage('missing', 'price_lag1', store) == []
  assert get_column_lineage('features', 'price_lag1', []) == []
  assert get_column_lineage('features', 'price_lag1') == []


def test_get_filters_by_dataset_and_column():
  '''Test get filters by dataset and column.'''
  from forecast_ml_framework.governance.lineage import (
    get_column_lineage, record_column_lineage)
  other = _record()
  other['target_dataset'] = 'other_ds'
  other['target_column'] = 'price_lag1'
  store = [
    record_column_lineage(_record()),
    record_column_lineage(other),
  ]
  result = get_column_lineage('other_ds', 'price_lag1', store)
  assert result == [other]


def test_get_supports_dict_store():
  '''Test get supports dict store.'''
  from forecast_ml_framework.governance.lineage import (
    get_column_lineage, record_column_lineage)
  record = record_column_lineage(_record())
  store = {'features': [record]}
  assert get_column_lineage('features', 'price_lag1', store) == [
    _record()]
  single = {'only': record}
  assert get_column_lineage('features', 'price_lag1', single) == [
    _record()]


def test_get_legacy_single_column_form():
  '''Test get legacy single column form.'''
  from forecast_ml_framework.governance.lineage import (
    get_column_lineage, record_column_lineage)
  store = [record_column_lineage(_record())]
  assert get_column_lineage('price_lag1', store=store) == [_record()]


def test_record_accepts_legacy_keyword_form():
  '''Test record accepts legacy keyword form.'''
  from forecast_ml_framework.governance.lineage import (
    record_column_lineage)
  result = record_column_lineage(
    source_column='a', target_column='b',
    transformation='copy', run_id='r1')
  assert result == {
    'source_dataset': None,
    'source_column': 'a',
    'target_dataset': None,
    'target_column': 'b',
    'transform': 'copy',
    'run_id': 'r1',
    'config_hash': None,
  }


def test_record_rejects_missing_required_field():
  '''Test record rejects missing required field.'''
  from forecast_ml_framework.governance.lineage import (
    record_column_lineage)
  record = _record()
  del record['target_column']
  with pytest.raises(ValueError):
    record_column_lineage(record)
  record = _record()
  record['run_id'] = ''
  with pytest.raises(ValueError):
    record_column_lineage(record)


def test_record_rejects_unknown_key():
  '''Test record rejects unknown key.'''
  from forecast_ml_framework.governance.lineage import (
    record_column_lineage)
  record = _record()
  record['bogus'] = 'x'
  with pytest.raises(ValueError):
    record_column_lineage(record)


def test_record_rejects_non_dict_record():
  '''Test record rejects non dict record.'''
  from forecast_ml_framework.governance.lineage import (
    record_column_lineage)
  with pytest.raises(TypeError):
    record_column_lineage('not-a-dict')


def test_get_rejects_bad_store_type():
  '''Test get rejects bad store type.'''
  from forecast_ml_framework.governance.lineage import (
    get_column_lineage)
  with pytest.raises(TypeError):
    get_column_lineage('features', 'price_lag1', 'not-a-store')


def test_record_does_not_mutate_input():
  '''Test record does not mutate input.'''
  from forecast_ml_framework.governance.lineage import (
    record_column_lineage)
  record = _record()
  result = record_column_lineage(record)
  result['target_column'] = 'changed'
  assert record == _record()


def test_record_accepts_transformation_alias_in_dict():
  '''Test record accepts transformation alias in dict.'''
  from forecast_ml_framework.governance.lineage import (
    record_column_lineage)
  record = _record()
  record['transformation'] = record.pop('transform')
  assert record_column_lineage(record) == _record()


def test_record_rejects_non_string_optional_field():
  '''Test record rejects non string optional field.'''
  from forecast_ml_framework.governance.lineage import (
    record_column_lineage)
  record = _record()
  record['config_hash'] = 123
  with pytest.raises(ValueError):
    record_column_lineage(record)


def test_get_accepts_alternate_store_first_order():
  '''Test get accepts alternate store first order.'''
  from forecast_ml_framework.governance.lineage import (
    get_column_lineage, record_column_lineage)
  store = [record_column_lineage(_record())]
  assert get_column_lineage(store, 'features', 'price_lag1') == [
    _record()]


def test_get_accepts_column_and_store_positional():
  '''Test get accepts column and store positional.'''
  from forecast_ml_framework.governance.lineage import (
    get_column_lineage, record_column_lineage)
  store = [record_column_lineage(_record())]
  assert get_column_lineage('price_lag1', store) == [_record()]


def test_get_returns_copies_not_store_references():
  '''Test get returns copies not store references.'''
  from forecast_ml_framework.governance.lineage import (
    get_column_lineage, record_column_lineage)
  store = [record_column_lineage(_record())]
  result = get_column_lineage('features', 'price_lag1', store)
  result[0]['target_column'] = 'changed'
  assert store == [_record()]
