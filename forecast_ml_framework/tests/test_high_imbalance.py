#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Tests for class-imbalance handling.'''

from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from forecast_ml_framework.pipelines.modeling.imbalance import (
  compute_imbalance,
)


def test_binary_imbalance_values():
  '''Binary imbalance gives rate, scale weight and class weights.'''
  y = np.array([0] * 90 + [1] * 10)
  out = compute_imbalance(y, 'classification')
  assert set(out) == {'real_rate', 'scale_pos_weight', 'class_weights'}
  assert out['real_rate'] == pytest.approx(0.1)
  assert out['scale_pos_weight'] == pytest.approx(9.0)
  weights = out['class_weights']
  assert weights[1] > weights[0]
  assert weights[0] == pytest.approx(100 / (2 * 90))
  assert weights[1] == pytest.approx(100 / (2 * 10))


def test_balanced_binary_is_unit_weight():
  '''Balanced binary data gives scale weight 1.'''
  y = np.array([0] * 50 + [1] * 50)
  out = compute_imbalance(y, 'classification')
  assert out['real_rate'] == pytest.approx(0.5)
  assert out['scale_pos_weight'] == pytest.approx(1.0)
  assert out['class_weights'][0] == pytest.approx(1.0)
  assert out['class_weights'][1] == pytest.approx(1.0)


def test_series_input_and_multiclass():
  '''Series input works; multiclass weights favour minorities.'''
  y = pd.Series([0] * 60 + [1] * 30 + [2] * 10)
  out = compute_imbalance(y, 'classification')
  assert out['class_weights'][2] > out['class_weights'][0]
  assert out['real_rate'] == pytest.approx(0.1)


def test_regression_imbalance_neutral():
  '''Regression returns neutral imbalance values.'''
  y = np.array([1.0, 2.0, 3.0, 4.0])
  out = compute_imbalance(y, 'regression')
  assert out['scale_pos_weight'] == pytest.approx(1.0)
  assert out['class_weights'] == {}


def test_empty_y_raises():
  '''Empty target raises ValueError.'''
  with pytest.raises(ValueError):
    compute_imbalance(np.array([]), 'classification')


def test_none_y_raises():
  '''y=None raises ValueError.'''
  with pytest.raises(ValueError):
    compute_imbalance(None, 'classification')


@pytest.mark.parametrize('bad_y', [
    np.array([0.0, 1.0, np.nan]),
    np.array([np.nan, 1.0]),
    pd.Series([0, 1, np.nan]),
])
def test_nan_y_raises(bad_y):
  '''y containing NaN raises ValueError.'''
  with pytest.raises(ValueError, match='NaN'):
    compute_imbalance(bad_y, 'classification')


def test_single_class_all_positive_real_rate_one():
  '''Single-class all-1 target has real_rate 1.0 (positive rate).'''
  out = compute_imbalance(np.ones(10, dtype=int), 'classification')
  assert out['real_rate'] == pytest.approx(1.0)
  assert out['scale_pos_weight'] == pytest.approx(1.0)
  assert out['class_weights'] == {1: pytest.approx(1.0)}


def test_single_class_all_negative_real_rate_zero():
  '''Single-class all-0 target has real_rate 0.0 (positive rate).'''
  out = compute_imbalance(np.zeros(10, dtype=int), 'classification')
  assert out['real_rate'] == pytest.approx(0.0)
  assert out['scale_pos_weight'] == pytest.approx(1.0)
  assert out['class_weights'] == {0: pytest.approx(1.0)}


def test_multiclass_minority_rate():
  '''Multiclass real_rate is the minority-class rate.'''
  y = np.array([0] * 50 + [1] * 30 + [2] * 20)
  out = compute_imbalance(y, 'classification')
  assert out['real_rate'] == pytest.approx(0.2)
  assert out['scale_pos_weight'] == pytest.approx(1.0)


@pytest.mark.parametrize('bad_y', [
  np.array([0.0, 1.0, np.inf]),
  np.array([0.0, 1.0, -np.inf]),
  np.array([np.inf]),
  pd.Series([0, 1, np.inf]),
  np.array([0, 1, np.inf], dtype=object),
])
def test_inf_y_raises(bad_y):
  '''y containing inf/-inf raises ValueError.'''
  with pytest.raises(ValueError, match='non-finite|inf|finite'):
    compute_imbalance(bad_y, 'classification')


@pytest.mark.parametrize('bad_y', [
  pd.Series([pd.Timestamp('2020-01-01'), pd.NaT,
             pd.Timestamp('2020-01-03')]),
  np.array([pd.Timestamp('2020-01-01'), pd.NaT], dtype=object),
  np.array([1, 0, None], dtype=object),
])
def test_nat_none_y_raises(bad_y):
  '''y containing NaT/None raises ValueError.'''
  with pytest.raises(ValueError, match='NaT|NaN|None|non-finite|finite'):
    compute_imbalance(bad_y, 'classification')


@pytest.mark.parametrize('bad_y', [
  np.array([0, 1, Decimal('Infinity')], dtype=object),
  np.array([0, 1, Decimal('-Infinity')], dtype=object),
  np.array([0, 1, Decimal('NaN')], dtype=object),
  [0, 1, Decimal('Infinity')],
  pd.Series([0, 1, Decimal('Infinity')], dtype=object),
])
def test_decimal_non_finite_y_raises(bad_y):
  '''y containing non-finite Decimal raises ValueError.'''
  with pytest.raises(ValueError, match='non-finite|inf|finite'):
    compute_imbalance(bad_y, 'classification')


def test_list_of_tuples_labels_preserved():
  '''List-of-tuples y keeps tuple labels, not flattened elements.'''
  y = [(0, 1)] * 3 + [(1, 0)]
  out = compute_imbalance(y, 'classification')
  assert set(out['class_weights']) == {(0, 1), (1, 0)}
  assert out['class_weights'][(0, 1)] == pytest.approx(4 / (2 * 3))
  assert out['class_weights'][(1, 0)] == pytest.approx(4 / (2 * 1))
  assert out['real_rate'] == pytest.approx(0.25)
  assert out['scale_pos_weight'] == pytest.approx(3.0)
  series_out = compute_imbalance(pd.Series(y), 'classification')
  assert out == series_out


def test_tuple_of_tuples_labels_preserved():
  '''Tuple-of-tuples y keeps tuple labels.'''
  y = ((0, 1), (0, 1), (1, 0))
  out = compute_imbalance(y, 'classification')
  assert set(out['class_weights']) == {(0, 1), (1, 0)}


@pytest.mark.parametrize('bad_y', [
  [0, 1, pd.NA],
  np.array([0, 1, pd.NA], dtype=object),
  pd.Series([0, 1, pd.NA]),
  pd.Series([pd.NA, 1]),
])
def test_pd_na_y_raises_value_error(bad_y):
  '''y containing pd.NA raises ValueError, not TypeError.'''
  with pytest.raises(ValueError, match='NaN|NaT|NA|None|non-finite|finite'):
    compute_imbalance(bad_y, 'classification')


# --- loop6 imbalance (RED) ---

@pytest.mark.parametrize('bad_y', [
  [[0, 1], [1, 0], [0, 1]],
  [[0], [1], [0]],
  np.array([[0, 1], [1, 0]]),
  np.array([[0, 1], [1, 0]], dtype=object),
])
def test_2d_list_or_array_labels_raise(bad_y):
  '''2-D list/array labels are rejected, not flattened.'''
  with pytest.raises(ValueError, match='2-D|2D|two-dimensional|list'):
    compute_imbalance(bad_y, 'classification')


@pytest.mark.parametrize('bad_y', [
  [(0, float('nan')), (1, 0)],
  [(0, float('inf')), (1, 0)],
  [(0, None), (1, 0)],
  [(0, pd.NA), (1, 0)],
  [(0, Decimal('Infinity')), (1, 0)],
  [(0, Decimal('NaN')), (1, 0)],
  [((float('nan'),), 0), ((1,), 0)],
])
def test_tuple_label_inner_nonfinite_or_none_raises(bad_y):
  '''NaN/Inf/None inside tuple labels are rejected.'''
  with pytest.raises(ValueError, match='NaN|NaT|NA|None|non-finite|finite'):
    compute_imbalance(bad_y, 'classification')


# --- loop7 imbalance (RED) ---

@pytest.mark.parametrize('bad_y', [
  [([0, 1], 0), (1, 0)],
  [(np.array([0, 1]), 0), (1, 0)],
  [({'a': 1}, 0), (1, 0)],
  [(0, [float('nan')]), (1, 0)],
  [(0, {'x': float('inf')}), (1, 0)],
])
def test_tuple_label_container_element_raises(bad_y):
  '''List/ndarray/dict inside tuple labels raise ValueError.'''
  with pytest.raises(ValueError, match='scalar|non-finite|finite|NaN|None'):
    compute_imbalance(bad_y, 'classification')


@pytest.mark.parametrize('bad_y', [
  [(0, complex(float('inf'), 0.0)), (1, 0)],
  [(0, complex(0.0, float('-inf'))), (1, 0)],
  [(0, complex(float('nan'), 0.0)), (1, 0)],
  [(0, np.complex128(complex(float('inf'), 0.0))), (1, 0)],
])
def test_tuple_label_complex_nonfinite_raises(bad_y):
  '''Complex inf/NaN inside tuple labels raise ValueError.'''
  with pytest.raises(ValueError, match='non-finite|finite|NaN|inf'):
    compute_imbalance(bad_y, 'classification')


@pytest.mark.parametrize('bad_y', [
  '0101',
  b'0101',
  (x for x in [0, 1, 1, 0]),
  {0, 1, 1, 0},
  {0: 1, 1: 1},
  range(3),
])
def test_non_list_iterable_y_raises(bad_y):
  '''Str/generator/other iterables as y raise ValueError.'''
  with pytest.raises(ValueError, match='list|tuple|ndarray|Series|array'):
    compute_imbalance(bad_y, 'classification')


# --- loop9 tuple-inner set/frozenset/bytearray (RED) ---

@pytest.mark.parametrize('bad_y', [
  [(frozenset({0}), 1), (1, 0)],
  [(0, frozenset({1})), (1, 0)],
  [({0, 1}, 0), (1, 0)],
  [(0, {1, 2}), (1, 0)],
  [(bytearray(b'\x00'), 0), (1, 0)],
  [(0, bytearray(b'\x01')), (1, 0)],
  [(0, frozenset({float('nan')})), (1, 0)],
])
def test_tuple_label_set_frozenset_bytearray_raises(bad_y):
  '''Set/frozenset/bytearray inside tuple labels raise ValueError.'''
  with pytest.raises(ValueError, match='scalar|non-finite|finite|NaN|None'):
    compute_imbalance(bad_y, 'classification')
