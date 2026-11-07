#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Direct tests for core prediction materialization and checks.

These tests import directly from
``forecast_ml_framework.core.predictions`` (not through
``serving.app`` or ``scoring.nodes`` aliases) so alias drift
would fail loudly instead of passing silently.
'''

from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from forecast_ml_framework.core.predictions import (
  check_predictions_finite,
  materialize_predictions,
)


def test_materialize_list_passthrough_identity():
  '''List passes through by identity.

  Fails if materialize copies the list instead of returning it.
  '''
  raw = [1.0, 2.0]
  assert materialize_predictions(raw) is raw


def test_materialize_tuple_passthrough_identity():
  '''Tuple passes through by identity.

  Fails if materialize converts tuples to lists.
  '''
  raw = (1.0, 2.0)
  assert materialize_predictions(raw) is raw


def test_materialize_dict_passthrough_identity():
  '''Dict passes through by identity.

  Fails if materialize converts dicts into lists.
  '''
  raw = {'a': 1.0}
  assert materialize_predictions(raw) is raw


def test_materialize_str_passthrough_identity():
  '''Str passes through by identity.

  Fails if materialize iterates the string into a list.
  '''
  raw = 'label'
  assert materialize_predictions(raw) is raw


def test_materialize_bytes_passthrough_identity():
  '''Bytes pass through by identity.

  Fails if materialize iterates bytes into ints.
  '''
  raw = b'label'
  assert materialize_predictions(raw) is raw


def test_materialize_none_passthrough():
  '''None passes through as None.

  Fails if materialize turns None into an empty list.
  '''
  assert materialize_predictions(None) is None


def test_materialize_ndarray_passthrough_identity():
  '''Ndarray passes through by identity.

  Fails if materialize copies the array.
  '''
  raw = np.array([1.0, 2.0])
  assert materialize_predictions(raw) is raw


def test_materialize_generator_single_consumption():
  '''Generator materializes once and is exhausted.

  Fails if materialize is lazy: a second list() of the
  generator would lose the values after materialize.
  '''
  raw = (value for value in [1.0, 2.0, 3.0])
  materialized = materialize_predictions(raw)
  assert materialized == [1.0, 2.0, 3.0]
  assert isinstance(materialized, list)
  assert not list(raw)
  check_predictions_finite(materialized)


def test_check_finite_int_float_pass():
  '''Finite int and float leaves pass.

  Fails if the numeric branch rejects plain numbers.
  '''
  check_predictions_finite(1)
  check_predictions_finite(1.5)
  check_predictions_finite([1, 2.5, -3])


def test_check_finite_decimal_pass():
  '''Finite Decimal leaves pass.

  Fails if Decimal leaves are treated as unknown types.
  '''
  check_predictions_finite(Decimal('2.5'))
  check_predictions_finite([Decimal('1.0'), Decimal('-2.5')])


def test_check_finite_numpy_scalars_pass():
  '''Numpy numeric scalars pass.

  Fails if numpy floating/integer leaves are rejected.
  '''
  check_predictions_finite(np.float64(1.5))
  check_predictions_finite(np.int64(3))
  check_predictions_finite([np.float32(0.5), np.int32(2)])


def test_check_bool_and_str_labels_pass():
  '''Bool and str label leaves pass.

  Fails if bool/str leaves are treated as unknown types.
  '''
  check_predictions_finite([True, False])
  check_predictions_finite(['yes', 'no'])
  check_predictions_finite({'a': [True, 'yes']})


def test_check_nested_traversal_pass():
  '''Nested dict/list/tuple leaves are traversed.

  Fails if traversal stops after the first level.
  '''
  check_predictions_finite({'a': [1.0, (2.0, 3.0)], 'b': 4.0})
  check_predictions_finite(np.array([[1.0, 2.0], [3.0, 4.0]]))


def test_check_masked_clean_unwraps():
  '''Clean MaskedArray unwraps and passes.

  Fails if clean masked arrays are treated as masked.
  '''
  clean = np.ma.masked_array([1.0, 2.0], mask=[0, 0])
  check_predictions_finite(clean)
  check_predictions_finite([clean, 3.0])


@pytest.mark.parametrize('value', [
  float('inf'),
  float('-inf'),
  float('nan'),
])
def test_check_nonfinite_float_raises(value):
  '''Non-finite float leaves raise ValueError.

  Fails if inf/-inf/nan slip through the finite check.
  '''
  with pytest.raises(ValueError, match='finite'):
    check_predictions_finite(value)


def test_check_masked_constant_raises():
  '''Masked constant raises ValueError.

  Fails if masked constants count as finite numbers.
  '''
  with pytest.raises(ValueError, match='finite'):
    check_predictions_finite(np.ma.masked)


def test_check_masked_element_raises():
  '''MaskedArray with one masked element raises.

  Fails if partially masked arrays are unwrapped blindly.
  '''
  masked = np.ma.masked_array([1.0, 2.0], mask=[0, 1])
  with pytest.raises(ValueError, match='finite'):
    check_predictions_finite(masked)
  with pytest.raises(ValueError, match='finite'):
    check_predictions_finite([1.0, masked])


def test_check_decimal_overflow_raises():
  '''Decimal overflowing float raises ValueError.

  Fails if Decimal leaves bypass the finite check.
  '''
  with pytest.raises(ValueError, match='finite'):
    check_predictions_finite(Decimal('1e9999'))


@pytest.mark.parametrize('value', [
  complex(float('nan'), 0.0),
  complex(0.0, float('inf')),
])
def test_check_nonfinite_complex_raises(value):
  '''Non-finite complex leaves raise ValueError.

  Fails if complex parts are not checked for finiteness.
  '''
  with pytest.raises(ValueError, match='finite'):
    check_predictions_finite(value)


@pytest.mark.parametrize('value', [
  pd.NA,
  pd.NaT,
  None,
  object(),
  b'x',
])
def test_check_bad_leaf_raises(value):
  '''Unknown leaf types raise ValueError.

  Fails if None/pd.NA/pd.NaT/object/bytes count as labels.
  '''
  with pytest.raises(ValueError, match='finite'):
    check_predictions_finite(value)


def test_check_nested_nonfinite_raises():
  '''Non-finite leaf deep in nesting raises.

  Fails if nested traversal misses the bad leaf.
  '''
  with pytest.raises(ValueError, match='finite'):
    check_predictions_finite({'a': [1.0, float('inf')]})
  with pytest.raises(ValueError, match='finite'):
    check_predictions_finite([{'a': (float('nan'),)}])


def test_check_tolist_raising_stub_raises():
  '''Leaf whose tolist() raises becomes ValueError.

  Fails if the tolist guard is removed and the raw error leaks.
  '''
  class Raiser:
    '''Leaf whose tolist always raises.'''

    def tolist(self):
      '''Raise like a broken pandas-like leaf.'''
      raise RuntimeError('boom')

  with pytest.raises(ValueError, match='finite'):
    check_predictions_finite(Raiser())


def test_check_tolist_list_and_tuple_pass():
  '''Leaves with tolist() convert and traverse.

  Fails if the tolist list/tuple branches are removed.
  '''
  class Listy:
    '''Leaf converting to a list via tolist.'''

    def tolist(self):
      '''Return a finite list.'''
      return [1.0, 2.0]

  class Tuply:
    '''Leaf converting to a tuple via tolist.'''

    def tolist(self):
      '''Return a finite tuple.'''
      return (1.0, 2.0)

  class BadListy:
    '''Leaf converting to a non-finite list.'''

    def tolist(self):
      '''Return a list holding nan.'''
      return [float('nan')]

  check_predictions_finite(Listy())
  check_predictions_finite(Tuply())
  with pytest.raises(ValueError, match='finite'):
    check_predictions_finite(BadListy())


def test_check_generator_consumes_without_materialize():
  '''Generator check consumes the one-shot iterator.

  Fails if traversal iterates generators lazily or leaves
  them unconsumed: the caller must materialize first when
  the values are needed afterwards (single-consumption).
  '''
  raw = (value for value in [1.0, 2.0])
  check_predictions_finite(raw)
  assert not list(raw)
