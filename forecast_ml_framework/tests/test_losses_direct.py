#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Direct tests for the loss allowlist.

Pins the sorted allowlist, the error-message contract, and
non-string handling so an allowlist drift fails loudly.
'''

import pytest

from forecast_ml_framework.losses import (
  allowed_loss_names,
  get_loss_callable,
  huber_loss,
)


def test_allowed_loss_names_sorted():
  '''Allowlist is returned sorted.

  Fails if allowed_loss_names returns registry order.
  '''
  names = allowed_loss_names()
  assert names == sorted(names)
  assert len(names) == len(set(names))


def test_allowed_loss_names_contains_sklearn_and_custom():
  '''Allowlist holds sklearn losses and custom losses.

  Fails if the registry drops the sklearn or custom entries.
  '''
  names = allowed_loss_names()
  assert 'sklearn.metrics.mean_absolute_error' in names
  assert 'sklearn.metrics.mean_squared_error' in names
  assert 'forecast_ml_framework.losses.huber_loss' in names
  assert 'forecast_ml_framework.losses.quantile_loss' in names
  assert 'forecast_ml_framework.losses.pinball_loss' in names


def test_get_loss_callable_returns_registered():
  '''Known name returns the registered callable.

  Fails if get_loss_callable returns anything else.
  '''
  assert get_loss_callable(
    'forecast_ml_framework.losses.huber_loss') is huber_loss


def test_get_loss_callable_unknown_raises_with_valid_list():
  '''Unknown name raises listing the valid names.

  Fails if the error omits the allowlist or the reason.
  '''
  with pytest.raises(
      ValueError, match='not allowlisted'):
    get_loss_callable('unknown.loss')
  with pytest.raises(
      ValueError, match='Valid loss callables'):
    get_loss_callable('unknown.loss')


@pytest.mark.parametrize('value', [None, 42])
def test_get_loss_callable_nonstring_raises_valueerror(value):
  '''Hashable non-string names raise ValueError.

  Fails if non-string input leaks through the allowlist.
  '''
  with pytest.raises(ValueError, match='not allowlisted'):
    get_loss_callable(value)


def test_get_loss_callable_unhashable_raises_typeerror():
  '''Unhashable names raise TypeError.

  Fails if unhashable input were coerced to a string
  before the dict lookup instead of raising TypeError.
  '''
  with pytest.raises(TypeError):
    get_loss_callable([])
