#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Tests for the BalanceSheet money primitive.'''

from world_of_supply.economy import BalanceSheet


def test_total_is_profit_plus_loss():
  '''Verify total sums profit and loss components.

  Returns:
    None: Test completes without assertion failure.
  '''
  sheet = BalanceSheet(profit=10, loss=-3)
  assert sheet.total() == 7


def test_addition_is_component_wise():
  '''Verify addition combines profit and loss separately.

  Returns:
    None: Test completes without assertion failure.
  '''
  combined = BalanceSheet(5, -2) + BalanceSheet(1, -4)
  assert combined.profit == 6
  assert combined.loss == -6


def test_subtraction_negates_components():
  '''Verify subtraction negates each component separately.

  Returns:
    None: Test completes without assertion failure.
  '''
  difference = BalanceSheet(5, -2) - BalanceSheet(1, -4)
  assert difference.profit == 4
  assert difference.loss == 2


def test_sum_over_collection():
  '''Verify built-in sum over balance sheets uses zero start.

  Returns:
    None: Test completes without assertion failure.
  '''
  sheets = [BalanceSheet(1, 0), BalanceSheet(2, -1), BalanceSheet()]
  total = sum(sheets)
  assert total.total() == 2


def test_repr_shows_net_first():
  '''Verify repr starts with net total for loss-only sheets.

  Returns:
    None: Test completes without assertion failure.
  '''
  text = repr(BalanceSheet(0, -5))
  assert text == '-5 (0 -5)'
