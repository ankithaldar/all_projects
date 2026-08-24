#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Tests for the BalanceSheet money primitive.'''

from world_of_supply.economy import BalanceSheet


def test_total_is_profit_plus_loss():
  sheet = BalanceSheet(profit=10, loss=-3)
  assert sheet.total() == 7


def test_addition_is_component_wise():
  combined = BalanceSheet(5, -2) + BalanceSheet(1, -4)
  assert combined.profit == 6
  assert combined.loss == -6


def test_subtraction_negates_components():
  difference = BalanceSheet(5, -2) - BalanceSheet(1, -4)
  assert difference.profit == 4
  assert difference.loss == 2


def test_sum_over_collection():
  total = sum([BalanceSheet(1, 0), BalanceSheet(2, -1), BalanceSheet()])
  assert total.total() == 2


def test_repr_shows_net_first():
  text = repr(BalanceSheet(0, -5))
  assert text == '-5 (0 -5)'
