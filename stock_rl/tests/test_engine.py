#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Tests for the backtest engine.

The look-ahead tests are the point of this file. A backtest that leaks the
future is worse than no backtest, because it produces a confident number
that is wrong.
'''

from datetime import datetime, timedelta

import pytest

from stock_rl.bars import Bar
from stock_rl.costs import DELIVERY, CostModel
from stock_rl.engine import run_backtest

#: Zero-cost model, so arithmetic assertions are not obscured by charges.
FREE = CostModel(stt_buy=0.0, stt_sell=0.0, exchange_pct=0.0,
                 sebi_pct=0.0, stamp_duty_buy=0.0, gst_pct=0.0,
                 dp_charge=0.0, slippage_bps=0.0)


def make_bars(prices):
  '''Return one flat bar per close price.

  Each bar opens at the previous close and closes at the given price, so
  the equity curve of a buy-and-hold signal is exactly the price series
  scaled by the starting capital.

  Args:
    prices: Sequence of close prices.

  Returns:
    List of bars with one hour spacing.
  '''
  start = datetime(2026, 1, 1)
  bars = []
  previous = prices[0]
  for index, price in enumerate(prices):
    bars.append(Bar(
      timestamp=start + timedelta(hours=index),
      open=previous,
      high=max(previous, price),
      low=min(previous, price),
      close=price,
      volume=1000.0,
    ))
    previous = price
  return bars


def always(weight):
  '''Return a signal function that always returns ``weight``.

  Args:
    weight: Target weight to return.

  Returns:
    A signal callable.
  '''
  return lambda history: weight


class TestNoLookAhead:
  '''The signal must never see the bar it is filled on.'''

  def test_signal_history_excludes_the_fill_bar(self):
    bars = make_bars([10, 11, 12, 13, 14])
    seen = []

    def record(history):
      seen.append(len(history))
      return 0.0

    run_backtest(bars, record, costs=FREE)
    # Decisions are made on bars 0..n-2 and see at most index+1 bars.
    assert seen == [1, 2, 3, 4]

  def test_first_bar_is_never_traded(self):
    # No signal precedes bar 0, so no fill may occur at its open.
    bars = make_bars([10, 11, 12])
    result = run_backtest(bars, always(1.0), costs=FREE)
    assert all(fill.bar_index >= 1 for fill in result.fills)

  def test_last_signal_is_discarded_not_filled(self):
    # There is no open after the final bar, so the final signal must not
    # appear as a fill.
    bars = make_bars([10, 11, 12, 13])
    result = run_backtest(bars, always(1.0), costs=FREE)
    assert max(fill.bar_index for fill in result.fills) <= len(bars) - 1

  def test_future_prices_cannot_change_past_decisions(self):
    # Signal fires only on the first bar. Mutating every later price must
    # not alter the fill that was decided on bar 0. Each run needs its own
    # signal instance, since the closure carries call state.
    def once():
      calls = {'count': 0}

      def signal(history):
        del history
        calls['count'] += 1
        return 1.0 if calls['count'] == 1 else 0.0

      return signal

    base = make_bars([10, 11, 12, 13])
    spiked = make_bars([10, 99, 98, 97])
    original = run_backtest(base, once(), costs=FREE)
    altered = run_backtest(spiked, once(), costs=FREE)
    assert original.fills[0].price == altered.fills[0].price
    assert original.fills[0].price == pytest.approx(10.0)

  def test_rising_price_does_not_buy_more_shares(self):
    # Regression guard. Sizing an order on the current bar's close while
    # filling at that bar's open is look-ahead inside a single bar, and
    # shows up as buying extra shares every time price rises.
    bars = make_bars([100, 110, 121, 133])
    result = run_backtest(bars, always(1.0), capital=1000.0, costs=FREE)
    quantities = [fill.quantity for fill in result.fills]
    assert len(quantities) == 1, f'rebalanced mid-hold: {quantities}'


class TestEquityArithmetic:
  '''The curve must match hand-computable arithmetic.'''

  def test_buy_and_hold_tracks_price_ratio(self):
    bars = make_bars([100, 110, 121])
    result = run_backtest(bars, always(1.0), capital=1000.0, costs=FREE)
    # Bought 10 shares at the open of bar 1 (which opens at 100).
    # Equity is cash + shares * close, over starting capital.
    assert result.equity[0] == pytest.approx(1.0)
    assert result.equity[-1] == pytest.approx(1210.0 / 1000.0)

  def test_staying_flat_never_loses_money(self):
    bars = make_bars([100, 50, 25])
    result = run_backtest(bars, always(0.0), costs=FREE)
    assert all(value == pytest.approx(1.0) for value in result.equity)
    assert not result.fills

  def test_first_return_is_zero(self):
    result = run_backtest(make_bars([10, 11]), always(1.0), costs=FREE)
    assert result.returns[0] == 0.0

  def test_equity_and_returns_are_aligned(self):
    bars = make_bars([100, 110, 121, 130])
    result = run_backtest(bars, always(1.0), costs=FREE)
    assert len(result.equity) == len(bars)
    assert len(result.returns) == len(bars)
    assert result.bars_processed == len(bars)

  def test_returns_reconstruct_equity(self):
    bars = make_bars([100, 110, 121, 130, 140])
    result = run_backtest(bars, always(1.0), costs=FREE)
    rebuilt = 1.0
    for index, ret in enumerate(result.returns):
      if index:
        rebuilt *= 1.0 + ret
      assert rebuilt == pytest.approx(result.equity[index])


class TestCosts:
  '''Turnover must cost money, and must cost more when it churns.'''

  def test_costs_are_charged_on_trades(self):
    bars = make_bars([100, 100, 100, 100])
    result = run_backtest(bars, always(1.0), capital=100_000.0)
    assert result.total_cost > 0.0
    assert all(fill.cost > 0.0 for fill in result.fills)

  def test_free_model_charges_nothing(self):
    bars = make_bars([100, 100, 100, 100])
    assert run_backtest(bars, always(1.0), costs=FREE).total_cost == 0.0

  def test_churning_costs_more_than_holding(self):
    bars = make_bars([100] * 8)
    churn = run_backtest(bars, _alternating(), costs=DELIVERY)
    hold = run_backtest(bars, always(1.0), costs=DELIVERY)
    assert churn.total_cost > hold.total_cost

  def test_costs_reduce_final_equity(self):
    bars = make_bars([100, 120, 140])
    free = run_backtest(bars, always(1.0), capital=100_000.0, costs=FREE)
    charged = run_backtest(bars, always(1.0), capital=100_000.0)
    assert charged.equity[-1] < free.equity[-1]


def _alternating():
  '''Return a signal that flips between fully long and fully flat.

  Returns:
    A signal callable.
  '''
  state = {'up': True}

  def flip(history):
    del history
    state['up'] = not state['up']
    return 1.0 if state['up'] else 0.0

  return flip


class TestConstraints:
  '''Weight bounds and minimum ticket size.'''

  def test_weights_are_clamped_to_max_weight(self):
    result = run_backtest(
      make_bars([100, 101, 102, 103]), always(5.0), max_weight=0.5)
    assert max(abs(weight) for weight in result.target_weights) <= 0.5

  def test_short_signals_are_clamped_not_obeyed(self):
    # The book is long-only. A negative target must floor to flat, never
    # to a short position.
    result = run_backtest(
      make_bars([100, 101, 102, 103]), always(-1.0), max_weight=0.5)
    assert min(result.target_weights) == pytest.approx(0.0)
    assert all(fill.quantity >= 0 for fill in result.fills)
    assert not result.fills

  def test_min_trade_value_suppresses_drift_churn(self):
    bars = make_bars([100] * 6)
    without = run_backtest(bars, _alternating(), costs=FREE)
    with_gate = run_backtest(
      bars, _alternating(), costs=FREE, min_trade_value=1e9)
    assert len(without.fills) > len(with_gate.fills)

  def test_shares_are_whole_numbers(self):
    result = run_backtest(make_bars([100, 101, 102]), always(1.0))
    assert all(isinstance(fill.quantity, int) for fill in result.fills)


class TestValidation:
  '''Bad input must fail loudly.'''

  def test_requires_two_bars(self):
    with pytest.raises(ValueError, match='at least 2 bars'):
      run_backtest(make_bars([100]), always(1.0))

  def test_requires_positive_capital(self):
    with pytest.raises(ValueError, match='capital must be positive'):
      run_backtest(make_bars([100, 101]), always(1.0), capital=0.0)

  @pytest.mark.parametrize('weight', [0.0, -0.5, 1.5])
  def test_rejects_out_of_range_max_weight(self, weight):
    with pytest.raises(ValueError, match='max_weight'):
      run_backtest(make_bars([100, 101]), always(1.0), max_weight=weight)
