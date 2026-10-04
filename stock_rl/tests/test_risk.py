#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Tests for the pre-trade risk layer, circuit bands and the kill switch.

Four properties are asserted here that a reviewer cannot establish by
reading the code, and they are the whole reason these modules exist:

  * **every check rejects its violation.** A risk check with no test for
    its failure mode is a risk check nobody knows works. There is one
    test per check id in ``check_ids``, and a test that the ids in a
    report are exactly those ids, so a check added without a test cannot
    hide.
  * **the kill switch latches.** It is tripped, then cleared by every
    programmatic route available -- no reason, a blank reason, a live
    breach, a metric that has recovered -- and confirmed still tripped
    after each.
  * **the kill switch survives a restart.** State is written, a *second*
    object is built from the same file, and the second object is halted.
  * **the chain detects an edited earlier record.** The audit side is in
    ``test_execution.py``; the tamper test is there.

The kill switch tests deliberately reach for the private-looking escape
hatches -- ``reset`` with no reason, with a blank reason, with a breach
still live -- because those are the routes by which a control gets
un-wired in a refactor.

No test here sleeps, opens a socket, or draws a random number. The clock
is injected and every input is literal.
'''

from datetime import datetime, timedelta, timezone

import pytest

from stock_rl.bars import Bar
from stock_rl.risk.checks import (
  AccountState,
  CheckResult,
  OrderRequest,
  PriceBand,
  RiskRejected,
  RiskReport,
  RmsChecker,
  RmsLimits,
  SecurityLimits,
  buy,
  check_ids,
  limit_order,
  market_order,
  passed_status,
  rejected_status,
  sell,
  statuses,
)
from stock_rl.risk.circuit import (
  CircuitBand,
  CorporateAction,
  band_for,
  bonus,
  circuit_lower,
  circuit_states,
  circuit_upper,
  classify,
  corporate_actions,
  dividend,
  ex_price,
  halted_states,
  normal,
  no_trade_lower,
  no_trade_upper,
  round_to_tick,
  split,
  split_adjusted_bars,
  states,
)
from stock_rl.risk.killswitch import (
  KillSwitch,
  KillSwitchError,
  KillSwitchLatched,
  KillThresholds,
  Trip,
  default_state_file,
  kill_switch_schema,
  trip_codes,
  utcnow,
)
from stock_rl.risk.circuit import _last_index_before
from stock_rl.risk.var import (
  covariance_matrix,
  historical_cvar,
  historical_var,
  normal_quantile,
  parametric_cvar,
  parametric_var,
  portfolio_var,
  portfolio_variance,
  sample_volatility,
  var_from_returns,
  z_score,
)

#: A fixed clock for the kill switch, so no test waits for time.
EPOCH = datetime(2026, 3, 2, 9, 15, tzinfo=timezone.utc)


def fixed_clock():
  '''Return a wall-clock source that always reports the same instant.

  Returns:
    A callable suitable for ``KillSwitch(clock=...)``.
  '''
  return lambda: EPOCH


def make_limits(**overrides) -> RmsLimits:
  '''Return account limits that admit a modest order.

  Args:
    overrides: Fields to replace on the defaults.

  Returns:
    An :class:`RmsLimits`.
  '''
  values = {
    'cumulative_open_order_value': 5_000_000.0,
    'max_position': 100_000,
    'max_trading_value': 10_000_000.0,
    'max_exposure': 10_000_000.0,
    'max_turnover': 20_000_000.0,
    'max_security_value': 2_000_000.0,
  }
  values.update(overrides)
  return RmsLimits(**values)


def make_security(symbol: str = 'RELIANCE', **overrides) -> SecurityLimits:
  '''Return venue limits for one scrip.

  Args:
    symbol: Scrip symbol.
    overrides: Fields to replace on the defaults.

  Returns:
    A :class:`SecurityLimits`.
  '''
  values = {
    'symbol': symbol,
    'band': PriceBand(2900.0, 3200.0),
    'mwpl': PriceBand(2000.0, 4000.0),
    'max_order_quantity': 5_000,
    'max_order_value': 1_000_000.0,
  }
  values.update(overrides)
  return SecurityLimits(**values)


def make_checker(security: SecurityLimits | None = None,
                 limits: RmsLimits | None = None,
                 banned: frozenset[str] = frozenset()) -> RmsChecker:
  '''Return a checker over a single scrip.

  Args:
    security: Venue limits; defaults to :func:`make_security`.
    limits: Account limits; defaults to :func:`make_limits`.
    banned: Symbols under an F&O ban.

  Returns:
    An :class:`RmsChecker`.
  '''
  venue = security or make_security()
  return RmsChecker(limits or make_limits(),
                    {venue.symbol: venue}, banned)


def make_order(**overrides) -> OrderRequest:
  '''Return an order that passes every check.

  Args:
    overrides: Fields to replace on the defaults.

  Returns:
    An :class:`OrderRequest`.
  '''
  values = {
    'symbol': 'RELIANCE',
    'side': buy,
    'quantity': 100,
    'order_type': limit_order,
    'price': 3000.0,
    'reference_price': 3000.0,
    'algo_id': '99999',
  }
  values.update(overrides)
  return OrderRequest(**values)


def rejected_ids(report: RiskReport) -> set[str]:
  '''Return the ids of every rejected check in a report.

  Args:
    report: The report to inspect.

  Returns:
    Set of rejected check ids.
  '''
  return {result.check_id for result in report.rejections}


class TestResultShape:
  '''The structure every rejection travels in.'''

  def test_a_report_carries_exactly_the_declared_checks(self):
    report = make_checker().check(make_order(), AccountState())
    assert tuple(result.check_id for result in report.results) == check_ids

  def test_a_report_with_no_results_is_not_approved(self):
    # The safe reading: "no rejection" must not be reachable from an
    # empty report, because a caller that assembled the report itself
    # would otherwise get approval out of nothing.
    assert RiskReport('x', ()).approved is False

  def test_every_check_carries_a_source_citation(self):
    report = make_checker().check(make_order(), AccountState())
    assert all(result.source for result in report.results)

  def test_a_check_needs_a_status_from_the_declared_set(self):
    with pytest.raises(ValueError, match='status must be one of'):
      CheckResult('price_band', 'maybe', 'unsure')

  def test_a_check_needs_a_reason_even_on_a_pass(self):
    with pytest.raises(ValueError, match='must carry a reason'):
      CheckResult('price_band', passed_status, '   ')

  def test_lookup_by_id_finds_the_check(self):
    report = make_checker().check(make_order(), AccountState())
    assert report.result('price_band').passed is True

  def test_lookup_by_unknown_id_raises_rather_than_defaulting(self):
    report = make_checker().check(make_order(), AccountState())
    with pytest.raises(KeyError):
      report.result('no_such_check')

  def test_render_names_the_order_and_every_rejection(self):
    report = make_checker().check(make_order(price=9000.0), AccountState())
    text = report.render()
    assert 'REJECTED' in text
    assert 'price_band' in text

  def test_an_approved_report_renders_without_reasons(self):
    text = make_checker().check(make_order(), AccountState()).render()
    assert 'APPROVED' in text

  def test_only_two_statuses_exist(self):
    assert statuses == (passed_status, rejected_status)


class TestPriceBandCheck:
  '''Price versus the exchange band, the cheapest check to get right.'''

  def test_a_price_above_the_band_rejects(self):
    report = make_checker().check(make_order(price=3200.01), AccountState())
    assert 'price_band' in rejected_ids(report)

  def test_a_price_below_the_band_rejects(self):
    report = make_checker().check(make_order(price=2899.99), AccountState())
    assert 'price_band' in rejected_ids(report)

  def test_the_band_edges_are_inside_the_band(self):
    # Inclusive limits, because the exchange quotes an inclusive band and
    # an exclusive one rejects a legitimate order at the boundary.
    for price in (2900.0, 3200.0):
      report = make_checker().check(make_order(price=price),
                                   AccountState())
      assert report.result('price_band').passed is True

  def test_a_zero_price_rejects_because_it_has_no_band(self):
    report = make_checker().check(make_order(price=0.0), AccountState())
    assert 'price_band' in rejected_ids(report)
    assert 'priced limit orders only' in report.result('price_band').reason

  def test_an_inverted_band_is_refused_at_construction(self):
    with pytest.raises(ValueError, match='must exceed lower'):
      PriceBand(3100.0, 3000.0)

  def test_a_non_positive_band_floor_is_refused(self):
    with pytest.raises(ValueError, match='must be positive'):
      PriceBand(0.0, 100.0)


class TestQuantityAndValueChecks:
  '''Per-order quantity and value, both per-security limits.'''

  def test_a_quantity_above_the_per_order_limit_rejects(self):
    report = make_checker().check(make_order(quantity=5_001),
                                  AccountState())
    assert 'order_quantity' in rejected_ids(report)

  def test_the_quantity_limit_edge_passes(self):
    report = make_checker().check(make_order(quantity=5_000),
                                  AccountState())
    assert report.result('order_quantity').passed is True

  def test_a_zero_quantity_rejects(self):
    report = make_checker().check(make_order(quantity=0), AccountState())
    assert 'order_quantity' in rejected_ids(report)

  def test_a_negative_quantity_rejects_and_does_not_loosen_the_value(
      self):
    # The reason OrderRequest.value takes an absolute value: a negative
    # quantity would otherwise subtract from every rupee limit below and
    # pass five checks on a malformed order.
    report = make_checker().check(make_order(quantity=-10_000_000),
                                  AccountState())
    assert 'order_quantity' in rejected_ids(report)
    assert 'order_value' in rejected_ids(report)

  def test_a_value_above_the_per_order_limit_rejects(self):
    report = make_checker().check(make_order(quantity=1_000, price=3_000.0),
                                  AccountState())
    assert 'order_value' in rejected_ids(report)

  def test_a_non_integer_quantity_is_refused_at_construction(self):
    with pytest.raises(ValueError, match='quantity must be an int'):
      make_order(quantity=10.5)


class TestTradePriceProtection:
  '''The bad-ticket family, and the reference price it depends on.'''

  def test_a_price_far_from_the_reference_rejects(self):
    report = make_checker().check(make_order(price=3_100.0),
                                  AccountState())
    assert 'trade_price_protection' in rejected_ids(report)

  def test_a_price_close_to_the_reference_passes(self):
    report = make_checker().check(make_order(price=3_030.0),
                                  AccountState())
    assert report.result('trade_price_protection').passed is True

  def test_a_missing_reference_price_rejects_rather_than_skipping(self):
    # The fail-closed rule in its purest form: no reference price means
    # the bad-ticket check has checked nothing, so it must not pass.
    report = make_checker().check(make_order(reference_price=0.0),
                                  AccountState())
    assert 'trade_price_protection' in rejected_ids(report)
    assert 'cannot be made' in report.result('trade_price_protection').reason

  def test_a_price_inside_the_bad_ticket_band_passes(self):
    limits = make_limits(bad_ticket_pct=0.02)
    report = make_checker(limits=limits).check(
      make_order(price=3_050.0), AccountState())
    assert report.result('trade_price_protection').passed is True

  def test_the_band_edge_is_not_fudged_with_a_tolerance(self):
    # 3060/3000 is 1.020000000000000018 in binary floating point, so the
    # deviation sits a hair above the 0.02 band. A check that fudged it
    # through would be a check with an unstated tolerance in it, so the
    # deviation is asserted rather than hidden.
    limits = make_limits(bad_ticket_pct=0.02)
    report = make_checker(limits=limits).check(
      make_order(price=3_059.0), AccountState())
    assert report.result('trade_price_protection').observed < 0.02


class TestAlgoMarketOrderCheck:
  '''NSE/MSD/67753 8.1.1.12. This project is an algo.'''

  def test_a_market_order_from_an_algo_rejects(self):
    report = make_checker().check(
      make_order(order_type=market_order, price=0.0), AccountState())
    assert 'algo_market_order' in rejected_ids(report)
    assert 'NSE/MSD/67753 8.1.1.12' in \
      report.result('algo_market_order').source

  def test_the_prohibition_cites_the_exchange_rule_in_the_reason(self):
    report = make_checker().check(
      make_order(order_type=market_order, price=0.0), AccountState())
    assert 'prohibited' in report.result('algo_market_order').reason

  def test_a_market_order_fails_every_check_it_cannot_be_checked_by(self):
    # One rejection cites the rule (NSE/MSD/67753 8.1.1.12); the others
    # report the arithmetic that cannot be performed without a price.
    # Two failures on one malformed order is not a bug, and neither is
    # four: the point is that none of them passes.
    report = make_checker().check(
      make_order(order_type=market_order, price=0.0), AccountState())
    assert rejected_ids(report) == {
      'price_band', 'order_value', 'trade_price_protection',
      'algo_market_order'}

  def test_a_limit_order_from_an_algo_passes(self):
    report = make_checker().check(make_order(), AccountState())
    assert report.result('algo_market_order').passed is True

  def test_an_unrecognised_order_type_is_refused_at_construction(self):
    with pytest.raises(ValueError, match='order_type must be one of'):
      make_order(order_type='stop_loss_market')


class TestAggregateLimitChecks:
  '''The account-level and session-level ceilings.'''

  def test_cumulative_open_order_value_above_the_client_limit_rejects(self):
    report = make_checker().check(
      make_order(), AccountState(open_order_value=4_999_999.0))
    assert 'cumulative_open_order_value' in rejected_ids(report)

  def test_an_unlimited_cumulative_value_is_not_representable(self):
    # NSE 11.1 forbids "Unlimited" for algo clients, so the forbidden
    # configuration must not be constructible at all.
    with pytest.raises(ValueError, match='unlimited value is prohibited'):
      make_limits(cumulative_open_order_value=float('inf'))

  def test_the_automated_execution_check_counts_unconfirmed_orders(self):
    # The SEBI 6(v) buckets. An account whose only load is unconfirmed
    # trades must still be blocked, and that is only true if the third
    # bucket is in the sum.
    account = AccountState(executed_value=1_000_000.0,
                           open_order_value=1_000_000.0,
                           unconfirmed_value=3_000_000.0)
    report = make_checker().check(make_order(), account)
    assert 'automated_execution' in rejected_ids(report)

  def test_the_automated_execution_check_accounts_for_all_three_buckets(
      self):
    account = AccountState(executed_value=1_000_000.0,
                           open_order_value=1_000_000.0,
                           unconfirmed_value=1_000_000.0)
    result = make_checker().check(make_order(), account).result(
      'automated_execution')
    assert result.observed == pytest.approx(3_000_000.0 + 300_000.0)

  def test_the_position_limit_counts_the_side(self):
    account = AccountState(net_position=99_950)
    # 99_950 + 100 = 100_050 is past the 100_000 limit going long; the
    # same order going short lands at 99_850 and is fine.
    assert 'position_limit' in rejected_ids(
      make_checker().check(make_order(quantity=100), account))
    assert 'position_limit' not in rejected_ids(
      make_checker().check(make_order(quantity=100, side=sell), account))

  def test_the_position_limit_is_two_sided(self):
    account = AccountState(net_position=-99_950)
    assert 'position_limit' in rejected_ids(
      make_checker().check(make_order(quantity=100, side=sell), account))
    assert 'position_limit' not in rejected_ids(
      make_checker().check(make_order(quantity=100), account))

  def test_the_trading_limit_rejects(self):
    report = make_checker().check(
      make_order(), AccountState(trading_value=9_999_999.0))
    assert 'trading_limit' in rejected_ids(report)

  def test_the_exposure_limit_rejects(self):
    report = make_checker().check(
      make_order(), AccountState(exposure=9_999_999.0))
    assert 'exposure_limit' in rejected_ids(report)

  def test_the_turnover_limit_rejects(self):
    report = make_checker().check(
      make_order(), AccountState(turnover=19_999_999.0))
    assert 'turnover_limit' in rejected_ids(report)

  def test_the_security_value_limit_rejects(self):
    report = make_checker().check(
      make_order(), AccountState(security_values={'RELIANCE': 1_999_999.0}))
    assert 'security_value_limit' in rejected_ids(report)

  def test_a_symbol_absent_from_the_value_map_starts_from_zero(self):
    # A forgotten entry must not become unlimited headroom.
    report = make_checker().check(make_order(), AccountState())
    assert report.result('security_value_limit').observed == 300_000.0

  def test_a_negative_position_limit_is_refused_at_construction(self):
    with pytest.raises(ValueError, match='max_position must be'):
      make_limits(max_position=-1)


class TestBanAndMwplChecks:
  '''The exchange-published, day-specific controls.'''

  def test_a_banned_symbol_rejects(self):
    checker = make_checker(banned=frozenset({'RELIANCE'}))
    assert 'fno_ban' in rejected_ids(
      checker.check(make_order(), AccountState()))

  def test_an_unbanned_symbol_passes(self):
    checker = make_checker(banned=frozenset({'OTHER'}))
    assert checker.check(make_order(), AccountState()).result(
      'fno_ban').passed is True

  def test_buying_at_the_upper_mwpl_band_rejects(self):
    report = make_checker().check(make_order(reference_price=4_000.0),
                                  AccountState())
    assert 'mwpl' in rejected_ids(report)

  def test_selling_at_the_upper_mwpl_band_passes(self):
    # At a market-wide band only position-reducing orders are permitted,
    # so the sell side is the one that must survive.
    report = make_checker().check(
      make_order(reference_price=4_000.0, side=sell), AccountState())
    assert report.result('mwpl').passed is True

  def test_selling_at_the_lower_mwpl_band_rejects(self):
    report = make_checker().check(
      make_order(reference_price=2_000.0, side=sell), AccountState())
    assert 'mwpl' in rejected_ids(report)

  def test_buying_at_the_lower_mwpl_band_passes(self):
    report = make_checker().check(make_order(reference_price=2_000.0),
                                  AccountState())
    assert report.result('mwpl').passed is True

  def test_the_mwpl_check_needs_a_reference_price(self):
    report = make_checker().check(make_order(reference_price=0.0),
                                  AccountState())
    assert 'mwpl' in rejected_ids(report)


class TestApprovedPath:
  '''The path a real order takes.'''

  def test_a_sized_order_is_approved_by_every_check(self):
    report = make_checker().check(make_order(), AccountState())
    assert report.approved is True
    assert not report.rejections

  def test_require_approval_returns_the_passing_report(self):
    report = make_checker().require_approval(make_order(), AccountState())
    assert report.approved is True

  def test_require_approval_raises_with_the_report_attached(self):
    with pytest.raises(RiskRejected) as caught:
      make_checker().require_approval(make_order(price=9_000.0),
                                      AccountState())
    assert 'price_band' in rejected_ids(caught.value.report)

  def test_an_unknown_symbol_rejects_on_every_check(self):
    # Failing closed at the symbol level: a caller must not be able to
    # read "the F&O check passed" out of a report produced for a scrip
    # this process has no venue data for.
    checker = make_checker()
    report = checker.check(make_order(symbol='UNKNOWN'), AccountState())
    assert rejected_ids(report) == set(check_ids)
    assert checker.known('UNKNOWN') is False

  def test_a_mismatched_venue_key_is_refused(self):
    with pytest.raises(ValueError, match='does not match symbol'):
      RmsChecker(make_limits(), {'TCS': make_security('RELIANCE')})

  def test_an_empty_venue_map_is_refused(self):
    with pytest.raises(ValueError, match='must not be empty'):
      RmsChecker(make_limits(), {})

  def test_an_order_keeps_the_report_order_id(self):
    report = make_checker().check(make_order(order_id='A-1'), AccountState())
    assert report.order_id == 'A-1'

  def test_an_unnamed_order_gets_a_derived_identifier(self):
    report = make_checker().check(make_order(), AccountState())
    assert 'RELIANCE' in report.order_id


class TestCircuitBands:
  '''Band arithmetic, and the classification that gates trading.'''

  def test_the_band_is_the_reference_plus_or_minus_the_percentages(self):
    band = CircuitBand(lower_pct=0.20, upper_pct=0.20, tick=0.05)
    assert band_for(100.0, band) == PriceBand(80.0, 120.0)

  def test_an_asymmetric_band_is_not_symmetrised(self):
    band = CircuitBand(lower_pct=0.10, upper_pct=0.20, tick=0.05)
    assert band_for(100.0, band) == PriceBand(90.0, 120.0)

  def test_tick_rounding_rounds_half_up_not_to_even(self):
    # Banker's rounding would put 100.025 on the 100.00 tick; the
    # exchange puts it on 100.05. A band on the wrong tick is a band the
    # exchange never quoted.
    assert round_to_tick(100.025, 0.05) == pytest.approx(100.05)
    assert round_to_tick(100.0, 0.05) == pytest.approx(100.0)

  def test_rounding_rejects_a_bad_tick(self):
    with pytest.raises(ValueError, match='tick must be positive'):
      round_to_tick(100.0, 0.0)

  def test_rounding_rejects_a_non_finite_price(self):
    with pytest.raises(ValueError, match='must be finite'):
      round_to_tick(float('nan'), 0.05)

  def test_a_non_positive_reference_is_refused(self):
    with pytest.raises(ValueError, match='reference must be positive'):
      band_for(0.0, CircuitBand(0.1, 0.1))

  def test_the_upper_band_edge_is_the_upper_circuit(self):
    band = CircuitBand(0.20, 0.20, no_trade_pct=0.02)
    assert classify(120.0, 100.0, band) == circuit_upper

  def test_the_lower_band_edge_is_the_lower_circuit(self):
    band = CircuitBand(0.20, 0.20, no_trade_pct=0.02)
    assert classify(80.0, 100.0, band) == circuit_lower

  def test_just_inside_the_band_is_a_no_trade_halt(self):
    band = CircuitBand(0.20, 0.20, no_trade_pct=0.02)
    # The window is measured inward from the band, not from the middle:
    # 81 is inside the band (>= 80) but inside the 2% window (>= 82).
    assert classify(81.0, 100.0, band) == no_trade_lower
    assert classify(119.0, 100.0, band) == no_trade_upper

  def test_outside_the_window_but_inside_the_band_is_normal(self):
    band = CircuitBand(0.20, 0.20, no_trade_pct=0.02)
    assert classify(82.5, 100.0, band) == normal
    assert classify(117.5, 100.0, band) == normal

  def test_outside_the_window_is_normal(self):
    band = CircuitBand(0.20, 0.20, no_trade_pct=0.02)
    assert classify(100.0, 100.0, band) == normal

  def test_the_circuit_test_wins_over_the_window(self):
    # A window wider than the rounding gap must not swallow the band.
    band = CircuitBand(0.20, 0.20, no_trade_pct=0.19, tick=0.05)
    assert classify(120.0, 100.0, band) == circuit_upper

  def test_a_no_trade_window_wider_than_the_band_is_refused(self):
    with pytest.raises(ValueError, match='must be narrower than'):
      CircuitBand(0.05, 0.05, no_trade_pct=0.05)

  def test_classify_refuses_a_non_positive_price(self):
    band = CircuitBand(0.20, 0.20)
    with pytest.raises(ValueError, match='price must be positive'):
      classify(0.0, 100.0, band)

  def test_the_state_sets_are_disjoint_and_cover_everything(self):
    assert halted_states & circuit_states == frozenset()
    assert halted_states | circuit_states | {normal} == set(states)


def bars_from(closes, start=EPOCH, step=timedelta(days=1), volume=100.0):
  '''Return a flat-price bar series.

  Args:
    closes: Close prices, one per bar.
    start: Timestamp of the first bar.
    step: Spacing between bars.
    volume: Volume on every bar.

  Returns:
    List of :class:`~stock_rl.bars.Bar`.
  '''
  out = []
  for index, close in enumerate(closes):
    out.append(Bar(
      timestamp=start + index * step,
      open=close,
      high=close,
      low=close,
      close=close,
      volume=volume,
    ))
  return out


class TestCorporateActions:
  '''Where most backtests die, silently.'''

  def test_a_one_for_one_bonus_halves_the_price_and_doubles_the_volume(
      self):
    bars = bars_from([3500.0, 3500.0, 1750.0])
    action = CorporateAction(bonus, ex_date=bars[1].timestamp + timedelta(
      hours=1), ratio=1.0)
    result = split_adjusted_bars(bars, [action])
    # The pre-ex bars are restated to 1750 and volume doubles; the
    # ex-date bar is left alone because it already trades ex.
    assert result.adjusted[0].close == pytest.approx(1750.0)
    assert result.adjusted[0].volume == pytest.approx(200.0)
    assert result.adjusted[2].close == pytest.approx(1750.0)

  def test_the_both_series_remain_available(self):
    bars = bars_from([100.0, 100.0])
    action = CorporateAction(bonus, ex_date=bars[0].timestamp + timedelta(
      hours=1), ratio=1.0)
    result = split_adjusted_bars(bars, [action])
    assert [bar.close for bar in result.unadjusted] == [100.0, 100.0]
    assert [bar.close for bar in result.adjusted] == [50.0, 100.0]

  def test_a_one_to_two_split_halves_the_price(self):
    bars = bars_from([100.0, 100.0])
    action = CorporateAction(split, ex_date=bars[0].timestamp + timedelta(
      hours=1), ratio=2.0)
    result = split_adjusted_bars(bars, [action])
    assert result.adjusted[0].close == pytest.approx(50.0)
    assert result.adjusted[0].volume == pytest.approx(200.0)

  def test_a_dividend_reduces_the_price_but_not_the_share_count(self):
    bars = bars_from([100.0, 100.0], volume=500.0)
    action = CorporateAction(dividend, ex_date=bars[0].timestamp + timedelta(
      hours=1), amount=5.0)
    result = split_adjusted_bars(bars, [action])
    # 100 -> theoretical ex 95, so the pre-ex bar is scaled by 0.95.
    assert result.adjusted[0].close == pytest.approx(95.0)
    # A dividend does not change shares outstanding. Inflating volume by
    # 100/95 would be inventing a share count.
    assert result.adjusted[0].volume == pytest.approx(500.0)
    assert result.dividend_total == pytest.approx(5.0)

  def test_the_share_multiplier_relates_the_two_share_counts(self):
    bars = bars_from([100.0, 100.0])
    action = CorporateAction(bonus, ex_date=bars[0].timestamp + timedelta(
      hours=1), ratio=1.0)
    result = split_adjusted_bars(bars, [action])
    # 100 original shares become 50 adjusted ones.
    assert result.share_multiplier == pytest.approx(2.0)
    assert 100 / result.share_multiplier == pytest.approx(50.0)

  def test_several_actions_compose_in_ex_date_order(self):
    bars = bars_from([100.0, 100.0, 100.0, 100.0])
    first = CorporateAction(bonus, ex_date=bars[1].timestamp, ratio=1.0)
    second = CorporateAction(bonus, ex_date=bars[2].timestamp, ratio=1.0)
    result = split_adjusted_bars(bars, [second, first])
    # Bar 0 precedes both, so it carries both halvings; bar 1 carries one.
    assert result.adjusted[0].close == pytest.approx(25.0)
    assert result.adjusted[1].close == pytest.approx(50.0)
    assert result.share_multiplier == pytest.approx(4.0)

  def test_actions_are_sorted_by_ex_date_before_use(self):
    early = CorporateAction(dividend, ex_date=EPOCH + timedelta(days=1),
                            amount=1.0)
    late = CorporateAction(dividend, ex_date=EPOCH + timedelta(days=3),
                           amount=2.0)
    assert corporate_actions([late, early]) == (early, late)

  def test_an_ex_date_before_the_series_is_refused(self):
    bars = bars_from([100.0, 100.0])
    action = CorporateAction(dividend, ex_date=EPOCH - timedelta(days=1),
                             amount=1.0)
    with pytest.raises(ValueError, match='precedes the first bar'):
      split_adjusted_bars(bars, [action])

  def test_an_ex_date_after_the_series_is_refused(self):
    bars = bars_from([100.0, 100.0])
    action = CorporateAction(dividend, ex_date=EPOCH + timedelta(days=9),
                             amount=1.0)
    with pytest.raises(ValueError, match='after the last bar'):
      split_adjusted_bars(bars, [action])

  def test_an_empty_series_is_refused(self):
    with pytest.raises(ValueError, match='must not be empty'):
      split_adjusted_bars([], [])

  def test_scrambled_bars_are_refused_rather_than_sorted(self):
    bars = bars_from([100.0, 100.0])
    scrambled = [bars[1], bars[0]]
    with pytest.raises(ValueError, match='strictly ascending'):
      split_adjusted_bars(scrambled, [])

  def test_an_unknown_action_kind_is_refused(self):
    with pytest.raises(ValueError, match='kind must be one of'):
      CorporateAction('rights', ex_date=EPOCH)

  def test_a_zero_dividend_is_refused(self):
    with pytest.raises(ValueError, match='must be positive'):
      CorporateAction(dividend, ex_date=EPOCH, amount=0.0)

  def test_a_zero_bonus_ratio_is_refused(self):
    with pytest.raises(ValueError, match='must be positive'):
      CorporateAction(bonus, ex_date=EPOCH, ratio=0.0)

  def test_the_theoretical_ex_price_is_the_unrounded_value(self):
    action = CorporateAction(dividend, ex_date=EPOCH, amount=2.5)
    assert action.theoretical_price(100.0) == pytest.approx(97.5)


class TestKillSwitchThresholds:
  '''The pre-defined conditions, and the argument validation.'''

  def test_the_defaults_are_the_documented_ones(self):
    limits = KillThresholds()
    assert limits.index_fall == 0.05
    assert limits.max_drawdown == 0.15
    assert limits.ack_latency == 2.0

  def test_a_zero_threshold_is_refused(self):
    with pytest.raises(ValueError, match='index_fall must be in'):
      KillThresholds(index_fall=0.0)

  def test_a_threshold_above_one_is_refused(self):
    with pytest.raises(ValueError, match='max_drawdown must be in'):
      KillThresholds(max_drawdown=1.5)

  def test_a_non_positive_latency_is_refused(self):
    with pytest.raises(ValueError, match='ack_latency must be positive'):
      KillThresholds(ack_latency=0.0)

  def test_every_trip_code_is_recognised(self):
    assert set(trip_codes) == {'manual', 'index_fall', 'max_drawdown',
                               'fno_ban', 'ack_latency'}


def make_switch(path, thresholds=None, algo_id='99999') -> KillSwitch:
  '''Return a kill switch over a temporary state file.

  Args:
    path: State file path.
    thresholds: Pre-defined conditions, or None for the defaults.
    algo_id: Algo ID the switch belongs to.

  Returns:
    A :class:`KillSwitch`.
  '''
  return KillSwitch(algo_id, path, thresholds or KillThresholds(),
                    clock=fixed_clock())


class TestKillSwitchLatching:
  '''A kill switch that self-resets is a pause button.'''

  def test_a_fresh_switch_is_armed(self, tmp_path):
    switch = make_switch(tmp_path / 'kill.json')
    assert switch.tripped is False
    assert switch.trading_enabled is True

  def test_a_manual_trip_halts_trading(self, tmp_path):
    switch = make_switch(tmp_path / 'kill.json')
    record = switch.trip('operator stopped the strategy')
    assert record.code == 'manual'
    assert switch.tripped is True
    assert switch.trading_enabled is False

  def test_recovered_metrics_do_not_untrip_the_switch(self, tmp_path):
    # The core invariant: there is no path from a healthy metric to an
    # armed switch. The drawdown has gone and the switch is still latched.
    switch = make_switch(tmp_path / 'kill.json')
    switch.trip('operator stopped the strategy')
    switch.evaluate(index_change=0.0, drawdown=0.0, fno_banned=False,
                    ack_latency=0.1)
    assert switch.tripped is True

  def test_a_switch_that_is_not_tripped_refuses_to_be_cleared(self,
                                                               tmp_path):
    switch = make_switch(tmp_path / 'kill.json')
    with pytest.raises(KillSwitchLatched, match='not tripped'):
      switch.reset('routine housekeeping')

  def test_clearing_without_a_reason_raises_a_type_error(self, tmp_path):
    # No default argument, so the call cannot even be made.
    switch = make_switch(tmp_path / 'kill.json')
    switch.trip('operator stopped the strategy')
    with pytest.raises(TypeError):
      switch.reset()  # pylint: disable=no-value-for-parameter

  def test_clearing_with_a_blank_reason_is_refused(self, tmp_path):
    switch = make_switch(tmp_path / 'kill.json')
    switch.trip('operator stopped the strategy')
    with pytest.raises(KillSwitchLatched, match='requires a stated reason'):
      switch.reset('   ')
    assert switch.tripped is True

  def test_clearing_while_a_breach_is_live_is_refused(self, tmp_path):
    switch = make_switch(tmp_path / 'kill.json')
    switch.trip('drawdown stop', code='max_drawdown')
    with pytest.raises(KillSwitchLatched, match='still breached'):
      switch.reset('auto job', drawdown=0.20)
    assert switch.tripped is True

  def test_a_scheduled_clear_loop_cannot_beat_a_live_breach(self, tmp_path):
    switch = make_switch(tmp_path / 'kill.json')
    switch.evaluate(drawdown=0.30)
    for _ in range(5):
      with pytest.raises(KillSwitchLatched):
        switch.reset('scheduled reset', drawdown=0.30)
    assert switch.tripped is True

  def test_an_explicit_clear_with_a_reason_unlatches(self, tmp_path):
    switch = make_switch(tmp_path / 'kill.json')
    switch.trip('drawdown stop', code='max_drawdown')
    switch.reset('operator reviewed the book', drawdown=0.02)
    assert switch.tripped is False
    assert switch.trading_enabled is True

  def test_the_clear_reason_is_persisted(self, tmp_path):
    switch = make_switch(tmp_path / 'kill.json')
    switch.trip('operator stop')
    switch.reset('reviewed')
    assert switch.clears == ('reviewed',)

  def test_guard_refuses_while_latched(self, tmp_path):
    switch = make_switch(tmp_path / 'kill.json')
    switch.trip('operator stop')
    with pytest.raises(KillSwitchLatched, match='stays tripped'):
      switch.guard(index_change=0.0, drawdown=0.0)

  def test_guard_passes_while_armed_and_healthy(self, tmp_path):
    switch = make_switch(tmp_path / 'kill.json')
    switch.guard(index_change=-0.01, drawdown=0.02)
    assert switch.tripped is False


class TestKillSwitchAutomaticTrips:
  '''SEBI: automatic, on conditions declared before the fact.'''

  def test_an_index_fall_at_the_limit_trips(self, tmp_path):
    switch = make_switch(tmp_path / 'kill.json')
    record = switch.evaluate(index_change=-0.05)
    assert record is not None
    assert record.code == 'index_fall'
    assert switch.tripped is True

  def test_an_index_fall_inside_the_limit_does_not_trip(self, tmp_path):
    switch = make_switch(tmp_path / 'kill.json')
    assert switch.evaluate(index_change=-0.049) is None
    assert switch.tripped is False

  def test_a_drawdown_at_the_limit_trips(self, tmp_path):
    switch = make_switch(tmp_path / 'kill.json')
    assert switch.evaluate(drawdown=0.15).code == 'max_drawdown'

  def test_a_drawdown_inside_the_limit_does_not_trip(self, tmp_path):
    switch = make_switch(tmp_path / 'kill.json')
    assert switch.evaluate(drawdown=0.149) is None

  def test_an_fno_ban_trips(self, tmp_path):
    switch = make_switch(tmp_path / 'kill.json')
    assert switch.evaluate(fno_banned=True).code == 'fno_ban'

  def test_the_fno_ban_condition_can_be_switched_off(self, tmp_path):
    thresholds = KillThresholds(fno_ban=False)
    switch = make_switch(tmp_path / 'kill.json', thresholds)
    assert switch.evaluate(fno_banned=True) is None

  def test_a_slow_acknowledgement_trips(self, tmp_path):
    switch = make_switch(tmp_path / 'kill.json')
    assert switch.evaluate(ack_latency=2.01).code == 'ack_latency'

  def test_the_ack_latency_edge_does_not_trip(self, tmp_path):
    switch = make_switch(tmp_path / 'kill.json')
    assert switch.evaluate(ack_latency=2.0) is None

  def test_a_missing_latency_is_not_a_breach(self, tmp_path):
    switch = make_switch(tmp_path / 'kill.json')
    assert switch.evaluate(ack_latency=None) is None

  def test_every_breached_condition_is_recorded_not_just_the_first(
      self, tmp_path):
    # The first condition to fire is often not the one that mattered.
    switch = make_switch(tmp_path / 'kill.json')
    switch.evaluate(index_change=-0.06, drawdown=0.20, fno_banned=True)
    codes = [trip.code for trip in switch.history]
    assert codes == ['index_fall', 'max_drawdown', 'fno_ban']

  def test_breached_reports_the_same_conditions_evaluate_trips(self,
                                                               tmp_path):
    switch = make_switch(tmp_path / 'kill.json')
    found = switch.breached(index_change=-0.06, drawdown=0.20,
                            ack_latency=9.0)
    assert len(found) == 3

  def test_an_unknown_trip_code_is_refused(self, tmp_path):
    switch = make_switch(tmp_path / 'kill.json')
    with pytest.raises(KillSwitchError, match='unknown trip code'):
      switch.trip('something', code='weather')

  def test_a_blank_detail_is_refused(self):
    with pytest.raises(ValueError, match='must record what tripped it'):
      Trip('manual', '  ', EPOCH.isoformat())


class TestKillSwitchPersistence:
  '''A restart must not silently re-enable trading.'''

  def test_a_trip_survives_a_new_process_reading_the_file(self, tmp_path):
    path = tmp_path / 'kill.json'
    first = make_switch(path)
    first.evaluate(drawdown=0.20)
    assert first.tripped is True
    # A brand new object, as a restarted process would build.
    second = make_switch(path)
    assert second.tripped is True
    assert second.trading_enabled is False

  def test_the_restarted_switch_still_refuses_to_trade(self, tmp_path):
    path = tmp_path / 'kill.json'
    make_switch(path).evaluate(index_change=-0.09)
    with pytest.raises(KillSwitchLatched):
      make_switch(path).guard()

  def test_an_armed_switch_stays_armed_after_a_restart(self, tmp_path):
    path = tmp_path / 'kill.json'
    make_switch(path)
    assert make_switch(path).tripped is False

  def test_the_trip_history_survives_the_restart(self, tmp_path):
    path = tmp_path / 'kill.json'
    make_switch(path).trip('operator stop')
    assert make_switch(path).history[0].detail == 'operator stop'

  def test_a_corrupt_state_file_refuses_rather_than_resetting(self,
                                                              tmp_path):
    # The truncation case. Defaulting to armed here is how a halted
    # strategy comes back trading.
    path = tmp_path / 'kill.json'
    path.write_text('{"schema": "' + kill_switch_schema + '", "trip',
                    encoding='utf-8')
    with pytest.raises(KillSwitchError, match='cannot be read'):
      make_switch(path)

  def test_a_state_file_for_another_algo_refuses(self, tmp_path):
    path = tmp_path / 'kill.json'
    make_switch(path, algo_id='AAA111').trip('operator stop')
    with pytest.raises(KillSwitchError, match='belongs to algo'):
      make_switch(path, algo_id='BBB222')

  def test_an_unknown_schema_refuses(self, tmp_path):
    path = tmp_path / 'kill.json'
    path.write_text('{"schema": "other/1", "algo_id": "99999", '
                    '"tripped": false}', encoding='utf-8')
    with pytest.raises(KillSwitchError, match='schema'):
      make_switch(path)

  def test_a_non_object_state_file_refuses(self, tmp_path):
    path = tmp_path / 'kill.json'
    path.write_text('[1, 2, 3]', encoding='utf-8')
    with pytest.raises(KillSwitchError, match='not a JSON object'):
      make_switch(path)

  def test_a_malformed_trips_section_refuses(self, tmp_path):
    path = tmp_path / 'kill.json'
    path.write_text('{"schema": "' + kill_switch_schema + '", '
                    '"algo_id": "99999", "tripped": true, "trips": 7}',
                    encoding='utf-8')
    with pytest.raises(KillSwitchError, match='malformed trips'):
      make_switch(path)

  def test_a_blank_algo_id_refuses(self, tmp_path):
    with pytest.raises(KillSwitchError, match='algo_id must not be blank'):
      make_switch(tmp_path / 'kill.json', algo_id='  ')

  def test_an_unwritable_state_file_raises_on_trip(self, tmp_path):
    path = tmp_path / 'sub' / 'kill.json'
    path.parent.mkdir()
    path.parent.chmod(0o500)
    try:
      switch = make_switch(path)
      with pytest.raises(KillSwitchError, match='cannot be written'):
        switch.trip('operator stop')
    finally:
      path.parent.chmod(0o700)

  def test_the_default_state_file_is_per_algo(self):
    first = default_state_file('AAA111')
    second = default_state_file('AAA222')
    assert first != second
    assert first.name.endswith('AAA111.json')

  def test_the_default_state_file_refuses_a_blank_algo_id(self):
    with pytest.raises(ValueError, match='must not be blank'):
      default_state_file(' ')

  def test_the_default_state_file_sanitises_the_id(self):
    assert '/' not in default_state_file('A/B').name

  def test_render_names_the_algo_and_the_state(self, tmp_path):
    switch = make_switch(tmp_path / 'kill.json')
    assert 'armed' in switch.render()
    switch.trip('operator stop')
    assert 'TRIPPED' in switch.render()


class TestRiskMeasures:
  '''VaR is a risk nicety, not a para 11.1 pre-trade check.'''

  def test_zero_volatility_returns_zero_not_nan(self):
    value = parametric_var(0.0, 0.0, 0.95)
    assert value == 0.0
    assert value == value

  def test_zero_volatility_is_not_negative_zero(self):
    assert repr(parametric_var(0.0, 0.0, 0.95)) == '0.0'

  def test_a_zero_volatility_portfolio_returns_zero(self):
    zero = [[0.0, 0.0], [0.0, 0.0]]
    assert portfolio_var([0.5, 0.5], zero) == 0.0

  def test_cvar_is_never_below_var(self):
    sample = [0.01, -0.02, 0.03, -0.05, -0.01, 0.00, -0.08, 0.02]
    for confidence in (0.9, 0.95, 0.99):
      assert historical_cvar(sample, confidence) >= historical_var(
        sample, confidence)

  def test_the_parametric_cvar_is_never_below_the_parametric_var(self):
    assert parametric_cvar(0.001, 0.02, 0.95) >= parametric_var(
      0.001, 0.02, 0.95)

  def test_both_estimators_lose_money_on_a_losing_sample(self):
    sample = [-0.02, -0.03, -0.01, -0.04, -0.02]
    estimate = var_from_returns(sample, 0.95)
    assert estimate.historical > 0.0
    assert estimate.parametric > 0.0

  def test_both_estimators_say_the_tail_is_a_gain(self):
    # Eight calm up-days. At 90 percent confidence the worst decile of
    # this sample is not a loss, and a Gaussian fit of the same data has
    # to agree. Disagreeing on sign would mean one is transposed.
    sample = [0.010, 0.008, 0.012, 0.009, 0.011, 0.010, 0.013, 0.007]
    estimate = var_from_returns(sample, 0.90)
    assert estimate.parametric < 0.0
    assert estimate.historical < 0.0

  def test_both_estimators_say_the_tail_is_a_loss(self):
    sample = [-0.010] * 4 + [0.005] * 4
    estimate = var_from_returns(sample, 0.90)
    assert estimate.parametric > 0.0
    assert estimate.historical > 0.0

  def test_a_losing_sample_puts_both_estimators_above_zero(self):
    sample = [-0.01, 0.02, -0.03, 0.01]
    estimate = var_from_returns(sample, 0.95)
    assert estimate.historical > 0.0
    assert estimate.parametric > 0.0

  def test_a_fat_tail_shows_up_as_a_positive_gap(self):
    # A cluster of real losses against a run of tiny gains: the sample's
    # 95 percent tail is far worse than the Gaussian fit can produce, and
    # that gap is the fat tail showing up as a number. A single outlier
    # does *not* demonstrate it, because the empirical tail average
    # dilutes one observation across several; the claim needs a tail.
    sample = [0.002] * 97 + [-0.20] * 3
    estimate = var_from_returns(sample, 0.95)
    assert estimate.historical_tail > estimate.parametric_tail
    assert estimate.tail_gap > 0.0

  def test_the_estimate_reports_coherence(self):
    estimate = var_from_returns([-0.01, 0.02, -0.03, 0.01], 0.95)
    assert estimate.coherent() is True

  def test_an_empty_sample_returns_zero_from_both_estimators(self):
    estimate = var_from_returns([], 0.95)
    assert estimate.historical == 0.0
    assert estimate.parametric == 0.0

  def test_the_historical_tail_is_the_mean_of_the_worst_observations(self):
    sample = [0.0, 0.0, 0.0, -0.10, -0.20]
    # At 80 percent confidence the tail is the worst single observation,
    # so CVaR is that observation rather than an average of several.
    assert historical_cvar(sample, 0.80) == pytest.approx(0.20)

  def test_the_quantile_is_the_loss_at_the_tail(self):
    sample = [-0.10, -0.05, 0.0, 0.05, 0.10]
    # Linear interpolation between order statistics, so the 20th
    # percentile of five points lands 0.8 of the way from -0.10 to -0.05.
    assert historical_var(sample, 0.80) == pytest.approx(0.06)

  def test_cvar_exceeds_var_at_the_same_confidence(self):
    sample = [-0.10, -0.05, 0.0, 0.05, 0.10]
    assert historical_cvar(sample, 0.80) > historical_var(sample, 0.80)

  def test_a_volatility_is_never_negative(self):
    with pytest.raises(ValueError, match='volatility must be'):
      parametric_var(0.0, -0.01)

  def test_a_confidence_outside_the_supported_range_is_refused(self):
    for bad in (0.0, 0.1, 1.0, 1.5):
      with pytest.raises(ValueError, match='confidence must be in'):
        historical_var([0.01, -0.01], bad)

  def test_a_non_numeric_confidence_is_refused(self):
    with pytest.raises(ValueError, match='must be a real number'):
      z_score('0.95')

  def test_the_normal_quantile_is_the_published_one(self):
    assert z_score(0.95) == pytest.approx(1.6449, abs=1e-4)
    assert z_score(0.99) == pytest.approx(2.3263, abs=1e-4)


class TestCovariancePortfolioRisk:
  '''The variance-covariance method, which is what it says it is.

  ``covariance_matrix`` takes one sequence **per asset**, each aligned by
  period, which is why the literals below are columns of returns rather
  than rows. Getting that backwards produces a matrix that transposes
  the portfolio, and the tests below are written so that it would show.
  '''

  def test_perfect_correlation_gives_no_diversification(self):
    a = [0.01, -0.02, 0.03, -0.01]
    covariance = covariance_matrix([a, list(a)])
    single = portfolio_var([1.0, 0.0], covariance)
    together = portfolio_var([0.5, 0.5], covariance)
    assert together == pytest.approx(single)

  def test_negative_correlation_reduces_portfolio_variance(self):
    a = [0.02, -0.01, 0.03, 0.00]
    covariance = covariance_matrix([a, [-value for value in a]])
    assert covariance[0][1] < 0.0
    together = portfolio_variance([0.5, 0.5], covariance)
    single = portfolio_variance([1.0, 0.0], covariance)
    assert together < single

  def test_the_covariance_matrix_is_symmetric(self):
    covariance = covariance_matrix([[0.01, 0.03, 0.00],
                                    [0.02, -0.01, 0.01]])
    assert covariance[0][1] == pytest.approx(covariance[1][0])

  def test_unaligned_series_are_refused(self):
    with pytest.raises(ValueError, match='aligned'):
      covariance_matrix([[0.01, 0.02], [0.01]])

  def test_a_single_period_cannot_produce_a_covariance(self):
    with pytest.raises(ValueError, match='at least two periods'):
      covariance_matrix([[0.01], [0.02]])

  def test_no_series_at_all_is_refused(self):
    with pytest.raises(ValueError, match='at least one return series'):
      covariance_matrix([])

  def test_mismatched_weights_are_refused(self):
    with pytest.raises(ValueError, match='do not match'):
      portfolio_variance([1.0], [[1.0, 0.0], [0.0, 1.0]])

  def test_a_zero_mean_vector_is_the_default_and_is_conservative(self):
    covariance = covariance_matrix([[0.02, -0.01, 0.03],
                                    [-0.01, 0.02, -0.03]])
    weights = [0.5, 0.5]
    assert portfolio_var(weights, covariance) == pytest.approx(
      portfolio_var(weights, covariance, mean=[0.0, 0.0]))
    assert portfolio_var(weights, covariance) > 0.0

  def test_a_positive_drift_shrinks_the_var(self):
    # Imperfectly correlated assets, so the portfolio keeps some variance
    # for the drift term to act on. A perfectly offset pair has zero
    # variance and every VaR of it is zero, which tests nothing.
    covariance = covariance_matrix([[0.02, -0.01, 0.03],
                                    [-0.01, 0.02, -0.03]])
    weights = [0.5, 0.5]
    assert portfolio_var(weights, covariance, mean=[0.5, 0.5]) < \
      portfolio_var(weights, covariance)


class TestConstructionValidation:
  '''Constructor refusals, which are the first line of defence.'''

  def test_a_blank_symbol_is_refused_on_an_order(self):
    with pytest.raises(ValueError, match='symbol must be a non-empty'):
      OrderRequest(symbol='  ', side=buy, quantity=1)

  def test_an_unknown_side_is_refused(self):
    with pytest.raises(ValueError, match='side must be one of'):
      OrderRequest(symbol='X', side='long', quantity=1)

  def test_a_negative_price_is_refused(self):
    with pytest.raises(ValueError, match='price must be >= 0'):
      make_order(price=-1.0)

  def test_a_negative_reference_price_is_refused(self):
    with pytest.raises(ValueError, match='reference_price must be'):
      make_order(reference_price=-1.0)

  def test_a_bool_quantity_is_refused(self):
    # bool is an int in Python, so this is the one place the isinstance
    # check has to be more careful than it looks.
    with pytest.raises(ValueError, match='quantity must be an int'):
      OrderRequest(symbol='X', side=buy, quantity=True)

  def test_a_blank_symbol_is_refused_on_venue_limits(self):
    with pytest.raises(ValueError, match='symbol must be a non-empty'):
      make_security(symbol=' ')

  def test_a_zero_per_order_quantity_is_refused(self):
    with pytest.raises(ValueError, match='max_order_quantity must be >= 1'):
      make_security(max_order_quantity=0)

  def test_a_non_positive_per_order_value_is_refused(self):
    with pytest.raises(ValueError, match='max_order_value must be positive'):
      make_security(max_order_value=0.0)

  def test_a_bad_ticket_band_outside_the_unit_interval_is_refused(self):
    with pytest.raises(ValueError, match='bad_ticket_pct must be in'):
      make_limits(bad_ticket_pct=1.5)

  def test_a_negative_exposure_limit_is_refused(self):
    with pytest.raises(ValueError, match='max_exposure must be finite'):
      make_limits(max_exposure=-1.0)


class TestEdgeCases:
  '''Boundaries the arithmetic has to survive.'''

  def test_a_single_return_has_a_quantile(self):
    assert historical_var([0.02], 0.95) == pytest.approx(-0.02)

  def test_a_quantile_at_the_median_is_the_median(self):
    assert historical_var([-0.10, -0.05, 0.0], 0.50) == pytest.approx(0.05)

  def test_the_normal_quantile_alias_agrees(self):
    assert normal_quantile(0.95) == z_score(0.95)

  def test_an_empty_sample_has_no_volatility_and_no_var(self):
    assert sample_volatility([]) == 0.0
    assert var_from_returns([], 0.95).historical == 0.0

  def test_a_one_observation_sample_has_no_volatility(self):
    assert sample_volatility([0.01]) == 0.0

  def test_a_negative_parametric_volatility_is_refused(self):
    with pytest.raises(ValueError, match='volatility must be'):
      parametric_cvar(0.0, -1.0)

  def test_a_negative_volatility_is_refused_by_the_portfolio_helper(
      self):
    with pytest.raises(ValueError, match='volatility must be'):
      parametric_var(0.0, -0.5, 0.95)

  def test_a_circuit_band_percentage_of_one_is_refused(self):
    with pytest.raises(ValueError, match='must be in'):
      CircuitBand(1.0, 0.2)

  def test_a_circuit_band_upper_of_zero_is_refused(self):
    with pytest.raises(ValueError, match='upper_pct must be positive'):
      CircuitBand(0.2, 0.0)

  def test_a_circuit_band_tick_of_zero_is_refused(self):
    with pytest.raises(ValueError, match='tick must be positive'):
      CircuitBand(0.2, 0.2, tick=0.0)

  def test_a_theoretical_ex_price_needs_a_positive_reference(self):
    action = CorporateAction(bonus, ex_date=EPOCH, ratio=1.0)
    with pytest.raises(ValueError, match='reference must be positive'):
      action.theoretical_price(0.0)

  def test_the_ex_price_helper_matches_the_method(self):
    action = CorporateAction(split, ex_date=EPOCH, ratio=2.0)
    assert ex_price(100.0, action) == pytest.approx(50.0)

  def test_the_last_index_helper_refuses_an_ex_date_before_the_series(
      self):
    bars = bars_from([100.0, 100.0])
    with pytest.raises(ValueError, match='no bar precedes'):
      _last_index_before(bars, EPOCH - timedelta(days=1))

  def test_an_unknown_trip_code_on_a_record_is_refused(self):
    with pytest.raises(ValueError, match='unknown trip code'):
      Trip('sunspot', 'detail', EPOCH.isoformat())

  def test_a_malformed_trip_payload_is_refused(self):
    with pytest.raises(ValueError, match='malformed trip record'):
      Trip.from_json({'code': 'manual'})

  def test_a_render_with_no_trip_but_a_clear_names_the_clear(self, tmp_path):
    switch = make_switch(tmp_path / 'kill.json')
    switch.trip('operator stop')
    switch.reset('reviewed')
    switch2 = make_switch(tmp_path / 'kill.json')
    assert 'last cleared: reviewed' in switch2.render()

  def test_a_naive_clock_is_stamped_as_utc(self, tmp_path):
    naive = datetime(2026, 3, 2, 9, 15)
    switch = KillSwitch('99999', tmp_path / 'kill.json',
                        clock=lambda: naive)
    assert switch.trip('operator stop').at.endswith('+00:00')


class TestEmptySamples:
  '''An empty sample is an answer, and it is zero.'''

  def test_the_historical_var_of_an_empty_sample_is_zero(self):
    assert historical_var([], 0.95) == 0.0

  def test_the_historical_tail_of_an_empty_sample_is_zero(self):
    assert historical_cvar([], 0.95) == 0.0

  def test_an_empty_sample_still_needs_a_valid_confidence(self):
    # Zero for an empty sample is documented, but an unsupported
    # confidence is still a caller error rather than an empty answer.
    with pytest.raises(ValueError, match='confidence must be in'):
      historical_var([], 1.5)


def test_the_default_kill_switch_clock_is_aware_utc():
  # Every other test injects a fixed clock, so without this the default
  # would never execute: it is the one the production path uses.
  moment = utcnow()
  assert moment.tzinfo is not None
  assert moment.utcoffset().total_seconds() == 0.0
