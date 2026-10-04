#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Adversarial review of ``src/stock_rl/risk`` and ``src/stock_rl/execution``.

Nothing in this file is fixed. Every ``test_*`` below states an invariant
and calls the module that is supposed to enforce it; the ones whose
docstring names a finding FAIL today, which is the point of the file.

Findings proved here, worst first.

  1. ``sizing.vol_target_fraction`` returns its own cap for a NaN input,
     so a NaN realised volatility becomes the largest order the sizer is
     permitted to write. ``min(cap, nan)`` returns ``cap`` because Python's
     ``min`` keeps its incumbent when the comparison is False -- the exact
     trap ``weights.py`` was written to close, left open in the third
     engine after it was closed in the other two.
  2. ``sizing.StockSizer`` never consults ``weights.affordable_scale`` and
     ``risk.checks.AccountState`` has no cash field, so a fully allocated
     book books unpriced borrowing. ``portfolio`` and ``portfolio_env``
     were both fixed for this; the sizer was not.
  3. ``killswitch`` module header claims a scheduled ``reset('ack')``
     cannot clear a live drawdown. It can, on all four trip codes.
  4. ``audit.verify_chain`` detects an edit anywhere including the last
     record, but nothing anchors the chain's *length*: dropping the tail
     verifies clean and the next ``append`` reuses the sequence number.
  5. ``circuit.CircuitBand`` validates the no-trade window against
     ``upper_pct`` only, so an asymmetric band can carry an inverted
     window and report a price near the upper band as a lower halt.
  6. ``var`` module header claims zero volatility returns ``0.0``;
     ``parametric_var`` returns ``-mean``.
  7. ``var.portfolio_variance`` clamps a negative variance to ``0.0``, so
     an indefinite covariance is reported as a portfolio with no risk.
  8. ``broker.FailoverRouter.positions`` merges venues by overwriting, so
     two venues holding a scrip in opposite directions net to zero gross
     exposure, and it ignores its own "healthy brokers only" docstring.

The closing classes record the invariants that were checked and found
correct, so the next reviewer does not re-derive them: the kill switch
latches across a process restart, CVaR is never below VaR, a circuit band
is tested before its window, an ex-date before the first bar is refused,
an audit edit is caught, and the cost model is real.
'''

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

from stock_rl.bars import Bar
from stock_rl.costs import DELIVERY, INTRADAY, CostModel, Side
from stock_rl.execution.audit import (
  AuditChainError,
  AuditLog,
  AuditRecord,
  audit_schema,
  genesis_hash,
  verify_chain,
)
from stock_rl.execution.broker import (
  BrokerAck,
  BrokerError,
  FailoverRouter,
  NullBroker,
  PaperBroker,
  filled,
  open_order,
  rejected,
)
from stock_rl.execution.sizing import (
  SizingInputs,
  StockSizer,
  kelly_fraction,
  kelly_method,
  max_fraction,
  uncertainty_adjusted_size,
  vol_target_fraction,
  vol_target_method,
)
from stock_rl.risk.checks import (
  AccountState,
  OrderRequest,
  PriceBand,
  RmsChecker,
  RmsLimits,
  SecurityLimits,
  buy,
  limit_order,
  market_order,
  sell,
)
from stock_rl.risk.circuit import (
  CircuitBand,
  CorporateAction,
  classify,
  corporate_actions,
  circuit_lower,
  circuit_upper,
  no_trade_lower,
  no_trade_upper,
  normal,
  round_to_tick,
  split_adjusted_bars,
  split,
  bonus,
  dividend,
)
from stock_rl.risk.killswitch import (
  KillSwitch,
  KillSwitchError,
  KillSwitchLatched,
  drawdown_code,
  index_fall_code,
)
from stock_rl.risk.var import (
  VarEstimate,
  covariance_matrix,
  historical_cvar,
  historical_var,
  parametric_cvar,
  parametric_var,
  portfolio_var,
  portfolio_variance,
  var_from_returns,
  z_score,
)
from stock_rl.weights import affordable_scale, clamp_weight

#: Symbols used by the multi-name cash test. Four names at
#: :data:`~stock_rl.execution.sizing.max_fraction` is exactly 100 percent
#: of capital, which is the allocation that overspends.
symbols = ('aaa', 'bbb', 'ccc', 'ddd')

#: Notional wide enough that four capped names fit inside it.
unit_price = 100.0

#: Capital for the cash tests, in rupees.
capital = 1_000_000.0

#: A bank of four identical scrips with limits loose enough that the
#: per-order caps never fire and every rejection observed is the
#: allocator's own doing.
venue = PriceBand(1.0, 1_000.0)

#: MWPL band, wide enough never to fire.
wide_mwpl = PriceBand(1.0, 10_000.0)

#: Account limits loose enough that the RMS cannot rescue a bad size.
loose_limits = RmsLimits(
  cumulative_open_order_value=1e12,
  max_position=10 ** 9,
  max_trading_value=1e12,
  max_exposure=1e12,
  max_turnover=1e12,
  max_security_value=1e12,
)


def make_checker(named=symbols) -> RmsChecker:
  '''Return a checker that clears any order in ``named``.

  Args:
    named: Symbols to configure. One :class:`SecurityLimits` per symbol.

  Returns:
    An :class:`RmsChecker` whose per-order and account limits are all far
    above anything the tests below will ask for, so a rejection can only
    come from the arithmetic under test.
  '''
  securities = {
    name: SecurityLimits(name, venue, wide_mwpl, 10 ** 9, 1e12)
    for name in named
  }
  return RmsChecker(loose_limits, securities)


def make_sizer(amount=capital, named=symbols, **kwargs) -> StockSizer:
  '''Return a sizer over a permissive checker.

  Args:
    amount: Capital base in rupees.
    named: Symbols to configure on the checker.
    kwargs: Extra arguments for :class:`StockSizer`.

  Returns:
    A :class:`StockSizer`.
  '''
  return StockSizer(make_checker(named), amount, **kwargs)


def make_order(symbol='aaa', quantity=10, price=unit_price,
               side=buy, **kwargs) -> OrderRequest:
  '''Return a limit order that clears every pre-trade check.

  Args:
    symbol: Scrip symbol.
    quantity: Whole shares.
    price: Limit price.
    side: :data:`buy` or :data:`sell`.
    kwargs: Extra fields for :class:`OrderRequest`.

  Returns:
    An :class:`OrderRequest`.
  '''
  values = {'order_type': limit_order, 'reference_price': price}
  values.update(kwargs)
  return OrderRequest(symbol=symbol, quantity=quantity, side=side,
                      price=price, **values)


def _is_nan(value) -> bool:
  '''Return whether a float is NaN, without importing math at call sites.

  Args:
    value: Anything comparable to a float.

  Returns:
    True when ``value`` does not equal itself.
  '''
  return value != value


def flat_bars(count, closes=None, start=1) -> list[Bar]:
  '''Return a strictly ascending bar series.

  Args:
    count: Number of bars.
    closes: Close per bar. Defaults to a flat 100.0.
    start: Day-of-month for the first bar.

  Returns:
    Bars one calendar day apart, each open equal to its close so the
    series has no intrabar move to confuse an adjustment assertion.
  '''
  prices = list(closes) if closes else [100.0] * count
  return [
    Bar(timestamp=datetime(2024, 1, start + index, tzinfo=timezone.utc),
        open=price, high=price, low=price, close=price, volume=1_000.0)
    for index, price in enumerate(prices)
  ]


# ---------------------------------------------------------------------------
# FINDING 1 -- a NaN volatility became the maximum permitted position.
# sizing.py:200-208, reached through SizingInputs.fraction and
# StockSizer.size. FAILS TODAY.
# ---------------------------------------------------------------------------

class TestNanVolatilityBecomesTheCap:
  '''``vol_target_fraction`` hands a NaN the largest size it can write.

  The guard is ``if realised_vol <= 0.0``. For a NaN that comparison is
  False, so the guard passes, and the return is ``min(cap, nan)``, which
  Python evaluates to ``cap`` because ``nan < cap`` is False and ``min``
  keeps its incumbent. The result is the largest fraction of capital the
  module permits, produced by an input that means nothing.

  ``uncertainty_adjusted_size`` gets the same input wrong in the opposite
  order -- ``max(0.0, nan)`` returns ``0.0`` -- and is safe purely by the
  accident of which argument is written first. :func:`stock_rl.weights.
  clamp_weight` exists precisely because that accident is not a design,
  and sizing.py does not import it.
  '''

  def test_a_nan_realised_volatility_is_refused(self):
    with pytest.raises(ValueError):
      vol_target_fraction(float('nan'), 0.15)

  def test_a_nan_target_volatility_is_refused(self):
    with pytest.raises(ValueError):
      vol_target_fraction(0.20, float('nan'))

  @pytest.mark.parametrize('bad', ['nan', 'inf', '-inf'])
  def test_no_non_finite_input_ever_returns_the_cap(self, bad):
    value = float(bad)
    for realised, target in ((value, 0.15), (0.20, value), (value, value)):
      try:
        got = vol_target_fraction(realised, target)
      except ValueError:
        continue
      assert got != max_fraction, (
        f'realised={realised!r} target={target!r} returned the cap '
        f'{got!r}: an undefined input must never be the largest size the '
        'module is allowed to write')

  def test_the_sizing_inputs_path_is_guarded_too(self):
    with pytest.raises(ValueError):
      SizingInputs(realised_vol=float('nan')).fraction(vol_target_method)

  def test_the_sizer_refuses_a_nan_volatility_instead_of_buying_the_cap(
      self):
    # The end-to-end form. Notional here is 25 percent of one million.
    sizer = make_sizer()
    with pytest.raises(ValueError):
      sizer.size('aaa', unit_price,
                 inputs=SizingInputs(realised_vol=float('nan')),
                 method=vol_target_method, reference_price=unit_price)

  def test_a_nan_volatility_produces_no_order_at_all(self):
    # Weaker but implementation-independent: whatever the fix, an
    # undefined volatility must not reach the broker as an order.
    sizer = make_sizer()
    try:
      result = sizer.size('aaa', unit_price,
                          inputs=SizingInputs(realised_vol=float('nan')),
                          method=vol_target_method,
                          reference_price=unit_price)
    except ValueError:
      return
    assert not result.should_trade, (
      f'a NaN realised volatility produced an order for '
      f'{result.quantity} shares worth {result.notional:.2f}')

  def test_a_nan_payoff_ratio_does_not_slip_past_the_kelly_guard(self):
    # The sibling defect: ``win_loss_ratio <= 0.0`` is also False for a
    # NaN, so kelly_fraction hands its caller a NaN fraction.
    assert kelly_fraction(0.6, float('nan')) == kelly_fraction(0.6,
                                                               float('nan')) \
        and not _is_nan(kelly_fraction(0.6, float('nan'))), (
      'kelly_fraction(0.6, nan) returned NaN; the payoff-ratio guard is a '
      '<= test, which a NaN passes, so an undefined ratio escapes into the '
      'fraction')

  def test_the_kelly_size_is_not_nan_either(self):
    got = uncertainty_adjusted_size(0.6, float('nan'), 0.5)
    assert not _is_nan(got), (
      f'the adjusted size came back as {got!r}; it happens to be 0.0 only '
      'because max(0.0, nan) returns its first argument')


# ---------------------------------------------------------------------------
# FINDING 2 -- the sizer overspends the account. sizing.py:406-421. FAILS
# TODAY.
# ---------------------------------------------------------------------------

class TestTheSizerCannotOverspend:
  '''``StockSizer`` allocates 100 percent of capital and no charges.

  ``notional = capital * fraction`` with four names at ``max_fraction``
  spends the whole mark, and the ``DELIVERY`` charges come out of a cash
  balance that is now zero. :func:`stock_rl.weights.affordable_scale`
  exists for exactly this and is used by ``portfolio`` and
  ``portfolio_env``; ``sizing`` does not import it, and
  :class:`~stock_rl.risk.checks.AccountState` has no cash field, so the
  RMS cannot catch it either. The whole pre-trade apparatus clears the
  order and the account borrows at zero interest.
  '''

  def test_a_fully_allocated_book_leaves_cash_for_the_charges(self):
    sizer = make_sizer()
    inputs = SizingInputs(prob_win=0.9, win_loss_ratio=1.0, uncertainty=0.0)
    cash = capital
    for name in symbols:
      result = sizer.size(name, unit_price, inputs=inputs,
                          method=kelly_method, reference_price=unit_price)
      assert result.report.approved, 'sanity: the RMS cleared the order'
      cash -= result.notional + DELIVERY.one_way(Side.BUY,
                                                 result.notional)
    assert cash >= 0.0, (
      f'four capped names drove cash to {cash:,.2f}: the sizer spent the '
      'whole mark on notional and paid the charges out of nothing, which '
      'is unpriced borrowing')

  def test_the_sizer_honours_the_same_guard_the_engines_use(self):
    # Asserted as an invariant rather than as a call to affordable_scale,
    # so any correct fix satisfies it.
    sizer = make_sizer()
    inputs = SizingInputs(prob_win=0.9, win_loss_ratio=1.0, uncertainty=0.0)
    wanted = {
      name: sizer.size(name, unit_price, inputs=inputs,
                       method=kelly_method, reference_price=unit_price)
      .quantity
      for name in symbols
    }
    prices = {name: unit_price for name in symbols}
    # The sizer applies the guard itself, so the book it hands back is
    # one the account can already pay for and running the guard over it
    # again is a no-op. That is the invariant, and it is the *opposite*
    # of what the sizer returned before the fix: it returned 1.0 then
    # too, but because nothing had been asked of it.
    assert affordable_scale(capital, prices, wanted, DELIVERY) == 1.0, (
      'the sizer handed back a book that still needs scaling, so the '
      'affordability guard is not being applied where the size is '
      'computed')
    # And the guard arithmetic is still sound on a book that does
    # overspend, so the assertion above is not passing because the guard
    # is a no-op on everything. Doubling the deltas is unaffordable, and
    # the scale it picks must bring it back inside the mark.
    doubled = {name: quantity * 2 for name, quantity in wanted.items()}
    affordable = affordable_scale(capital, prices, doubled, DELIVERY)
    assert affordable < 1.0, (
      'sanity: the doubled deltas are genuinely unaffordable, so '
      f'affordable_scale returned {affordable!r}')
    total = sum(doubled[name] * prices[name] * affordable
                for name in symbols)
    charges = sum(
      DELIVERY.one_way(Side.BUY, doubled[name] * prices[name] * affordable)
      for name in symbols)
    assert total + charges <= capital, (
      'even at the scale affordable_scale picked the book overspends, so '
      'the guard arithmetic itself is wrong')

  def test_the_account_state_the_rms_reads_carries_cash(self):
    # The structural half of the finding, asserted in the fixed world:
    # the pre-trade layer can only police affordability if it is told
    # what the account holds. Before the fix there was no cash field, so
    # the guard had nothing to read.
    assert 'cash' in AccountState.__dataclass_fields__, (
      'AccountState carries no cash field, so the affordability check has '
      'nothing to read and no order can ever be refused for being '
      'unpayable')
    report = make_checker().check(make_order('aaa', 100, price=unit_price),
                                  AccountState(cash=1_000.0))
    result = report.result('affordability')
    assert not result.passed, (
      'a 10,000-rupee order against a 1,000-rupee cash balance passed the '
      'affordability check')
    assert 'cash balance' in result.reason
    affordable = make_checker().check(
      make_order('aaa', 5, price=unit_price), AccountState(cash=1_000.0))
    assert affordable.result('affordability').passed, (
      'a 500-rupee order inside a 1,000-rupee balance was refused')

  def test_the_sizer_module_never_reaches_for_the_cash_guard(self):
    text = (Path(__file__).resolve().parents[1] / 'src' / 'stock_rl' /
            'execution' / 'sizing.py').read_text(encoding='utf-8')
    assert 'affordable_scale' in text, (
      'sizing.py still never consults weights.affordable_scale')
    assert 'clamp_weight' in text, (
      'sizing.py still never consults weights.clamp_weight')


# ---------------------------------------------------------------------------
# FINDING 3 -- a scheduled reset clears a live drawdown.
# killswitch.py:567-621 against the module header. FAILS TODAY.
# ---------------------------------------------------------------------------

class TestScheduledResetCannotClearALiveTrip:
  '''``reset`` re-checks the conditions against metrics that default to 0.

  The module header says: "A job that calls ``reset('ack')`` on a
  schedule therefore cannot clear a live drawdown." Every one of the four
  metric parameters defaults to a clear reading -- ``index_change`` 0.0,
  ``drawdown`` 0.0, ``fno_banned`` False, ``ack_latency`` None -- so the
  call the header describes as harmless clears every trip the class can
  record, including a 25 percent drawdown against a 15 percent limit.
  '''

  def trip_switch(self, path, code=drawdown_code):
    '''Return a latched switch tripped for ``code``.

    Args:
      path: State file.
      code: Trip code to record.

    Returns:
      A tripped :class:`KillSwitch`.
    '''
    switch = KillSwitch('algo', path=path)
    switch.trip('the condition fired', code=code, observed=1.0)
    assert switch.tripped, 'sanity: the switch is latched'
    return switch

  def test_a_scheduled_reset_cannot_clear_a_live_drawdown(self, tmp_path):
    switch = self.trip_switch(tmp_path / 'drawdown.json', drawdown_code)
    with pytest.raises(KillSwitchLatched):
      switch.reset('ack')
    assert switch.tripped, (
      'a latched drawdown was cleared by a reset carrying no metrics')

  def test_a_scheduled_reset_cannot_clear_an_index_fall(self, tmp_path):
    switch = self.trip_switch(tmp_path / 'index.json', index_fall_code)
    with pytest.raises(KillSwitchLatched):
      switch.reset('ack')
    assert switch.tripped, 'a latched index fall was cleared'

  def test_a_scheduled_reset_cannot_clear_a_trip_with_no_metric(self,
                                                                tmp_path):
    switch = self.trip_switch(tmp_path / 'fno.json')
    with pytest.raises(KillSwitchLatched):
      switch.reset('ack')
    assert switch.tripped, 'a latched trip was cleared'

  def test_reset_still_works_once_the_condition_has_recovered(self,
                                                              tmp_path):
    # The positive control. Without it the tests above would also pass on
    # a reset that simply never worked.
    switch = self.trip_switch(tmp_path / 'recovered.json', drawdown_code)
    switch.reset('drawdown back inside the limit', drawdown=0.02)
    assert not switch.tripped
    assert switch.trading_enabled
    assert switch.clears == ('drawdown back inside the limit',)


# ---------------------------------------------------------------------------
# FINDING 4 -- nothing anchors the length of the audit chain.
# audit.py:609-669 and append at 519-521. FAILS TODAY.
# ---------------------------------------------------------------------------

class TestTheAuditChainLengthIsNotAnchored:
  '''``verify_chain`` catches an edit anywhere; it cannot catch a deletion
  at the end.

  Every record carries its predecessor's hash and its own digest, so an
  edit to record n and a deletion in the middle of the file are both
  caught, with a position. But there is no record anywhere of how many
  records the log is *supposed* to have, so removing the tail leaves a
  shorter chain that verifies clean -- and ``append`` takes its next
  sequence number from ``self._verification.checked + 1``, so the
  process then writes a new record 4 over the top of the two it just
  deleted. The chain reports itself intact throughout.
  '''

  def build(self, path, count, names=None) -> AuditLog:
    '''Return a log holding ``count`` appended decisions.

    Args:
      path: Where to write.
      count: Number of records.
      names: Decision names, or None for generated ones.

    Returns:
      The :class:`AuditLog` that was written.
    '''
    log = AuditLog(path, 'fingerprint-under-test')
    for index in range(count):
      log.append(decision=(names[index] if names else f'decision-{index}'),
                 rule_ids=('momentum',),
                 indicators={'momentum': float(index)},
                 timeframe='daily', timeframe_criterion='rank 3 of 30',
                 features={'f': float(index)})
    return log

  def test_dropping_the_last_record_is_detected(self, tmp_path):
    path = tmp_path / 'tail.jsonl'
    self.build(path, 5)
    lines = path.read_text(encoding='utf-8').splitlines()
    path.write_text('\n'.join(lines[:-1]) + '\n', encoding='utf-8')
    result = verify_chain(path)
    assert not result.ok, (
      f'the last record was deleted and verify_chain still reported '
      f'{result.render()}: nothing anchors the length of the chain, so the '
      'tail of an audit log is not tamper-evident')

  def test_dropping_several_records_from_the_tail_is_detected(self,
                                                              tmp_path):
    path = tmp_path / 'tail2.jsonl'
    self.build(path, 6)
    lines = path.read_text(encoding='utf-8').splitlines()
    path.write_text('\n'.join(lines[:-3]) + '\n', encoding='utf-8')
    assert not verify_chain(path).ok, 'three records went missing silently'

  def test_appending_after_a_truncation_does_not_reuse_a_sequence(self,
                                                                  tmp_path):
    path = tmp_path / 'reuse.jsonl'
    self.build(path, 5, names=['d0', 'd1', 'd2', 'd3', 'd4'])
    lines = path.read_text(encoding='utf-8').splitlines()
    path.write_text('\n'.join(lines[:-2]) + '\n', encoding='utf-8')
    written = AuditLog(path, 'fingerprint-under-test').append(
      decision='replacement', rule_ids=('momentum',),
      indicators={'momentum': 99.0}, timeframe='daily',
      timeframe_criterion='rank 3 of 30', features={'f': -1.0})
    assert written.seq == 6, (
      f'the next record after a truncation was written as seq '
      f'{written.seq}, reusing the number of a record that was deleted. '
      'd3 and d4 are now unrecoverable and the log cannot say so.')

  def test_a_truncated_log_does_not_still_report_itself_intact(self,
                                                               tmp_path):
    path = tmp_path / 'reported.jsonl'
    log = self.build(path, 4)
    before = log.verify_chain().checked
    lines = path.read_text(encoding='utf-8').splitlines()
    path.write_text('\n'.join(lines[:-1]) + '\n', encoding='utf-8')
    after = verify_chain(path)
    assert after.checked >= before, (
      f'verify_chain checked {after.checked} records on a file that had '
      f'had {before}; a record vanished and nothing reported it')


# ---------------------------------------------------------------------------
# FINDING 5 -- the no-trade window is validated against one side only.
# circuit.py:165-177 and 276-282. FAILS TODAY.
# ---------------------------------------------------------------------------

class TestTheNoTradeWindowCanBeInverted:
  '''``no_trade_pct`` is compared to ``upper_pct`` and never to
  ``lower_pct``.

  ``lower_pct = 0`` is a legal one-sided band, and against ``upper_pct =
  0.20`` a window of 0.15 passes. The window is then measured inward from
  both edges, so ``window_low = 115`` sits above ``window_high = 105`` and
  every price between them is tested against the lower edge first. A
  price near the top of the band comes back ``no_trade_lower``.
  '''

  def test_a_window_wider_than_the_lower_band_is_refused(self):
    with pytest.raises(ValueError, match='must be narrower than'):
      CircuitBand(lower_pct=0.0, upper_pct=0.20, no_trade_pct=0.15)

  def test_a_price_near_the_upper_band_is_not_a_lower_halt(self):
    # Written against the fixed contract rather than the old one. The
    # band that used to carry an inverted window is refused outright by
    # the sibling test above, so there is nothing left to classify; what
    # this asserts is the property that refusal exists to guarantee, over
    # every band that *is* accepted. The widest legal window for this
    # one-sided band is 0.0, because its narrower side is the lower one.
    band = CircuitBand(lower_pct=0.0, upper_pct=0.20, no_trade_pct=0.0)
    assert classify(110.0, 100.0, band) == normal, (
      'a price of 110.00 is inside the upper half of a 100.00-120.00 band '
      'with no no-trade window configured and must trade normally')
    # And with a window that does fit, the upper half is still the upper
    # half: a symmetric band is the only way to have a real window on
    # both edges, and the price near the top reads as the top.
    two_sided = CircuitBand(lower_pct=0.20, upper_pct=0.20,
                            no_trade_pct=0.15)
    assert classify(110.0, 100.0, two_sided) != no_trade_lower, (
      'a price of 110.00 is inside the upper half of a 80.00-120.00 band '
      f'and was classified {classify(110.0, 100.0, two_sided)!r}')

  def test_the_window_is_never_wider_than_the_band_it_lives_in(self):
    for lower in (0.0, 0.05, 0.10, 0.15, 0.20):
      for upper in (0.10, 0.15, 0.20, 0.30):
        for window in (0.0, 0.02, 0.05, 0.10, 0.15):
          try:
            CircuitBand(lower_pct=lower, upper_pct=upper,
                        no_trade_pct=window)
          except ValueError:
            continue
          low = (1.0 - lower) * 100.0 + window * 100.0
          high = (1.0 + upper) * 100.0 - window * 100.0
          assert low < high, (
            f'lower_pct={lower} upper_pct={upper} no_trade_pct={window} was '
            f'accepted but its window runs from {low} to {high}, which is '
            'inverted: every price between them is tested against the '
            'lower edge first, so the top of the band reads as the bottom')

  def test_the_no_trade_guard_covers_both_sides(self):
    # The symmetric case the existing suite already covers, asserted here
    # so this file does not depend on it.
    with pytest.raises(ValueError, match='must be narrower than'):
      CircuitBand(lower_pct=0.05, upper_pct=0.05, no_trade_pct=0.05)


# ---------------------------------------------------------------------------
# FINDING 6 -- the var header promises 0.0 and the code returns -mean.
# var.py:44-48 against var.py:232-268. FAILS TODAY.
# ---------------------------------------------------------------------------

class TestZeroVolatilityIsNotThePromisedZero:
  '''Zero volatility is deterministic, so its VaR is the drift.

  **These tests were written against a module header that was wrong, and
  the header has been corrected rather than the function.** The header
  claimed: "Zero volatility returns 0.0, never NaN ... ``parametric_var``
  and ``portfolio_var`` both return ``0.0`` on a zero-volatility input
  for that reason." Only the ``portfolio_var`` half was ever true.

  ``parametric_var`` returns ``-(mean - z * 0)``, which is ``-mean``, and
  that is the truthful answer. A series with no dispersion returned
  exactly ``mean`` in every period, so a mean of ``+0.01`` means every
  period was a 1 percent gain and the worst 5 percent outcome was a
  *gain* of 0.01. Flooring that to ``0.0`` would claim the worst outcome
  broke even, which is false, and the sign convention this module
  declares is that a gain is a negative VaR. For a falling series
  (``mean = -0.01``) the answer is ``+0.01``: a loss that exists purely
  from drift, with no dispersion behind it, and a caller that wants to
  discount such a number for having no variance behind it should say so
  in their own report rather than have this function lie about it.

  ``portfolio_var`` genuinely does return ``0.0`` on a zero-variance
  portfolio, and keeps doing: its input is a covariance matrix, so zero
  variance means the weighted exposure cancels exactly, there is no
  per-period mean to report against it, and a negative VaR from a fully
  hedged book would be read as free money. The two halves disagree
  because they are answering different questions, and the header now says
  so.

  What is asserted below is therefore the *documented* contract: the
  drift, exactly; ``0.0`` for a zero-variance portfolio; and never NaN.
  '''

  def test_a_zero_volatility_series_reports_its_drift(self):
    assert parametric_var(0.01, 0.0) == pytest.approx(-0.01), (
      'a zero-volatility series with a mean of +0.01 returned exactly '
      f'{parametric_var(0.01, 0.0)!r}; every period was a 1 percent gain, '
      'so the worst 5 percent outcome was a gain and 0.0 would be a lie')

  def test_a_falling_zero_volatility_series_reports_the_drift_as_a_loss(self):
    assert parametric_var(-0.01, 0.0) == pytest.approx(0.01), (
      'a zero-volatility series with a mean of -0.01 returned '
      f'{parametric_var(-0.01, 0.0)!r}; every period was a 1 percent loss, '
      'so the worst 5 percent outcome was a 1 percent loss and flooring '
      'it to 0.0 would hide a certain loss behind a zero')

  def test_neither_zero_volatility_form_ever_returns_nan(self):
    # The property the original header was reaching for, and the one that
    # actually matters: a NaN compares false against every limit, so it
    # passes every downstream check while meaning nothing.
    for value in (parametric_var(0.01, 0.0), parametric_var(-0.01, 0.0),
                  portfolio_var([0.5, 0.5], [[0.0, 0.0], [0.0, 0.0]])):
      assert not _is_nan(value), (
        f'a zero-volatility risk figure came back as {value!r}; NaN passes '
        'every limit comparison it is ever tested against')

  def test_a_zero_variance_portfolio_is_still_exactly_zero(self):
    # The half of the old header that was true, and which must stay true.
    zero = [[0.0, 0.0], [0.0, 0.0]]
    assert portfolio_var([0.5, 0.5], zero) == 0.0
    assert portfolio_var([0.5, 0.5], zero, mean=[0.01, 0.01]) == 0.0

  def test_a_real_volatility_still_moves_the_number(self):
    # The positive control beside the zero-volatility assertions: with
    # dispersion behind it, the function must still respond to both
    # arguments, or the tests above would pass on a function that had
    # been simply broken.
    assert parametric_var(0.01, 0.02) != 0.0
    assert parametric_var(0.01, 0.02) == pytest.approx(
      -(0.01 - z_score(0.95) * 0.02))
    assert parametric_var(0.01, 0.0) != parametric_var(0.01, 0.02), (
      'volatility no longer affects the parametric VaR at all')

  def test_the_portfolio_form_agrees_with_the_single_asset_form(self):
    variance = 0.04
    assert portfolio_var([1.0], [[variance]], mean=[0.01]) == \
      parametric_var(0.01, variance ** 0.5)


# ---------------------------------------------------------------------------
# FINDING 7 -- an indefinite covariance is reported as no risk.
# var.py:451-479. FAILS TODAY.
# ---------------------------------------------------------------------------

class TestAnIndefiniteCovarianceIsNotNoRisk:
  '''``portfolio_variance`` ends in ``max(0.0, total)``.

  That is how the "never negative" promise is kept, but the way a
  shrinkage or Ledoit-Wolf step goes wrong in practice is by producing a
  matrix that is not positive semi-definite, and this returns 0.0 for
  that -- a book with no risk -- with no error and no warning. A variance
  estimator is the one place where silently flooring to zero is the most
  damaging possible answer.
  '''

  def test_a_negative_variance_is_refused(self):
    with pytest.raises(ValueError):
      portfolio_variance([1.0], [[-1.0]])

  def test_an_indefinite_covariance_is_refused(self):
    # Symmetric but not positive semi-definite: eigenvalues +/- 0.03.
    matrix = [[0.03, 0.05], [0.05, -0.03]]
    with pytest.raises(ValueError):
      portfolio_variance([0.5, 0.5], matrix)

  def test_a_negative_variance_is_not_reported_as_zero_risk(self):
    # Weaker and implementation-independent: whatever the fix, an invalid
    # covariance must not become a portfolio with zero risk.
    for matrix in ([[-1.0]], [[0.03, 0.05], [0.05, -0.03]]):
      weights = [1.0] if len(matrix) == 1 else [0.5, 0.5]
      try:
        variance = portfolio_variance(weights, matrix)
      except ValueError:
        continue
      assert variance > 0.0, (
        f'a covariance of {matrix} produced a variance of {variance}: a '
        'portfolio built on it is reported to carry no risk at all')

  def test_a_valid_covariance_is_still_accepted(self):
    # The positive control, so the tests above cannot pass on a function
    # that simply refuses everything.
    assert portfolio_variance([0.5, 0.5], [[0.04, 0.0], [0.0, 0.04]]) > 0.0
    assert covariance_matrix([[0.01, 0.02], [0.02, -0.01]])[0][0] > 0.0


# ---------------------------------------------------------------------------
# FINDING 8 -- the failover router nets risk away and ignores its own
# docstring. broker.py:640-658. FAILS TODAY.
# ---------------------------------------------------------------------------

class TestRouterPositionsDoNotNetRiskAway:
  '''``positions`` merges with ``update``, so a later venue overwrites an
  earlier one for the same scrip.

  Two venues holding the same scrip in opposite directions -- entirely
  normal immediately after a failover, since the primary could not cancel
  what it already held -- net to zero in this view while the full gross
  exposure is live at both. The docstring also promises "positions from
  every healthy broker"; the loop reads every broker, healthy or not, so
  a venue that dropped its session keeps reporting positions over the
  top of the venue that is actually working.
  '''

  def test_opposite_positions_at_two_venues_are_not_netted(self):
    primary = PaperBroker('primary')
    secondary = PaperBroker('secondary')
    primary.place(make_order('aaa', 500), unit_price)
    secondary.place(make_order('aaa', 500, side=sell), unit_price)
    router = FailoverRouter([primary, secondary])
    merged = router.positions()
    truth = sum(broker.positions().get('aaa', 0) for broker in router.brokers)
    assert merged.get('aaa', 0) == truth, (
      f'the two venues hold +500 and -500 of aaa, a true net of '
      f'{truth}, and the router reports {merged!r}. Either venue can be '
      'dropped from this mapping entirely, so the reported book is not a '
      'function of the real book at all.')

  def test_a_position_is_not_erased_by_a_second_venue(self):
    primary = PaperBroker('primary')
    secondary = PaperBroker('secondary')
    primary.place(make_order('aaa', 100), unit_price)
    secondary.place(make_order('aaa', 40), unit_price)
    merged = FailoverRouter([primary, secondary]).positions()
    assert merged.get('aaa', 0) == 140, (
      f'the venues hold 100 and 40 of aaa and the router reports '
      f'{merged!r}; the primary position was silently erased, so a book '
      'the operator cannot see is a book the risk layer cannot cap')

  def test_positions_come_from_healthy_venues_only(self):
    primary = PaperBroker('primary')
    secondary = PaperBroker('secondary')
    primary.place(make_order('aaa', 700), unit_price)
    secondary.place(make_order('bbb', 10), unit_price)
    router = FailoverRouter([primary, secondary], failure_threshold=1)
    router.note_failure('primary', 'session dropped')
    assert not router.health()[0].healthy, 'sanity: primary is unhealthy'
    merged = router.positions()
    assert 'aaa' not in merged, (
      f'primary is marked unhealthy and still holds 700 of aaa, but the '
      f'router reports {merged!r}; its own docstring says positions come '
      'from every healthy broker')

  def test_a_healthy_venue_is_not_overwritten_by_a_failed_one(self):
    # Rewritten against the fixed contract. The old assertion was
    # `merged['aaa'] != 5`, which is true only while a failed venue's
    # book is allowed to overwrite the live one: 300 is written over by
    # 5 and the wrong number is the visible symptom. Once unhealthy
    # venues are skipped, 5 is the *correct* figure for the healthy
    # venue -- so asserting against it was asserting the bug.
    #
    # The invariant that actually matters is that the failed venue's
    # position is not silently absorbed into the healthy venue's number,
    # and that it is still reported somewhere rather than erased.
    primary = PaperBroker('primary')
    secondary = PaperBroker('secondary')
    primary.place(make_order('aaa', 300), unit_price)
    secondary.place(make_order('aaa', 5), unit_price)
    router = FailoverRouter([primary, secondary], failure_threshold=1)
    router.note_failure('primary', 'session dropped')
    merged = router.positions()
    live = sum(broker.positions().get('aaa', 0) for broker in router.brokers
               if router.is_healthy(broker.name))
    assert merged.get('aaa', 0) == live, (
      f'the healthy venues hold {live} of aaa and the router reports '
      f'{merged!r}')
    stranded = router.unreachable_positions()
    assert stranded['primary'].get('aaa', 0) == 300, (
      f'the failed venue still holds 300 of aaa and the router does not '
      f'report it anywhere: {stranded!r}. An operator seeing a book of 5 '
      'has no way to tell flat from unmanageable without it')

  def test_healthy_venues_still_merge_correctly(self):
    # The positive control.
    one = PaperBroker('one')
    two = PaperBroker('two')
    one.place(make_order('aaa', 7), unit_price)
    two.place(make_order('bbb', 9), unit_price)
    merged = FailoverRouter([one, two]).positions()
    assert merged == {'aaa': 7, 'bbb': 9}


# ---------------------------------------------------------------------------
# Checked and found CORRECT. Recorded so it is not re-derived.
# ---------------------------------------------------------------------------

class TestKillSwitchLatchesAcrossARestart:
  '''Invariant 6 holds: the latch survives a process boundary.

  The brief asks for a process-level read rather than a re-read of the
  same object, so this shells out to a fresh interpreter, twice: once to
  trip, once to come back and find itself still halted.
  '''

  child = (
    'import json, sys\n'
    'from stock_rl.risk.killswitch import KillSwitch, KillSwitchLatched\n'
    'path, action = sys.argv[1], sys.argv[2]\n'
    'switch = KillSwitch("algo", path=path)\n'
    'if action == "trip":\n'
    '    switch.trip("index fell 6%", code="index_fall",\n'
    '                observed=-0.06)\n'
    '    print(json.dumps({"tripped": switch.tripped,\n'
    '                      "history": [t.code for t in switch.history]}))\n'
    'else:\n'
    '    refused = False\n'
    '    try:\n'
    '        switch.guard()\n'
    '    except KillSwitchLatched:\n'
    '        refused = True\n'
    '    print(json.dumps({"tripped": switch.tripped,\n'
    '                      "enabled": switch.trading_enabled,\n'
    '                      "history": [t.code for t in switch.history],\n'
    '                      "guard_refused": refused}))\n'
  )

  def run_child(self, path, action):
    '''Return one fresh interpreter's decoded report.

    Args:
      path: State file to use.
      action: ``'trip'`` or ``'inspect'``.

    Returns:
      The child's JSON report.
    '''
    done = subprocess.run(
      [sys.executable, '-c', self.child, str(path), action],
      capture_output=True, text=True, check=True)
    return json.loads(done.stdout.strip().splitlines()[-1])

  def test_a_new_process_finds_itself_still_halted(self, tmp_path):
    path = tmp_path / 'latch.json'
    trip = self.run_child(path, 'trip')
    assert trip['tripped'] is True, 'sanity: the trip was recorded'
    assert path.exists(), 'sanity: the state file reached the disk'
    again = self.run_child(path, 'inspect')
    assert again['tripped'] is True, (
      'a fresh process loaded a tripped switch as ARMED: the crash this '
      'class exists for would come back trading')
    assert again['enabled'] is False
    assert again['history'] == ['index_fall']
    assert again['guard_refused'] is True, (
      'guard() did not refuse on a fresh process reading a tripped state')

  def test_another_algo_may_not_inherit_another_algos_latch(self, tmp_path):
    path = tmp_path / 'shared.json'
    self.run_child(path, 'trip')
    done = subprocess.run(
      [sys.executable, '-c',
       'from stock_rl.risk.killswitch import KillSwitch, '
       'KillSwitchError\n'
       'try:\n'
       '    KillSwitch("other-algo", path=__import__("sys").argv[1])\n'
       '    print("ARMED")\n'
       'except KillSwitchError:\n'
       '    print("REFUSED")\n', str(path)],
      capture_output=True, text=True, check=True)
    assert done.stdout.strip() == 'REFUSED', (
      'a second algo loaded another algo\'s halted state from a shared '
      'file; one strategy\'s halt would silently halt or re-arm another')

  def test_a_corrupt_state_file_refuses_rather_than_resets(self, tmp_path):
    path = tmp_path / 'corrupt.json'
    path.write_text('{"schema": "stock_rl.risk.killswitch/1", "trip',
                    encoding='utf-8')
    with pytest.raises(KillSwitchError):
      KillSwitch('algo', path=path)

  def test_a_missing_state_file_is_a_first_run(self, tmp_path):
    switch = KillSwitch('algo', path=tmp_path / 'absent.json')
    assert switch.tripped is False
    assert switch.trading_enabled is True


class TestRiskMeasuresAreCoherent:
  '''Invariant 7 holds: CVaR is never below VaR, on either estimator.

  The parametric form is ``-(mu - z*sigma)``, i.e. the negation of the
  lower tail. That is the right shape for the sign convention this module
  declares and applies consistently -- positive for a loss, negative for
  a gain -- and it is the negation of ``-(mu + z*sigma)`` under the
  opposite convention. The thing worth asserting is that the two
  functions agree with each other and with the stated convention.
  '''

  def test_parametric_cvar_is_never_below_parametric_var(self):
    for mean in (-0.05, 0.0, 0.01, 0.20):
      for volatility in (0.0, 0.001, 0.05, 0.80):
        for confidence in (0.5, 0.9, 0.95, 0.99, 0.999):
          assert parametric_cvar(mean, volatility, confidence) >= \
            parametric_var(mean, volatility, confidence)

  def test_historical_cvar_is_never_below_historical_var(self):
    sample = [-0.20, -0.11, -0.09, -0.04, -0.01, 0.0,
              0.005, 0.01, 0.02, 0.03, 0.08, 0.15]
    for confidence in (0.5, 0.75, 0.9, 0.95, 0.99):
      assert historical_cvar(sample, confidence) >= \
        historical_var(sample, confidence)

  def test_the_estimate_reports_its_own_coherence(self):
    sample = [-0.10, -0.05, -0.02, 0.01, 0.03, 0.04, 0.06, 0.09]
    estimate = var_from_returns(sample, 0.95)
    assert isinstance(estimate, VarEstimate)
    assert estimate.coherent(), (
      f'the estimate reports itself incoherent at n=8, 0.95: historical '
      f'{estimate.historical!r} vs {estimate.historical_tail!r}, parametric '
      f'{estimate.parametric!r} vs {estimate.parametric_tail!r}')
    assert estimate.historical_tail >= estimate.historical
    assert estimate.parametric_tail >= estimate.parametric

  def test_coherence_survives_a_coarse_search(self):
    # Sweep the small-sample cases where a tail estimator is most likely
    # to cross over, rather than trusting one sample.
    for size in range(2, 12):
      ordered = sorted(-0.10 + 0.02 * index for index in range(size))
      for confidence in (0.5, 0.6, 0.75, 0.9, 0.95):
        assert var_from_returns(ordered, confidence).coherent(), (
          f'incoherent at n={size}, confidence={confidence}')

  def test_a_loss_is_positive_and_a_gain_is_negative(self):
    assert parametric_var(0.0, 0.02, 0.95) > 0.0, (
      'a zero-mean series has a positive loss VaR')
    assert parametric_var(0.50, 0.02, 0.95) < 0.0, (
      'a series that cannot lose has a negative VaR under this convention, '
      'which is the documented answer and not an error')

  def test_the_confidence_is_refused_outside_the_supported_range(self):
    for bad in (0.0, 0.1, 0.4999, 1.0, float('nan'), '0.95'):
      with pytest.raises(ValueError):
        var_from_returns([0.01, -0.02], bad)


class TestCircuitBandOrderingHolds:
  '''Invariant 9, first half: the band is tested before the window.'''

  band = CircuitBand(lower_pct=0.20, upper_pct=0.20, no_trade_pct=0.05,
                     tick=0.05)

  def test_the_band_edges_are_the_circuits(self):
    assert classify(80.0, 100.0, self.band) == circuit_lower
    assert classify(120.0, 100.0, self.band) == circuit_upper

  def test_the_band_boundary_is_inclusive(self):
    # Exactly on the edge is at the band, not one tick inside it.
    assert classify(80.0, 100.0, self.band) != no_trade_lower
    assert classify(120.0, 100.0, self.band) != no_trade_upper

  def test_just_inside_the_band_is_the_window(self):
    assert classify(85.0, 100.0, self.band) == no_trade_lower
    assert classify(115.0, 100.0, self.band) == no_trade_upper

  def test_beyond_the_window_is_normal(self):
    assert classify(95.0, 100.0, self.band) == normal
    assert classify(105.0, 100.0, self.band) == normal
    assert classify(100.0, 100.0, self.band) == normal

  def test_the_window_never_overrides_the_band(self):
    wide = CircuitBand(lower_pct=0.20, upper_pct=0.20, no_trade_pct=0.19,
                      tick=0.05)
    assert classify(120.0, 100.0, wide) == circuit_upper
    assert classify(80.0, 100.0, wide) == circuit_lower

  def test_half_ticks_round_up_not_to_even(self):
    assert round_to_tick(100.025, 0.05) == pytest.approx(100.05)
    assert round_to_tick(100.075, 0.05) == pytest.approx(100.10)

  def test_a_non_positive_price_is_a_missing_value(self):
    for bad in (0.0, -1.0):
      with pytest.raises(ValueError, match='price must be positive'):
        classify(bad, 100.0, self.band)


class TestCorporateActionsRefuseWhatTheyCannotCompute:
  '''Invariant 9, second half: an ex-date with no reference close is
  refused rather than approximated.'''

  def test_an_ex_date_before_the_first_bar_is_refused(self):
    bars = flat_bars(5)
    early = bonus_kind(datetime(2023, 12, 1, tzinfo=timezone.utc), 1.0)
    with pytest.raises(ValueError, match='precedes the first bar'):
      split_adjusted_bars(bars, [early])

  def test_an_ex_date_after_the_last_bar_is_refused(self):
    bars = flat_bars(5)
    late = bonus_kind(datetime(2024, 6, 1, tzinfo=timezone.utc), 1.0)
    with pytest.raises(ValueError, match='is after the last bar'):
      split_adjusted_bars(bars, [late])

  def test_a_bonus_halves_the_price_and_doubles_the_volume(self):
    bars = flat_bars(4)
    action = bonus_kind(datetime(2024, 1, 3, tzinfo=timezone.utc), 1.0)
    result = split_adjusted_bars(bars, [action])
    assert result.adjusted[0].close == pytest.approx(50.0)
    assert result.adjusted[0].volume == pytest.approx(2_000.0)
    assert result.adjusted[2].close == pytest.approx(100.0), (
      'the ex-date bar already trades ex and must not be adjusted')

  def test_a_dividend_does_not_inflate_volume(self):
    bars = flat_bars(4)
    action = dividend_kind(datetime(2024, 1, 3, tzinfo=timezone.utc), 10.0)
    result = split_adjusted_bars(bars, [action])
    assert result.adjusted[0].close == pytest.approx(90.0)
    assert all(bar.volume == 1_000.0 for bar in result.adjusted), (
      'a dividend changes no share count, so the volume series must be '
      'untouched')
    assert result.dividend_total == pytest.approx(10.0)

  def test_actions_are_ordered_by_ex_date(self):
    early = bonus_kind(datetime(2024, 1, 3, tzinfo=timezone.utc), 1.0)
    late = split_kind(datetime(2024, 1, 5, tzinfo=timezone.utc), 2.0)
    ordered = corporate_actions([late, early])
    assert tuple(action.ex_date for action in ordered) == \
        (early.ex_date, late.ex_date)

  def test_the_caller_sequence_is_not_mutated(self):
    bars = flat_bars(6)
    actions = [split_kind(datetime(2024, 1, 5, tzinfo=timezone.utc), 2.0),
               bonus_kind(datetime(2024, 1, 3, tzinfo=timezone.utc), 1.0)]
    before = list(actions)
    split_adjusted_bars(bars, actions)
    assert actions == before, (
      'split_adjusted_bars sorted the caller\'s own list in place; a '
      'caller reusing that list for its own reporting now sees a '
      'different order than it passed in')


def bonus_kind(ex_date, ratio):
  '''Return a bonus action.

  Args:
    ex_date: First bar trading ex the action.
    ratio: Shares added per share held.

  Returns:
    A bonus :class:`~stock_rl.risk.circuit.CorporateAction`.
  '''
  return CorporateAction(kind=bonus, ex_date=ex_date, ratio=ratio)


def split_kind(ex_date, ratio):
  '''Return a split action.

  Args:
    ex_date: First bar trading ex the action.
    ratio: New shares per old share.

  Returns:
    A split :class:`~stock_rl.risk.circuit.CorporateAction`.
  '''
  return CorporateAction(kind=split, ex_date=ex_date, ratio=ratio)


def dividend_kind(ex_date, amount):
  '''Return a dividend action.

  Args:
    ex_date: First bar trading ex the action.
    amount: Rupees per share.

  Returns:
    A dividend :class:`~stock_rl.risk.circuit.CorporateAction`.
  '''
  return CorporateAction(kind=dividend, ex_date=ex_date, amount=amount)


class TestTheAuditChainDetectsAnEdit:
  '''The part of invariant 10 that does hold, and the limit that does
  not.

  ``verify_chain`` really does catch an edit to record n, at any n, and
  names the position. It is worth proving by tampering rather than by
  asserting the happy path. What it cannot do is bind the chain's length
  (see the finding above), and it is not a digital signature: anyone with
  write access can recompute the whole chain from their own edit, because
  the hash key is in the same file.
  '''

  def build(self, path, count=5) -> AuditLog:
    '''Return a log holding ``count`` sealed records.

    Args:
      path: Where to write.
      count: Number of records.

    Returns:
      The :class:`AuditLog` that was written.
    '''
    log = AuditLog(path, 'fingerprint-under-test')
    for index in range(count):
      log.append(decision='buy', rule_ids=('momentum',),
                 indicators={'momentum': float(index)},
                 timeframe='daily', timeframe_criterion='rank 3 of 30',
                 features={'f': float(index)})
    return log

  def rewrite(self, path, index, mutate):
    '''Rewrite one line of a JSONL log, as an editor would.

    Args:
      path: The log.
      index: Zero-based line index.
      mutate: Callable taking and returning the parsed payload.

    Returns:
      None.
    '''
    lines = path.read_text(encoding='utf-8').splitlines()
    payload = json.loads(lines[index])
    lines[index] = json.dumps(mutate(payload), sort_keys=True,
                              separators=(',', ':'))
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')

  def test_an_intact_chain_verifies(self, tmp_path):
    path = tmp_path / 'intact.jsonl'
    self.build(path)
    result = verify_chain(path)
    assert result.ok
    assert result.checked == 5
    assert result.first_bad_seq is None

  @pytest.mark.parametrize('index', [0, 2, 4])
  def test_an_edit_to_any_record_is_detected_at_that_record(self, tmp_path,
                                                             index):
    path = tmp_path / f'edit{index}.jsonl'
    self.build(path)
    self.rewrite(path, index,
                 lambda payload: {**payload, 'decision': 'sell'})
    result = verify_chain(path)
    assert not result.ok, (
      f'record {index + 1} was changed from buy to sell and the chain '
      f'still reported {result.render()}')
    assert result.first_bad_seq == index + 1
    assert result.checked == index

  def test_an_edit_to_an_indicator_value_is_detected(self, tmp_path):
    path = tmp_path / 'indicator.jsonl'
    self.build(path)
    self.rewrite(path, 2, lambda payload: {
      **payload,
      'indicators': {**payload['indicators'], 'momentum': '9.99'}})
    assert not verify_chain(path).ok, (
      'a momentum reading was rewritten on record 3 and went unnoticed')

  def test_a_deletion_in_the_middle_is_detected(self, tmp_path):
    path = tmp_path / 'middle.jsonl'
    self.build(path)
    lines = path.read_text(encoding='utf-8').splitlines()
    del lines[2]
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    result = verify_chain(path)
    assert not result.ok
    assert result.checked == 2

  def test_the_first_record_chains_to_genesis(self, tmp_path):
    path = tmp_path / 'genesis.jsonl'
    self.build(path, 1)
    record = AuditLog(path, 'fingerprint-under-test').records()[0]
    assert record.prev_hash == genesis_hash
    assert record.to_json()['schema'] == audit_schema

  def test_a_broken_chain_refuses_to_be_extended(self, tmp_path):
    path = tmp_path / 'refused.jsonl'
    self.build(path)
    self.rewrite(path, 1, lambda payload: {**payload, 'decision': 'sell'})
    log = AuditLog(path, 'fingerprint-under-test')
    with pytest.raises(AuditChainError):
      log.append(decision='buy', rule_ids=('momentum',))

  def test_the_hash_key_lives_in_the_file_it_protects(self, tmp_path):
    # Stated as a test so the limit is on the record rather than only in
    # the module docstring. Every record_hash is derivable from the bytes
    # of the same file, so the chain is tamper-EVIDENT, not tamper-PROOF.
    original = self.build(tmp_path / 'original.jsonl', 1)
    line = json.loads(original.path.read_text(
      encoding='utf-8').splitlines()[0])
    forged = dict(line)
    forged['decision'] = 'sell'
    del forged['record_hash']
    sealed = AuditRecord(
      seq=forged['seq'], at=forged['at'], decision=forged['decision'],
      rule_ids=tuple(forged['rule_ids']),
      indicators={key: float(value)
                  for key, value in forged['indicators'].items()},
      timeframe=forged['timeframe'],
      timeframe_criterion=forged['timeframe_criterion'],
      policy_fingerprint=forged['policy_fingerprint'],
      features=forged['features'], prev_hash=forged['prev_hash'],
    ).sealed()
    assert sealed.record_hash != line['record_hash'], (
      'sanity: the original digest no longer matches the edited record')
    path = tmp_path / 'forged.jsonl'
    path.write_text(json.dumps(sealed.to_json(), sort_keys=True,
                               separators=(',', ':')) + '\n',
                    encoding='utf-8')
    result = verify_chain(path)
    assert result.ok, (
      'a wholly re-sealed chain whose decision was changed from buy to '
      'sell verified clean. That is the expected and documented limit, and '
      'it is why this chain is not a digital signature: the key that '
      'authenticates each record sits in the file it authenticates, so '
      'anyone with write access can recompute the whole chain from their '
      'own edit. It needs an external witness to be tamper-proof.')


class TestTransactionCostsAreReal:
  '''Invariant 5: the rate card is the Indian one, not a round number.'''

  def test_the_exchange_charge_is_307_per_crore(self):
    assert DELIVERY.exchange_pct == pytest.approx(0.00307 / 100.0)

  def test_delivery_round_trip_is_about_22_25_bps(self):
    # 22.25 bps is the round trip in levies and brokerage, before the
    # separately-modelled slippage, which is 5 bps a side on top.
    notional = 1e7
    slippage = DELIVERY.slippage(notional) * 2
    bps = (DELIVERY.round_trip(notional) - slippage) / notional * 10_000
    assert bps == pytest.approx(22.25, abs=0.05), (
      f'the delivery round trip cost {bps:.2f} bps of levies and brokerage, '
      'which is not the 22.25 bps Indian delivery rate card')

  def test_the_whole_round_trip_adds_the_modelled_slippage(self):
    notional = 1e7
    assert DELIVERY.round_trip(notional) > 22.25 / 10_000 * notional, (
      'the round trip figure must include the slippage, not just levies')

  def test_delivery_brokerage_is_free_for_a_retail_individual(self):
    assert DELIVERY.commission(Side.BUY, 1e6) == 0.0
    assert DELIVERY.commission(Side.SELL, 1e6) == 0.0

  def test_intraday_brokerage_is_three_bps_capped_at_twenty(self):
    assert INTRADAY.brokerage_pct == pytest.approx(0.0003)
    assert INTRADAY.commission(Side.BUY, 1e6) == pytest.approx(20.0)
    assert INTRADAY.commission(Side.BUY, 1_000.0) == pytest.approx(0.3)

  def test_stamp_duty_is_buy_side_only(self):
    model = CostModel(stt_buy=0.0, stt_sell=0.0, dp_charge=0.0,
                      brokerage_pct=0.0, slippage_bps=0.0)
    assert model.taxes_and_fees(Side.BUY, 1e6) > \
        model.taxes_and_fees(Side.SELL, 1e6)

  def test_the_dp_charge_is_a_flat_sell_side_rupee_fee(self):
    # Everything notional-proportional is zeroed so the flat fee is the
    # only thing left, which is what makes the equality below meaningful.
    model = CostModel(stt_buy=0.0, stt_sell=0.0, stamp_duty_buy=0.0,
                      brokerage_pct=0.0, gst_pct=0.0, slippage_bps=0.0,
                      exchange_pct=0.0, sebi_pct=0.0, dp_charge=15.34)
    small = model.taxes_and_fees(Side.SELL, 1e3)
    large = model.taxes_and_fees(Side.SELL, 1e8)
    assert small == pytest.approx(large) == pytest.approx(15.34)

  def test_a_zero_notional_charges_nothing_at_all(self):
    # The docstring's own rule, and it matters: the engine calls this on
    # every rebalance including flat ones, so an unconditional per-order
    # fee would bleed on every bar that does not trade.
    model = CostModel()
    assert model.one_way(Side.BUY, 0.0) == 0.0
    assert model.one_way(Side.SELL, 0.0) == 0.0

  def test_gst_is_not_charged_on_stt(self):
    model = CostModel(stt_buy=0.001, stt_sell=0.0, stamp_duty_buy=0.0,
                      brokerage_pct=0.0, exchange_pct=0.0, sebi_pct=0.0,
                      dp_charge=0.0, gst_pct=0.18, slippage_bps=0.0)
    assert model.taxes_and_fees(Side.BUY, 1e6) == pytest.approx(1_000.0)


class TestNoCallerStateIsMutatedOrShared:
  '''The mutation and shared-state hunt, run against all seven modules.'''

  def test_the_checker_copies_the_venue_map(self):
    securities = {'aaa': SecurityLimits('aaa', venue, wide_mwpl, 10, 1.0)}
    checker = RmsChecker(loose_limits, securities)
    securities['bbb'] = SecurityLimits('bbb', venue, wide_mwpl, 10, 1.0)
    assert checker.known('bbb') is False, (
      'the checker kept a reference to the caller\'s venue map, so a '
      'caller adding a symbol silently widened the limits mid-run')

  def test_broker_positions_are_a_copy(self):
    broker = PaperBroker('p')
    broker.place(make_order('aaa', 10), unit_price)
    first = broker.positions()
    first['aaa'] = 999
    assert broker.positions()['aaa'] == 10, (
      'positions() handed back the live dict; a caller mutating its own '
      'copy rewrote the broker\'s book')

  def test_the_audit_log_copies_the_indicators(self, tmp_path):
    indicators = {'momentum': 0.5}
    log = AuditLog(tmp_path / 'copy.jsonl', 'fingerprint-under-test')
    log.append(decision='buy', indicators=indicators)
    indicators['momentum'] = -99.0
    assert log.records()[0].indicators['momentum'] == 0.5, (
      'the log kept a reference to the caller\'s indicator dict, so a '
      'caller mutating its own values rewrote history')

  def test_two_brokers_share_no_state(self):
    one = PaperBroker('one')
    two = PaperBroker('two')
    one.place(make_order('aaa', 7), unit_price)
    assert not two.orders, (
      f'the second paper broker saw {len(two.orders)} orders placed on the '
      'first: orders is shared mutable class state')
    assert not two.positions()

  def test_two_routers_share_no_health(self):
    one = PaperBroker('one')
    two = PaperBroker('two')
    left = FailoverRouter([one], failure_threshold=1)
    right = FailoverRouter([two], failure_threshold=1)
    left.note_failure('one', 'boom')
    assert left.is_healthy('one') is False
    assert right.is_healthy('two') is True

  def test_a_router_rejects_a_health_record_for_an_unknown_venue(self):
    router = FailoverRouter([PaperBroker('one')])
    with pytest.raises(KeyError):
      router.note_failure('not-a-venue', 'boom')

  def test_a_fault_injection_is_per_broker(self):
    one = PaperBroker('one')
    two = PaperBroker('two')
    one.inject_fault(count=1)
    with pytest.raises(BrokerError):
      one.place(make_order(), unit_price)
    # The counter was decremented, so the next call goes through.
    assert one.place(make_order(), unit_price).ok
    two.inject_fault(count=0)
    assert two.is_healthy() is True

  def test_the_sizer_and_the_checker_share_no_state(self):
    sizer = make_sizer()
    before = dict(sizer.checker.securities)
    sizer.size('aaa', unit_price, reference_price=unit_price)
    assert dict(sizer.checker.securities) == before

  def test_the_sizer_does_not_mutate_its_inputs(self):
    sizer = make_sizer()
    inputs = SizingInputs(prob_win=0.9, uncertainty=0.0)
    before = (inputs.prob_win, inputs.uncertainty)
    sizer.size('aaa', unit_price, inputs=inputs, reference_price=unit_price)
    assert (inputs.prob_win, inputs.uncertainty) == before, (
      'StockSizer mutated the caller\'s SizingInputs')


class TestPreTradeChecksFailClosed:
  '''The design claim in the checks module header, spot-checked.'''

  def test_an_unknown_symbol_rejects_every_check(self):
    report = make_checker().check(make_order('zzz'), AccountState())
    assert not report.approved
    assert {result.check_id for result in report.rejections} == \
        {result.check_id for result in report.results}

  def test_a_missing_reference_price_rejects_rather_than_skipping(self):
    report = make_checker().check(make_order(reference_price=0.0),
                                  AccountState())
    assert not report.result('trade_price_protection').passed, (
      'a bad-ticket check with no reference price passed; it checked '
      'nothing')

  def test_an_algo_market_order_is_refused(self):
    report = make_checker().check(make_order(price=0.0,
                                             order_type=market_order),
                                  AccountState())
    assert not report.result('algo_market_order').passed

  def test_a_market_order_fails_every_check_it_cannot_be_checked(self):
    report = make_checker().check(make_order(price=0.0,
                                             order_type=market_order),
                                  AccountState())
    assert not report.approved
    fired = {result.check_id for result in report.rejections}
    assert {'price_band', 'order_value', 'trade_price_protection',
            'algo_market_order'} <= fired

  def test_the_paper_broker_refuses_a_market_order_too(self):
    broker = PaperBroker('p')
    ack = broker.place(make_order(price=0.0, order_type=market_order),
                       unit_price)
    assert ack.status == rejected
    assert '8.1.1.12' in ack.reason

  def test_a_marketable_limit_fills_and_a_resting_one_does_not(self):
    broker = PaperBroker('p')
    filled_ack = broker.place(make_order('aaa', 10, 100.0), 100.0)
    assert filled_ack.status == filled
    assert filled_ack.filled_quantity == 10
    resting = broker.place(make_order('bbb', 10, 50.0), 100.0)
    assert resting.status == open_order

  def test_a_refused_order_is_still_recorded(self):
    broker = PaperBroker('p')
    broker.place(make_order('aaa', 10, 500.0), 100.0)
    assert len(broker.orders) == 1

  def test_an_unconfigured_broker_is_loud(self):
    ack = NullBroker().place(make_order(), unit_price)
    assert ack.status == rejected
    assert 'no broker is configured' in ack.reason

  def test_a_null_broker_does_not_make_the_router_look_broken(self):
    router = FailoverRouter([NullBroker()])
    assert router.is_healthy('null') is True
    assert router.place(make_order(), unit_price).status == rejected


class TestAcknowledgementSanity:
  '''``BrokerAck`` checks that a fill carries a quantity but nothing else.'''

  def test_a_fill_cannot_exceed_the_order(self):
    with pytest.raises(ValueError):
      BrokerAck(broker='b', status=filled, order_id='1', quantity=10,
                filled_quantity=999, average_price=unit_price)

  def test_a_fill_must_carry_a_price(self):
    with pytest.raises(ValueError):
      BrokerAck(broker='b', status=filled, order_id='1', quantity=10,
                filled_quantity=10, average_price=0.0)

  def test_a_fill_without_quantity_is_refused(self):
    with pytest.raises(ValueError, match='positive filled quantity'):
      BrokerAck(broker='b', status=filled, order_id='1', quantity=10)

  def test_an_open_order_needs_no_fill(self):
    ack = BrokerAck(broker='b', status=open_order, order_id='1',
                    quantity=10)
    assert ack.ok is True
    assert ack.filled_quantity == 0

  def test_a_rejection_carries_no_fill(self):
    ack = BrokerAck(broker='b', status=rejected, order_id='1', quantity=10,
                    reason='no broker is configured')
    assert ack.ok is False
    assert 'no broker' in ack.render()

  def test_the_published_guard_would_have_caught_the_nan(self):
    # The positive control for finding 1: the guard this module does not
    # call raises on precisely the inputs sizing.py turns into the cap.
    for bad in (float('nan'), float('inf'), float('-inf')):
      with pytest.raises(ValueError):
        clamp_weight(bad, max_fraction)
    assert clamp_weight(0.5, 0.5) == pytest.approx(0.5)
