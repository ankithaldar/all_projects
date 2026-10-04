#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Tests for sizing, the broker seam and the audit chain.

Three properties are asserted here that a reviewer cannot establish by
reading the code:

  * **the audit hash chain detects an edited earlier record.** A record
    is appended, an *earlier* one is rewritten in the file by hand, and
    verification is asked for. It must fail, and it must name the
    position. A chain that only detects an edit to the last record is
    not a chain.
  * **the sizer refuses to breach a risk cap.** A size is computed that
    the pre-trade checks would reject, and the exception has to arrive
    rather than a shrunk order.
  * **the kill-switch and audit clocks are injected.** No test sleeps,
    and no wall-clock read makes a log non-reproducible.

There is **no network, no broker credential and no vendor SDK** anywhere
in this file, and no test could reach one: the only brokers here are
:class:`~stock_rl.execution.broker.PaperBroker`,
:class:`~stock_rl.execution.broker.NullBroker` and an in-process
:class:`~stock_rl.execution.broker.FailoverRouter`.

Failure injection is deterministic (:meth:`PaperBroker.inject_fault`)
rather than random, so a failover test asserts a sequence of
acknowledgements rather than a distribution over one.
'''

import json
from datetime import datetime, timezone

import pytest

from stock_rl.execution.audit import (
  AuditChainError,
  AuditLog,
  AuditRecord,
  ChainVerification,
  audit_schema,
  decision_log_class,
  feature_hash,
  genesis_hash,
  utcnow as audit_utcnow,
  verify_chain,
)
from stock_rl.execution.broker import (
  Broker,
  BrokerAck,
  BrokerError,
  FailoverRouter,
  NullBroker,
  PaperBroker,
  cancelled,
  filled,
  open_order,
  order_statuses,
  rejected,
)
from stock_rl.execution.sizing import (
  RiskCapBreach,
  SizingInputs,
  SizingResult,
  StockSizer,
  kelly_fraction,
  kelly_method,
  max_fraction,
  uncertainty_adjusted_size,
  vol_target_fraction,
  vol_target_method,
)
from stock_rl.risk.checks import (
  OrderRequest,
  PriceBand,
  RiskRejected,
  RmsChecker,
  RmsLimits,
  SecurityLimits,
  buy,
  limit_order,
  market_order,
  sell,
)
from stock_rl.rl.policy import ConstantWeights, policy_fingerprint

#: A fixed clock for the audit log, so a written file is byte-identical
#: between runs and the assertion can be on the bytes.
EPOCH = datetime(2026, 3, 2, 9, 15, tzinfo=timezone.utc)

#: A stand-in for the policy fingerprint the log records. Tests that care
#: about the fingerprint's provenance build it from a real policy.
FINGERPRINT = policy_fingerprint(ConstantWeights(weight=0.07))


def fixed_clock():
  '''Return a wall-clock source that always reports the same instant.

  Returns:
    A callable suitable for ``AuditLog(clock=...)``.
  '''
  return lambda: EPOCH


def make_checker(max_order_value=1_000_000.0,
                max_order_quantity=5_000) -> RmsChecker:
  '''Return a single-scrip checker with tight per-order limits.

  Args:
    max_order_value: Per-order rupee limit, in rupees.
    max_order_quantity: Per-order share limit.

  Returns:
    An :class:`RmsChecker`.
  '''
  security = SecurityLimits(
    symbol='RELIANCE',
    band=PriceBand(2_900.0, 3_200.0),
    mwpl=PriceBand(2_000.0, 4_000.0),
    max_order_quantity=max_order_quantity,
    max_order_value=max_order_value,
  )
  limits = RmsLimits(
    cumulative_open_order_value=5_000_000.0,
    max_position=100_000,
    max_trading_value=10_000_000.0,
    max_exposure=10_000_000.0,
    max_turnover=20_000_000.0,
    max_security_value=2_000_000.0,
  )
  return RmsChecker(limits, {'RELIANCE': security})


def make_order(**overrides) -> OrderRequest:
  '''Return a small order that clears every pre-trade check.

  Args:
    overrides: Fields to replace on the defaults.

  Returns:
    An :class:`OrderRequest`.
  '''
  values = {
    'symbol': 'RELIANCE',
    'side': buy,
    'quantity': 10,
    'order_type': limit_order,
    'price': 3_000.0,
    'reference_price': 3_000.0,
  }
  values.update(overrides)
  return OrderRequest(**values)


def make_sizer(capital=1_000_000.0, checker=None, **kwargs) -> StockSizer:
  '''Return a sizer over a single-scrip checker.

  The default capital is one million rather than ten, because at the
  default cap of a quarter ten million wants 2.5 million of exposure
  against a per-order limit of one million. Every happy-path sizing test
  would then be testing the caps rather than the sizing.

  Args:
    capital: Capital base in rupees.
    checker: Checker to use, or None for a default one.
    kwargs: Extra arguments for :class:`StockSizer`.

  Returns:
    A :class:`StockSizer`.
  '''
  return StockSizer(checker or make_checker(), capital, **kwargs)


def make_log(path, fingerprint=FINGERPRINT) -> AuditLog:
  '''Return an audit log over a temporary file.

  Args:
    path: Path to the JSONL file.
    fingerprint: Policy fingerprint to record.

  Returns:
    An :class:`AuditLog` on the injected clock.
  '''
  return AuditLog(path, fingerprint, clock=fixed_clock())


def rewrite_line(path, index, change):
  '''Rewrite one line of a JSONL log in place, as an editor would.

  Args:
    path: Path to the log.
    index: Zero-based line index.
    change: Callable taking and returning the parsed payload.

  Returns:
    The payload as written, so a test can assert what it forged.
  '''
  lines = path.read_text(encoding='utf-8').splitlines()
  payload = json.loads(lines[index])
  forged = change(payload)
  lines[index] = json.dumps(forged, sort_keys=True, separators=(',', ':'))
  path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
  return forged


class TestKellySizing:
  '''The Kelly rule and its uncertainty haircut.'''

  def test_the_kelly_fraction_is_edge_over_odds(self):
    # p = 0.6, b = 2: edge = 0.6*2 - 0.4 = 0.8, over odds of 2.
    assert kelly_fraction(0.6, 2.0) == pytest.approx(0.4)

  def test_kelly_is_zero_at_the_break_even_probability(self):
    assert kelly_fraction(0.5, 1.0) == pytest.approx(0.0)

  def test_a_negative_edge_returns_a_negative_fraction(self):
    # Returned rather than clamped: the caller needs to see the sign, and
    # the only actionable reading of "negative size" is "do not trade".
    assert kelly_fraction(0.4, 1.0) < 0.0

  def test_a_zero_uncertainty_adjusted_edge_sizes_nothing(self):
    # The load-bearing case. prob_win of 0.5 on an even payoff is the
    # break-even point, so the uncertainty-adjusted size must be exactly
    # zero and not "a small positive number".
    assert uncertainty_adjusted_size(0.5, 1.0, 0.5) == 0.0

  def test_a_negative_edge_is_floored_at_zero_not_traded_short(self):
    assert uncertainty_adjusted_size(0.4, 1.0, 0.5) == 0.0

  def test_the_haircut_scales_the_fraction_linearly(self):
    full = uncertainty_adjusted_size(0.6, 1.0, 0.0)
    half = uncertainty_adjusted_size(0.6, 1.0, 0.5)
    assert half == pytest.approx(full * 0.5)

  def test_the_size_is_capped(self):
    # A 0.8 full-Kelly edge is refused down to max_fraction rather than
    # handed to the RMS to reject.
    assert uncertainty_adjusted_size(0.9, 1.0, 0.0) == max_fraction

  def test_an_uncertainty_of_one_is_not_a_configuration(self):
    # Excluded from the range: a size of exactly zero is the no-trade
    # answer and is spelled as zero fraction, not as uncertainty 1.0.
    with pytest.raises(ValueError, match=r'uncertainty must be in'):
      uncertainty_adjusted_size(0.6, 1.0, 1.0)

  def test_a_probability_outside_the_unit_interval_is_refused(self):
    with pytest.raises(ValueError, match='prob_win must be in'):
      kelly_fraction(1.5, 1.0)

  def test_a_non_positive_payoff_ratio_is_refused(self):
    with pytest.raises(ValueError, match='win_loss_ratio must be positive'):
      kelly_fraction(0.6, 0.0)

  def test_the_size_never_exceeds_one(self):
    assert max_fraction <= 1.0


class TestVolTargetSizing:
  '''Volatility targeting, the rule with no fitted parameter.'''

  def test_the_size_is_the_volatility_ratio(self):
    # An explicit cap, because the default cap is the project's risk
    # appetite and would mask the ratio being asserted.
    assert vol_target_fraction(0.20, 0.10, cap=1.0) == pytest.approx(0.5)

  def test_a_calmer_asset_is_sized_larger(self):
    assert vol_target_fraction(0.10, 0.15, cap=1.0) > \
      vol_target_fraction(0.40, 0.15, cap=1.0)

  def test_the_default_cap_is_the_project_max_fraction(self):
    assert vol_target_fraction(0.01, 0.20) == max_fraction

  def test_the_size_is_capped(self):
    assert vol_target_fraction(0.01, 0.20, cap=0.30) == 0.30

  def test_a_zero_volatility_input_is_refused_not_divided_by(self):
    # target / 0 would be the largest possible size, returned with no
    # error and no evidence behind it.
    with pytest.raises(ValueError, match='must be positive'):
      vol_target_fraction(0.0, 0.15)

  def test_a_non_positive_target_is_refused(self):
    with pytest.raises(ValueError, match='target_vol must be positive'):
      vol_target_fraction(0.20, 0.0)

  def test_a_cap_outside_the_unit_interval_is_refused(self):
    with pytest.raises(ValueError, match='cap must be in'):
      vol_target_fraction(0.20, 0.15, cap=1.5)


class TestSizingInputs:
  '''The two rules behind one input record.'''

  def test_the_kelly_method_uses_the_haircut(self):
    inputs = SizingInputs(prob_win=0.6, uncertainty=0.5)
    assert inputs.fraction(kelly_method) == pytest.approx(
      uncertainty_adjusted_size(0.6, 1.0, 0.5))

  def test_the_vol_target_method_ignores_the_edge(self):
    edged = SizingInputs(prob_win=0.9, uncertainty=0.0, realised_vol=0.20)
    flat = SizingInputs(prob_win=0.1, uncertainty=0.9, realised_vol=0.20)
    assert edged.fraction(vol_target_method) == \
      flat.fraction(vol_target_method)

  def test_an_unknown_method_is_refused(self):
    with pytest.raises(ValueError, match='method must be one of'):
      SizingInputs().fraction('markowitz')

  def test_the_vol_target_method_needs_a_realised_volatility(self):
    with pytest.raises(ValueError, match='realised_vol must be positive'):
      SizingInputs().fraction(vol_target_method)


class TestStockSizer:
  '''Sizing, and the refusal to size past a cap.'''

  def test_a_zero_edge_produces_no_order_at_all(self):
    # Zero size is not an order: a zero-quantity order would be rejected
    # by the quantity check, which is right and pointless.
    sizer = make_sizer()
    result = sizer.size('RELIANCE', 3_000.0,
                        inputs=SizingInputs(prob_win=0.5, uncertainty=0.5))
    assert result.should_trade is False
    assert result.order is None
    assert result.quantity == 0
    assert result.report is None

  def test_a_zero_size_says_so_in_its_render(self):
    sizer = make_sizer()
    text = sizer.size('RELIANCE', 3_000.0,
                      inputs=SizingInputs(prob_win=0.5)).render()
    assert 'no order' in text

  def test_a_real_edge_produces_a_cleared_order(self):
    sizer = make_sizer()
    result = sizer.size('RELIANCE', 3_000.0, reference_price=3_000.0,
                        inputs=SizingInputs(prob_win=0.6, uncertainty=0.2))
    assert result.should_trade is True
    assert result.report.approved is True
    assert result.quantity > 0
    assert result.notional == pytest.approx(result.quantity * 3_000.0)

  def test_the_sized_order_carries_the_whole_decision(self):
    sizer = make_sizer()
    result = sizer.size('RELIANCE', 3_000.0, reference_price=3_000.0,
                        algo_id='99999', order_id='A-1',
                        inputs=SizingInputs(prob_win=0.9, uncertainty=0.0))
    assert result.order.algo_id == '99999'
    assert result.order.order_id == 'A-1'
    assert result.order.reference_price == 3_000.0

  def test_the_sizer_refuses_to_breach_the_per_order_value_cap(self):
    # Ten million of capital at the quarter cap wants 2.5 million of
    # exposure against a 100_000 rupee per-order limit. The sizer must
    # refuse, not shrink.
    sizer = make_sizer(capital=10_000_000.0,
                       checker=make_checker(max_order_value=100_000.0))
    with pytest.raises(RiskCapBreach) as caught:
      sizer.size('RELIANCE', 3_000.0,
                 inputs=SizingInputs(prob_win=0.9, uncertainty=0.0))
    assert 'order_value' in {r.check_id for r in caught.value.report.rejections}

  def test_the_sizer_refuses_to_breach_the_per_order_quantity_cap(self):
    sizer = make_sizer(checker=make_checker(max_order_quantity=5))
    with pytest.raises(RiskCapBreach):
      sizer.size('RELIANCE', 3_000.0,
                 inputs=SizingInputs(prob_win=0.9, uncertainty=0.0))

  def test_a_risk_cap_breach_is_a_risk_rejection(self):
    # So a caller catching the RMS type catches the sizer too.
    sizer = make_sizer(checker=make_checker(max_order_value=100_000.0))
    with pytest.raises(RiskRejected):
      sizer.size('RELIANCE', 3_000.0,
                 inputs=SizingInputs(prob_win=0.9, uncertainty=0.0))

  def test_a_cap_breach_does_not_produce_an_order(self):
    sizer = make_sizer(checker=make_checker(max_order_value=100_000.0))
    with pytest.raises(RiskCapBreach):
      sizer.size('RELIANCE', 3_000.0,
                 inputs=SizingInputs(prob_win=0.9, uncertainty=0.0))

  def test_an_unknown_symbol_is_refused_by_the_sizer_too(self):
    sizer = make_sizer()
    with pytest.raises(RiskCapBreach):
      sizer.size('UNKNOWN', 3_000.0,
                 inputs=SizingInputs(prob_win=0.9, uncertainty=0.0))

  def test_a_missing_reference_price_is_refused_rather_than_skipped(
      self):
    # The sizer passes reference_price straight through, and the
    # bad-ticket check rejects a zero one. Sizing must not paper over it.
    sizer = make_sizer()
    with pytest.raises(RiskCapBreach):
      sizer.size('RELIANCE', 3_000.0,
                 inputs=SizingInputs(prob_win=0.9, uncertainty=0.0))

  def test_a_supplied_reference_price_clears_the_order(self):
    sizer = make_sizer()
    result = sizer.size('RELIANCE', 3_000.0, reference_price=3_000.0,
                        inputs=SizingInputs(prob_win=0.9, uncertainty=0.0))
    assert result.report.approved is True

  def test_the_size_is_rounded_down_to_a_whole_lot(self):
    sizer = make_sizer(capital=1_000_000.0, lot_size=50)
    result = sizer.size('RELIANCE', 3_000.0, reference_price=3_000.0,
                        inputs=SizingInputs(prob_win=0.9, uncertainty=0.0))
    assert result.quantity % 50 == 0

  def test_rounding_down_never_pushes_past_a_limit(self):
    # 540_000 at the quarter cap is 135_000, which is 45 shares at 3_000.
    # Rounded to a lot of 10 that is 40, inside the 41-share cap. Rounded
    # to *nearest* the 45 would have been kept and blown the cap, and
    # rounded up it would have been 50.
    sizer = make_sizer(capital=540_000.0, lot_size=10,
                       checker=make_checker(max_order_quantity=41,
                                            max_order_value=5_000_000.0))
    result = sizer.size('RELIANCE', 3_000.0, reference_price=3_000.0,
                        inputs=SizingInputs(prob_win=0.9, uncertainty=0.0))
    assert result.quantity == 40

  def test_a_size_below_one_lot_trades_nothing(self):
    sizer = make_sizer(capital=1_000.0, lot_size=100)
    result = sizer.size('RELIANCE', 3_000.0, reference_price=3_000.0,
                        inputs=SizingInputs(prob_win=0.9, uncertainty=0.0))
    assert result.should_trade is False

  def test_the_vol_target_method_also_clears_through_the_rms(self):
    sizer = make_sizer()
    result = sizer.size('RELIANCE', 3_000.0, reference_price=3_000.0,
                        inputs=SizingInputs(realised_vol=0.30,
                                            target_vol=0.15),
                        method=vol_target_method)
    # The 0.5 ratio is above the project's cap, so the cap is what the
    # sizer commits.
    assert result.fraction == max_fraction
    assert result.report.approved is True

  def test_a_non_positive_price_is_refused(self):
    with pytest.raises(ValueError, match='price must be positive'):
      make_sizer().size('RELIANCE', 0.0)

  def test_a_non_finite_capital_is_refused(self):
    with pytest.raises(ValueError, match='capital must be finite'):
      make_sizer(capital=float('inf'))

  def test_a_zero_lot_size_is_refused(self):
    with pytest.raises(ValueError, match='lot_size must be'):
      make_sizer(lot_size=0)

  def test_a_cap_above_one_is_refused(self):
    with pytest.raises(ValueError, match='cap must be in'):
      make_sizer(cap=1.5)

  def test_the_result_carries_the_side(self):
    result = make_sizer().size('RELIANCE', 3_000.0, side=sell,
                               reference_price=3_000.0,
                               inputs=SizingInputs(prob_win=0.9))
    assert result.side == sell
    assert result.order.side == sell

  def test_a_result_is_frozen_so_it_cannot_be_edited_after_the_fact(self):
    result = make_sizer().size('RELIANCE', 3_000.0,
                               inputs=SizingInputs(prob_win=0.5))
    with pytest.raises(AttributeError):
      result.quantity = 99

  def test_an_empty_result_is_still_a_result(self):
    assert SizingResult('X', buy, 0.0, 0, 0.0).should_trade is False


class TestPaperBroker:
  '''The in-memory venue. No network, no randomness.'''

  def test_a_marketable_buy_fills_at_the_market_price(self):
    broker = PaperBroker()
    ack = broker.place(make_order(price=3_010.0), 3_000.0)
    assert ack.status == filled
    assert ack.filled_quantity == 10
    assert ack.average_price == 3_000.0
    assert ack.ok is True

  def test_a_marketable_sell_fills_at_the_market_price(self):
    broker = PaperBroker()
    ack = broker.place(make_order(side=sell, price=2_990.0), 3_000.0)
    assert ack.status == filled
    assert broker.positions()['RELIANCE'] == -10

  def test_a_resting_limit_order_stays_open(self):
    broker = PaperBroker()
    ack = broker.place(make_order(price=2_900.0), 3_000.0)
    assert ack.status == open_order
    assert not broker.positions()

  def test_the_broker_refuses_a_market_order(self):
    # Defence in depth on NSE/MSD/67753 8.1.1.12: a caller that
    # bypasses the risk layer still cannot send one.
    broker = PaperBroker()
    ack = broker.place(make_order(order_type=market_order, price=0.0),
                       3_000.0)
    assert ack.status == rejected
    assert 'NSE/MSD/67753' in ack.reason

  def test_a_refusal_is_still_recorded(self):
    broker = PaperBroker()
    broker.place(make_order(order_type=market_order, price=0.0), 3_000.0)
    assert len(broker.orders) == 1

  def test_a_live_order_can_be_cancelled(self):
    broker = PaperBroker()
    placed = broker.place(make_order(price=2_900.0), 3_000.0)
    ack = broker.cancel(placed.order_id)
    assert ack.status == cancelled

  def test_cancelling_twice_is_refused(self):
    broker = PaperBroker()
    placed = broker.place(make_order(price=2_900.0), 3_000.0)
    broker.cancel(placed.order_id)
    assert broker.cancel(placed.order_id).status == rejected

  def test_cancelling_an_unknown_id_is_refused(self):
    assert PaperBroker().cancel('nope').status == rejected

  def test_a_cancel_appends_rather_than_edits_the_order_log(self):
    broker = PaperBroker()
    placed = broker.place(make_order(price=2_900.0), 3_000.0)
    broker.cancel(placed.order_id)
    # The original record still says "open", because an order log that is
    # edited in place cannot answer what the order looked like when it
    # was sent.
    assert broker.orders[0].ack.status == open_order
    assert broker.orders[1].ack.status == cancelled

  def test_a_non_positive_market_price_is_refused(self):
    with pytest.raises(ValueError, match='market_price must be positive'):
      PaperBroker().place(make_order(), 0.0)

  def test_a_blank_broker_name_is_refused(self):
    with pytest.raises(ValueError, match='name must not be blank'):
      PaperBroker('  ')

  def test_an_injected_fault_raises_once_and_then_clears(self):
    broker = PaperBroker()
    broker.inject_fault(1)
    with pytest.raises(BrokerError):
      broker.place(make_order(), 3_000.0)
    assert broker.place(make_order(), 3_000.0).status == filled

  def test_an_unhealthy_paper_broker_reports_it(self):
    broker = PaperBroker('degraded', healthy=False)
    assert broker.is_healthy() is False

  def test_the_paper_broker_satisfies_the_protocol(self):
    assert isinstance(PaperBroker(), Broker)

  def test_the_render_names_the_fill(self):
    broker = PaperBroker()
    text = broker.place(make_order(price=3_010.0), 3_000.0).render()
    assert 'filled' in text


class TestNullBroker:
  '''No broker configured must be loud, not an AttributeError.'''

  def test_an_order_is_refused_with_a_reason(self):
    ack = NullBroker().place(make_order(), 3_000.0)
    assert ack.status == rejected
    assert 'no broker is configured' in ack.reason

  def test_the_refusal_names_the_absence_of_a_network_path(self):
    ack = NullBroker().place(make_order(), 3_000.0)
    assert 'no credentials' in ack.reason

  def test_a_cancellation_is_refused(self):
    assert NullBroker().cancel('anything').status == rejected

  def test_it_holds_no_positions(self):
    assert not NullBroker().positions()

  def test_it_reports_healthy_because_it_is_working_as_designed(self):
    # Marking it unhealthy would make a failover router skip it and hide
    # the real problem: it is not broken, it is refusing.
    assert NullBroker().is_healthy() is True

  def test_it_satisfies_the_protocol(self):
    assert isinstance(NullBroker(), Broker)

  def test_a_blank_name_is_refused(self):
    with pytest.raises(ValueError, match='name must not be blank'):
      NullBroker('')


class TestBrokerAck:
  '''The acknowledgement every venue must be able to produce.'''

  def test_an_unrecognised_status_is_refused(self):
    with pytest.raises(ValueError, match='status must be one of'):
      BrokerAck('paper', 'pending')

  def test_a_blank_broker_name_is_refused(self):
    with pytest.raises(ValueError, match='broker name must not be blank'):
      BrokerAck('  ', filled)

  def test_a_fill_with_no_quantity_is_refused(self):
    # The failure a paper broker can hide: an ack claiming a fill with
    # zero quantity reads as a fill downstream and books no position.
    with pytest.raises(ValueError, match='positive filled quantity'):
      BrokerAck('paper', filled, order_id='x', filled_quantity=0)

  def test_every_declared_status_constructs(self):
    for status in order_statuses:
      # A filled ack must carry a quantity, so the payload is status-aware
      # rather than uniformly empty.
      ack = BrokerAck('paper', status,
                      filled_quantity=1 if status == filled else 0)
      assert ack.status == status

  def test_a_rejection_renders_its_reason(self):
    ack = BrokerAck('paper', rejected, reason='because')
    assert 'because' in ack.render()

  def test_an_ack_is_frozen(self):
    ack = BrokerAck('paper', rejected)
    with pytest.raises(AttributeError):
      ack.status = filled


class TestFailoverRouter:
  '''Multi-broker failover, entirely in process.'''

  def test_the_first_broker_is_used_when_it_works(self):
    first = PaperBroker('first')
    second = PaperBroker('second')
    router = FailoverRouter([first, second])
    ack = router.place(make_order(price=3_010.0), 3_000.0)
    assert ack.broker == 'first'

  def test_a_failing_venue_fails_over_to_the_next(self):
    first = PaperBroker('first')
    first.inject_fault(1)
    second = PaperBroker('second')
    router = FailoverRouter([first, second])
    ack = router.place(make_order(price=3_010.0), 3_000.0)
    assert ack.broker == 'second'
    assert ack.status == filled

  def test_a_venue_is_marked_unhealthy_after_the_threshold(self):
    first = PaperBroker('first')
    first.inject_fault(2)
    router = FailoverRouter([first, PaperBroker('second')],
                            failure_threshold=2)
    router.place(make_order(price=3_010.0), 3_000.0)
    assert router.is_healthy('first') is True
    router.place(make_order(price=3_010.0), 3_000.0)
    assert router.is_healthy('first') is False

  def test_an_unhealthy_venue_is_not_tried_again(self):
    first = PaperBroker('first')
    second = PaperBroker('second')
    first.inject_fault(4)
    router = FailoverRouter([first, second], failure_threshold=2)
    for _ in range(3):
      router.place(make_order(price=3_010.0), 3_000.0)
    # The dead venue recorded nothing after its first two failures, so
    # the third order never went back to it.
    assert not first.orders
    assert len(second.orders) == 3
    assert router.positions()['RELIANCE'] == 30

  def test_a_success_clears_the_failure_streak(self):
    broker = PaperBroker('only')
    router = FailoverRouter([broker], failure_threshold=2)
    broker.inject_fault(1)
    router.place(make_order(price=3_010.0), 3_000.0)
    assert router.health()[0].failures == 1
    router.place(make_order(price=3_010.0), 3_000.0)
    assert router.health()[0].failures == 0

  def test_all_venues_down_returns_a_rejection_not_an_exception(self):
    first = PaperBroker('first')
    second = PaperBroker('second')
    first.inject_fault(5)
    second.inject_fault(5)
    router = FailoverRouter([first, second])
    ack = router.place(make_order(price=3_010.0), 3_000.0)
    assert ack.status == rejected
    assert 'no broker accepted' in ack.reason

  def test_the_rejection_names_every_venue_tried(self):
    first = PaperBroker('first')
    second = PaperBroker('second')
    first.inject_fault(5)
    second.inject_fault(5)
    router = FailoverRouter([first, second])
    ack = router.place(make_order(price=3_010.0), 3_000.0)
    assert 'first' in ack.reason
    assert 'second' in ack.reason

  def test_positions_are_the_union_of_the_venues(self):
    # After a failover the first venue legitimately holds nothing and the
    # second holds the book; returning the first non-empty mapping would
    # report an empty account at exactly that moment.
    first = PaperBroker('first')
    second = PaperBroker('second')
    router = FailoverRouter([first, second])
    second.place(make_order(price=3_010.0), 3_000.0)
    assert router.positions()['RELIANCE'] == 10

  def test_health_reports_the_failure_count_and_reason(self):
    broker = PaperBroker('only')
    router = FailoverRouter([broker])
    broker.inject_fault(1, 'socket timeout')
    router.place(make_order(price=3_010.0), 3_000.0)
    entry = router.health()[0]
    assert entry.failures == 1
    assert 'socket timeout' in entry.last_error

  def test_health_renders_for_a_log(self):
    router = FailoverRouter([PaperBroker('only')])
    assert 'healthy' in router.render()

  def test_a_cancel_fails_over_too(self):
    first = PaperBroker('first')
    placed = first.place(make_order(price=2_900.0), 3_000.0)
    first.inject_fault(1)
    router = FailoverRouter([first, PaperBroker('second')])
    ack = router.cancel(placed.order_id)
    assert ack.broker == 'second'
    assert ack.status == rejected

  def test_an_empty_rotation_is_refused(self):
    with pytest.raises(ValueError, match='at least one broker'):
      FailoverRouter([])

  def test_duplicate_names_are_refused(self):
    # Health is tracked by name, so two venues sharing one would make a
    # failure mark the wrong one.
    with pytest.raises(ValueError, match='must be unique'):
      FailoverRouter([PaperBroker('same'), PaperBroker('same')])

  def test_a_zero_failure_threshold_is_refused(self):
    with pytest.raises(ValueError, match='failure_threshold must be'):
      FailoverRouter([PaperBroker('only')], failure_threshold=0)

  def test_recording_a_failure_for_an_unknown_venue_raises(self):
    router = FailoverRouter([PaperBroker('only')])
    with pytest.raises(KeyError):
      router.note_failure('stranger', 'boom')

  def test_recording_a_success_for_an_unknown_venue_raises(self):
    router = FailoverRouter([PaperBroker('only')])
    with pytest.raises(KeyError):
      router.note_success('stranger')

  def test_a_null_broker_rejects_rather_than_being_skipped(self):
    router = FailoverRouter([NullBroker(), PaperBroker('paper')])
    ack = router.place(make_order(price=3_010.0), 3_000.0)
    assert ack.status == rejected
    assert ack.broker == 'null'


class TestAuditRecordShape:
  '''What a decision record has to carry, and what it must refuse.'''

  def test_the_log_records_the_policy_fingerprint(self, tmp_path):
    log = make_log(tmp_path / 'audit-shape.jsonl')
    record = log.append('buy', rule_ids=('rsi',), indicators={'rsi': 28.0},
                        timeframe='daily')
    assert record.policy_fingerprint == FINGERPRINT

  def test_the_fingerprint_is_a_real_policy_fingerprint(self, tmp_path):
    # Discharging NSE 9.1/9.9 means the identity of the *code*, so the
    # log must carry the same string the registration record would.
    log = make_log(tmp_path / 'audit-fp.jsonl')
    assert log.append('buy').policy_fingerprint == policy_fingerprint(
      ConstantWeights(weight=0.07))

  def test_a_blank_fingerprint_is_refused(self, tmp_path):
    with pytest.raises(ValueError, match='policy_fingerprint must be'):
      AuditLog(tmp_path / 'audit-blank.jsonl', '   ')

  def test_a_blank_decision_is_refused(self, tmp_path):
    log = make_log(tmp_path / 'audit-blank2.jsonl')
    with pytest.raises(ValueError, match='decision must be a non-blank'):
      log.append('  ')

  def test_the_record_carries_the_indicators_the_rules_saw(self, tmp_path):
    log = make_log(tmp_path / 'audit-ind.jsonl')
    record = log.append('sell', indicators={'rsi': 71.5, 'ema_gap': -0.02})
    assert dict(record.indicators) == {'rsi': 71.5, 'ema_gap': -0.02}

  def test_the_record_carries_the_rule_ids_and_the_timeframe(self, tmp_path):
    log = make_log(tmp_path / 'audit-rule.jsonl')
    record = log.append('hold', rule_ids=('rsi>70', 'price<ema200'),
                        timeframe='weekly',
                        timeframe_criterion='weekly momentum regime')
    assert record.rule_ids == ('rsi>70', 'price<ema200')
    assert record.timeframe == 'weekly'
    assert record.timeframe_criterion == 'weekly momentum regime'

  def test_the_first_record_chains_onto_the_genesis_hash(self, tmp_path):
    log = make_log(tmp_path / 'audit-gen.jsonl')
    assert log.append('buy').prev_hash == genesis_hash

  def test_the_second_record_chains_onto_the_first(self, tmp_path):
    log = make_log(tmp_path / 'audit-chain.jsonl')
    first = log.append('buy')
    second = log.append('sell')
    assert second.prev_hash == first.record_hash
    assert second.seq == 2

  def test_supplying_both_feature_forms_is_refused(self, tmp_path):
    log = make_log(tmp_path / 'audit-both.jsonl')
    with pytest.raises(ValueError, match='not both'):
      log.append('buy', features={'x': 1.0}, feature_digest='deadbeef')

  def test_a_pre_computed_feature_digest_is_accepted(self, tmp_path):
    log = make_log(tmp_path / 'audit-digest.jsonl')
    record = log.append('buy', feature_digest=feature_hash({'x': 1.0}))
    assert record.features == feature_hash({'x': 1.0})

  def test_an_unknown_record_class_is_refused(self, tmp_path):
    with pytest.raises(KeyError):
      AuditLog(tmp_path / 'audit-class.jsonl', FINGERPRINT,
               record_class='no_such_class')


class TestFeatureHash:
  '''The feature-vector digest, and what it must not swallow.'''

  def test_the_hash_does_not_depend_on_key_order(self):
    assert feature_hash({'a': 1.0, 'b': 2.0}) == feature_hash(
      {'b': 2.0, 'a': 1.0})

  def test_a_different_value_gives_a_different_hash(self):
    assert feature_hash({'a': 1.0}) != feature_hash({'a': 1.0000000001})

  def test_a_nan_feature_is_refused(self):
    # A NaN would hash to the same digest whatever it meant, so two
    # genuinely different states could share a fingerprint.
    with pytest.raises(ValueError, match='not finite'):
      feature_hash({'a': float('nan')})

  def test_an_infinite_feature_is_refused(self):
    with pytest.raises(ValueError, match='not finite'):
      feature_hash({'a': float('inf')})


class TestAuditChain:
  '''The hash chain, which is the whole point of the file.'''

  def test_an_intact_chain_verifies(self, tmp_path):
    log = make_log(tmp_path / 'log.jsonl')
    for _ in range(3):
      log.append('buy')
    result = log.verify_chain()
    assert result.ok is True
    assert result.checked == 3
    assert result.first_bad_seq is None

  def test_a_missing_file_verifies_as_an_empty_chain(self, tmp_path):
    result = verify_chain(tmp_path / 'absent.jsonl')
    assert result.ok is True
    assert result.checked == 0

  def test_a_standalone_verification_needs_no_open_log(self, tmp_path):
    path = tmp_path / 'log.jsonl'
    make_log(path).append('buy')
    assert verify_chain(path).ok is True

  def test_an_edited_earlier_record_is_detected(self, tmp_path):
    # The load-bearing tamper test. Editing record 1 of 3 changes its own
    # hash, so verification fails *at record 1* and names the position.
    path = tmp_path / 'log.jsonl'
    log = make_log(path)
    log.append('buy')
    log.append('sell')
    log.append('buy')
    forged = rewrite_line(path, 0, lambda payload: {
      **payload, 'decision': 'sell'})
    assert forged['decision'] == 'sell'
    result = verify_chain(path)
    assert result.ok is False
    assert result.first_bad_seq == 1

  def test_the_edited_record_says_it_was_edited(self, tmp_path):
    path = tmp_path / 'log.jsonl'
    log = make_log(path)
    log.append('buy')
    log.append('sell')
    rewrite_line(path, 0, lambda payload: {**payload, 'decision': 'sell'})
    assert 'was edited' in verify_chain(path).reason

  def test_a_removed_record_is_detected(self, tmp_path):
    # Deleting a record leaves the following one's prev_hash dangling.
    path = tmp_path / 'log.jsonl'
    log = make_log(path)
    log.append('buy')
    log.append('sell')
    log.append('hold')
    lines = path.read_text(encoding='utf-8').splitlines()
    del lines[1]
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    result = verify_chain(path)
    assert result.ok is False
    # The record that no longer fits is 3, and the reason says the link
    # broke rather than the file being renumbered: those are different
    # events and the operator's next step differs.
    assert result.first_bad_seq == 3
    assert 'broken between records' in result.reason

  def test_a_renumbered_record_is_detected(self, tmp_path):
    path = tmp_path / 'log.jsonl'
    log = make_log(path)
    log.append('buy')
    log.append('sell')
    rewrite_line(path, 1, lambda payload: {**payload, 'seq': 7})
    result = verify_chain(path)
    assert result.ok is False
    assert 'renumbered' in result.reason

  def test_an_edited_indicator_value_is_detected(self, tmp_path):
    # Re-deciding after the fact with a more flattering indicator is the
    # realistic tamper, and it is caught because the value is hashed.
    path = tmp_path / 'log.jsonl'
    log = make_log(path)
    log.append('buy', indicators={'rsi': 28.0})
    rewrite_line(path, 0, lambda payload: {
      **payload, 'indicators': {'rsi': '72.0'}})
    assert verify_chain(path).ok is False

  def test_a_swapped_policy_fingerprint_is_detected(self, tmp_path):
    # The one tamper that matters for 9.1/9.9: pointing a record at a
    # different registered algo.
    path = tmp_path / 'log.jsonl'
    log = make_log(path)
    log.append('buy')
    rewrite_line(path, 0, lambda payload: {
      **payload, 'policy_fingerprint': 'someone-elses-algo'})
    assert verify_chain(path).ok is False

  def test_an_unparseable_line_is_detected(self, tmp_path):
    path = tmp_path / 'log.jsonl'
    make_log(path).append('buy')
    with path.open('a', encoding='utf-8') as handle:
      handle.write('{not json\n')
    result = verify_chain(path)
    assert result.ok is False
    assert 'unreadable' in result.reason

  def test_a_recomputed_hash_still_breaks_the_next_link(self, tmp_path):
    # The honest limit of a chain with no external witness: an editor who
    # recomputes the edited record's own hash still cannot repair the
    # *next* record's prev_hash without editing that too. Each further
    # edit extends the forgery; there is no single point to fix.
    path = tmp_path / 'log.jsonl'
    log = make_log(path)
    log.append('buy')
    log.append('sell')
    forged = rewrite_line(path, 0, lambda payload: {
      **payload, 'decision': 'sell'})
    repaired = AuditRecord.from_json(forged)
    lines = path.read_text(encoding='utf-8').splitlines()
    lines[0] = json.dumps(repaired.sealed().to_json(), sort_keys=True,
                           separators=(',', ':'))
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    assert verify_chain(path).first_bad_seq == 2

  def test_appending_to_a_broken_chain_is_refused(self, tmp_path):
    # Evidence must not be buried under more records.
    path = tmp_path / 'log.jsonl'
    log = make_log(path)
    log.append('buy')
    log.append('sell')
    rewrite_line(path, 0, lambda payload: {**payload, 'decision': 'sell'})
    reopened = make_log(path)
    with pytest.raises(AuditChainError, match='broken chain'):
      reopened.append('buy')

  def test_a_broken_chain_stays_readable(self, tmp_path):
    path = tmp_path / 'log.jsonl'
    log = make_log(path)
    log.append('buy')
    log.append('sell')
    rewrite_line(path, 0, lambda payload: {**payload, 'decision': 'sell'})
    assert len(make_log(path).records()) == 2

  def test_the_broken_chain_message_names_the_position(self, tmp_path):
    path = tmp_path / 'log.jsonl'
    log = make_log(path)
    for _ in range(3):
      log.append('buy')
    rewrite_line(path, 1, lambda payload: {**payload, 'decision': 'x'})
    assert 'at record 2 after 1 verified' in verify_chain(path).render()


class TestAuditRoundTrip:
  '''Records must survive the JSON they are written as.'''

  def test_a_record_round_trips_through_json(self, tmp_path):
    log = make_log(tmp_path / 'audit-rt.jsonl')
    written = log.append('buy', rule_ids=('rsi<30',),
                         indicators={'rsi': 29.5, 'pct': 0.125},
                         timeframe='daily',
                         timeframe_criterion='daily momentum regime',
                         features={'rsi': 29.5})
    parsed = AuditRecord.from_json(written.to_json())
    assert parsed == written

  def test_the_indicators_round_trip_to_the_exact_double(self, tmp_path):
    log = make_log(tmp_path / 'audit-rt2.jsonl')
    written = log.append('buy', indicators={'x': 0.1 + 0.2})
    assert AuditRecord.from_json(
      written.to_json()).indicators['x'] == 0.1 + 0.2

  def test_the_written_file_parses_back_to_the_same_records(self, tmp_path):
    path = tmp_path / 'log.jsonl'
    log = make_log(path)
    written = [log.append('buy'), log.append('sell')]
    assert make_log(path).records() == tuple(written)

  def test_one_record_is_exactly_one_line(self, tmp_path):
    path = tmp_path / 'log.jsonl'
    log = make_log(path)
    log.append('buy')
    log.append('sell')
    assert len(path.read_text(encoding='utf-8').splitlines()) == 2

  def test_a_second_log_extends_the_same_chain(self, tmp_path):
    path = tmp_path / 'log.jsonl'
    first = make_log(path)
    first.append('buy')
    second = make_log(path)
    record = second.append('sell')
    assert record.seq == 2
    assert verify_chain(path).ok is True

  def test_the_written_file_is_byte_identical_across_runs(self, tmp_path):
    first = tmp_path / 'one.jsonl'
    second = tmp_path / 'two.jsonl'
    for path in (first, second):
      make_log(path).append('buy', rule_ids=('r',),
                            indicators={'rsi': 28.0}, timeframe='daily')
    assert first.read_bytes() == second.read_bytes()

  def test_the_schema_tag_is_written_on_every_record(self, tmp_path):
    path = tmp_path / 'log.jsonl'
    make_log(path).append('buy')
    line = json.loads(path.read_text(encoding='utf-8').strip())
    assert line['schema'] == audit_schema

  def test_a_foreign_schema_is_refused_on_read(self, tmp_path):
    path = tmp_path / 'log.jsonl'
    make_log(path).append('buy')
    rewrite_line(path, 0, lambda payload: {**payload, 'schema': 'x/1'})
    with pytest.raises(ValueError, match='is not'):
      make_log(path).records()

  def test_a_malformed_record_is_refused_on_read(self):
    with pytest.raises(ValueError, match='malformed audit record'):
      AuditRecord.from_json({'schema': audit_schema, 'seq': 1})

  def test_a_non_finite_record_is_refused_on_read(self):
    with pytest.raises(ValueError, match='malformed audit record'):
      AuditRecord.from_json({'schema': audit_schema, 'seq': 'x'})


class TestAuditRetention:
  '''Retention is sourced, not asserted in prose.'''

  def test_the_period_is_eight_years(self, tmp_path):
    log = make_log(tmp_path / 'audit-ret.jsonl')
    assert log.retention_years == 8

  def test_the_source_names_the_regulation(self, tmp_path):
    log = make_log(tmp_path / 'audit-ret2.jsonl')
    assert 'Reg. 16' in log.retention_source

  def test_the_record_class_is_the_algo_audit_trail(self):
    assert decision_log_class == 'algo_audit_trail'

  def test_there_is_no_seven_year_figure_anywhere(self, tmp_path):
    # The number appears in design documents attributed to SEBI and has no
    # instrument behind it. A plausible wrong constant is worse than a
    # missing one.
    log = make_log(tmp_path / 'audit-ret3.jsonl')
    assert log.retention_years != 7
    assert '7 year' not in log.retention_source

  def test_expiry_is_calendar_years_not_a_365_day_approximation(self, tmp_path):
    log = make_log(tmp_path / 'log.jsonl')
    expiry = log.expires_on(datetime(2026, 3, 2, 9, 15))
    assert expiry is not None
    assert (expiry.year, expiry.month, expiry.day) == (2034, 3, 2)

  def test_the_log_names_its_period_and_source_in_a_render(self, tmp_path):
    text = make_log(tmp_path / 'log.jsonl').render()
    assert '8 years' in text
    assert 'Reg. 16' in text


class TestChainVerification:
  '''The verification result is a report, not a boolean.'''

  def test_an_intact_verification_renders_a_pass(self):
    assert 'verified' in ChainVerification(True, 5).render()

  def test_a_failure_names_the_record_and_what_verified(self):
    text = ChainVerification(False, 9, 4, 'edited').render()
    assert 'CHAIN BROKEN at record 4' in text
    assert '9 verified' in text

  def test_verified_is_an_alias_of_ok(self):
    assert ChainVerification(True, 0).verified is True


class TestMoreConstructionAndFailurePaths:
  '''The refusals that only fire on an unusual input.'''

  def test_a_negative_fault_count_is_refused(self):
    with pytest.raises(ValueError, match='count must be >= 0'):
      PaperBroker().inject_fault(-1)

  def test_a_zero_fault_count_clears_the_injection(self):
    broker = PaperBroker()
    broker.inject_fault(2)
    broker.inject_fault(0)
    assert broker.is_healthy() is True

  def test_an_unhealthy_venue_is_skipped_by_a_cancel(self):
    # The router does not have to wait for a place to be skipped: health
    # is marked as soon as the threshold is reached.
    first = PaperBroker('first')
    router = FailoverRouter([first, PaperBroker('second')],
                            failure_threshold=1)
    first.inject_fault(1)
    router.place(make_order(price=3_010.0), 3_000.0)
    assert router.is_healthy('first') is False
    ack = router.cancel('anything')
    assert ack.broker == 'second'

  def test_a_cancel_with_no_venue_left_is_a_rejection(self):
    first = PaperBroker('first')
    router = FailoverRouter([first], failure_threshold=1)
    first.inject_fault(5)
    router.place(make_order(price=3_010.0), 3_000.0)
    ack = router.cancel('anything')
    assert ack.status == rejected
    assert 'tried' in ack.reason

  def test_positions_survives_a_broker_that_raises(self):
    class ExplodingBroker(NullBroker):
      '''A broker whose position query always fails.'''

      def positions(self):
        raise BrokerError('no book available')

    router = FailoverRouter([ExplodingBroker('boom'),
                             PaperBroker('paper')])
    router.brokers[1].place(make_order(price=3_010.0), 3_000.0)
    assert router.positions()['RELIANCE'] == 10

  def test_a_non_finite_size_is_refused(self):
    # A subnormal price is the only public route to an infinite share
    # count: notional / price overflows before the floor ever runs, and
    # the guard names the quantity rather than letting int() raise three
    # steps later. It is also not a contrived input -- a data vendor that
    # quotes a price in the wrong unit produces exactly this.
    sizer = make_sizer()
    with pytest.raises(ValueError, match='size must be finite'):
      sizer.size('RELIANCE', 1e-320, reference_price=1e-320,
                 inputs=SizingInputs(prob_win=0.9, uncertainty=0.0))

  def test_a_negative_raw_size_is_floored_at_zero(self):
    # A fraction of zero already routes here: the size is zero, so no
    # order is produced and the lot count is never negative.
    sizer = make_sizer()
    result = sizer.size('RELIANCE', 3_000.0, reference_price=3_000.0,
                        inputs=SizingInputs(prob_win=0.5, uncertainty=0.5))
    assert result.quantity == 0

  def test_a_sized_result_renders_its_fill(self):
    sizer = make_sizer()
    text = sizer.size('RELIANCE', 3_000.0, reference_price=3_000.0,
                      inputs=SizingInputs(prob_win=0.9)).render()
    assert 'RELIANCE buy' in text

  def test_a_json_line_that_is_not_an_object_is_refused(self, tmp_path):
    path = tmp_path / 'log.jsonl'
    make_log(path).append('buy')
    with path.open('a', encoding='utf-8') as handle:
      handle.write('[1, 2, 3]\n')
    assert verify_chain(path).ok is False

  def test_a_blank_line_is_skipped_rather_than_refused(self, tmp_path):
    # A trailing newline at the end of the file is not a record, and a
    # reader that treated it as one could never read a well-formed file.
    path = tmp_path / 'log.jsonl'
    make_log(path).append('buy')
    with path.open('a', encoding='utf-8') as handle:
      handle.write('\n\n')
    assert verify_chain(path).checked == 1

  def test_the_record_constructor_is_covered_by_the_round_trip(self):
    assert AuditRecord(seq=1, at=EPOCH.isoformat(), decision='buy',
                       rule_ids=(), indicators={}, timeframe='daily',
                       timeframe_criterion='', policy_fingerprint='x',
                       features='', prev_hash=genesis_hash).seq == 1


class TestAuditWriteFailure:
  '''A record that cannot be written must raise, not disappear.'''

  def test_an_unwritable_log_raises_on_append(self, tmp_path):
    directory = tmp_path / 'locked'
    directory.mkdir()
    path = directory / 'log.jsonl'
    log = make_log(path)
    directory.chmod(0o500)
    try:
      with pytest.raises(AuditChainError, match='could not be written'):
        log.append('buy')
    finally:
      directory.chmod(0o700)

  def test_a_refused_write_says_the_decision_is_unaccounted_for(
      self, tmp_path):
    directory = tmp_path / 'locked2'
    directory.mkdir()
    log = make_log(directory / 'log.jsonl')
    directory.chmod(0o500)
    try:
      with pytest.raises(AuditChainError, match='nobody can account'):
        log.append('sell')
    finally:
      directory.chmod(0o700)


def test_the_default_audit_clock_is_aware_utc():
  # Every other test injects a fixed clock, so without this the default
  # would never execute: it is the one the production path uses.
  moment = audit_utcnow()
  assert moment.tzinfo is not None
  assert moment.utcoffset().total_seconds() == 0.0


def test_a_naive_decision_time_is_stamped_as_utc(tmp_path):
  # The log must not reject a naive timestamp: a caller with a naive clock
  # would otherwise have to know to attach a zone, and a rejected record
  # is a decision that is not accounted for.
  log = make_log(tmp_path / 'naive.jsonl')
  record = log.append('buy', at=datetime(2026, 3, 2, 9, 15))
  assert record.at == '2026-03-02T09:15:00+00:00'


class TestOrderPathEndToEnd:
  '''Size, clear, send and record, in one process, in order.

  This is the property the project-structure note argues for and that a
  microservices split would break: risk and execution in the same process
  means a partition between them cannot fail open. The test walks the
  whole path because a seam that is only exercised in isolation is a seam
  nobody has checked composes.
  '''

  def test_a_sized_order_is_cleared_sent_and_recorded(self, tmp_path):
    checker = make_checker(max_order_value=500_000.0)
    sizer = StockSizer(checker, 1_000_000.0)
    broker = PaperBroker('paper')
    log = make_log(tmp_path / 'log.jsonl')

    result = sizer.size('RELIANCE', 3_000.0, reference_price=3_000.0,
                        algo_id='99999', order_id='A-1',
                        inputs=SizingInputs(prob_win=0.9, uncertainty=0.2))
    ack = broker.place(result.order, 3_000.0)
    record = log.append(result.order.order_id,
                        rule_ids=('kelly', 'rms_pre_trade'),
                        indicators={'prob_win': 0.9, 'uncertainty': 0.2},
                        timeframe='daily',
                        timeframe_criterion='daily momentum regime',
                        feature_digest=feature_hash(
                          {'prob_win': 0.9, 'uncertainty': 0.2}))

    assert result.report.approved is True
    assert ack.status == filled
    assert broker.positions()['RELIANCE'] == result.quantity
    assert record.decision == 'A-1'
    assert verify_chain(tmp_path / 'log.jsonl').ok is True

  def test_a_capped_size_stops_the_path_before_the_broker(self,
                                                          tmp_path):
    checker = make_checker(max_order_value=100_000.0)
    sizer = StockSizer(checker, 10_000_000.0)
    broker = PaperBroker('paper')
    with pytest.raises(RiskCapBreach):
      sizer.size('RELIANCE', 3_000.0, reference_price=3_000.0,
                 inputs=SizingInputs(prob_win=0.9, uncertainty=0.0))
    # Nothing reached the venue and nothing reached the log: the order was
    # never a thing that existed.
    assert not broker.orders
    assert not (tmp_path / 'log.jsonl').exists()
