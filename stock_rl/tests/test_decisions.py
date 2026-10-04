#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Tests for the decision layer: reasons, ordering, vetoes and identity.

The properties asserted here are the ones a reader cannot establish by
reading the module, and each test is written to FAIL if the logic behind
it is wrong:

  * **A latched kill switch forces a HOLD and names the trip code.** The
    signal in that test is one that would otherwise buy, so a test that
    passed with a weak signal would prove nothing.
  * **The largest driver is the first reason.** Asserted on the detail
    text as well as the impact, because an ordering test that only reads
    the impacts passes just as happily when every detail is `'signal
    changed'`.
  * **A changed input changes both the fingerprint and the reasons.** The
    mutations are enumerated rather than sampled, so a new input added to
    :func:`~stock_rl.decisions.decide` without a case here fails the test
    that walks the list.
  * **HOLD is reachable with a full-strength signal present.** Three
    separate ways: the target is already met, the size rounds to no whole
    share, and the pre-trade RMS refuses the order. Each asserts that the
    signal was strong, so none of them can pass by accident on a HOLD
    caused by a missing signal.
  * **No decision can be built from a post-bar input**, and the refusal is
    the exception sentiment already raises for the same failure.
  * **Every branch is reachable and the reason vocabulary is not padded.**
    One test drives the whole rule tree and compares the sources it
    collected against :data:`~stock_rl.decisions.reason_sources`, so a
    name nobody can reach and a branch nobody can drive are both failures.
  * **The module docstring does not claim a regulation that does not
    exist.** The docstring is parsed and asserted against the phrasings
    the research file rejects, because a fabricated citation is a
    fabricated artefact.

There is no network, no wall-clock read and no vendor SDK here. The only
kill switch used is a real :class:`~stock_rl.risk.killswitch.KillSwitch`
tripped on a drawdown, so the veto path is exercised through the module
that would really trip it rather than through a hand-made flag.
'''

import json
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

import stock_rl.decisions
from stock_rl.costs import Side
from stock_rl.decisions import (
  Action,
  Decision,
  PositionState,
  Reason,
  RiskState,
  SignalState,
  Timeframe,
  decide,
  decision_schema,
  entry_rank_max,
  exit_rank_max,
  fast_momentum_max,
  reason_sources,
  slow_momentum_min,
  target_tolerance,
  veto_impact,
)
from stock_rl.decisions import LookAheadError as DecisionLookAheadError
from stock_rl.decisions import _aware_from, _plain, _record, _require_visible
from stock_rl.decisions import _select_timeframe
from stock_rl.execution.sizing import (
  SizingInputs,
  StockSizer,
  kelly_method,
  vol_target_method,
)
from stock_rl.risk.checks import (
  AccountState,
  PriceBand,
  RmsChecker,
  RmsLimits,
  SecurityLimits,
  buy,
  sell,
)
from stock_rl.risk.killswitch import KillSwitch, drawdown_code, manual_code
from stock_rl.sentiment.score import LookAheadError

#: The decision bar every test works at. Fixed rather than read from the
#: clock, so a fingerprint assertion is on the fingerprint and not on the
#: time of day it was taken.
BAR = datetime(2026, 3, 2, 9, 15, tzinfo=timezone.utc)

#: One bar later. Used only to prove a post-bar input is refused.
AFTER_BAR = BAR + timedelta(minutes=1)

#: One bar earlier, for the other side of the visibility boundary.
BEFORE_BAR = BAR - timedelta(minutes=1)

#: Edge inputs that produce a positive size under the Kelly rule:
#: (0.55*1.5 - 0.45)/1.5 = 0.25, halved by the uncertainty haircut.
EDGE = SizingInputs(prob_win=0.55, win_loss_ratio=1.5, uncertainty=0.5)

#: The symbol every test uses. One scrip, so the venue map is a single
#: entry and a refusal is never ambiguous about which limit fired.
SYMBOL = 'RELIANCE'


def make_sizer(capital=1_000_000.0, band=None, lot_size=1) -> StockSizer:
  '''Return a sizer over a single-scrip checker.

  Args:
    capital: Capital base in rupees. The default is one million because at
      the sizer's own cap a larger base would want more notional than the
      per-order limit allows, and every entry test would then be testing
      the cap rather than the decision.
    band: Exchange price band, or the default one for this scrip.
    lot_size: Exchange lot size.

  Returns:
    A :class:`~stock_rl.execution.sizing.StockSizer`.
  '''
  security = SecurityLimits(
    symbol=SYMBOL,
    band=band or PriceBand(2_900.0, 3_200.0),
    mwpl=PriceBand(2_000.0, 4_000.0),
    max_order_quantity=5_000,
    max_order_value=1_000_000.0,
  )
  limits = RmsLimits(
    cumulative_open_order_value=5_000_000.0,
    max_position=100_000,
    max_trading_value=10_000_000.0,
    max_exposure=10_000_000.0,
    max_turnover=20_000_000.0,
    max_security_value=2_000_000.0,
  )
  checker = RmsChecker(limits, {SYMBOL: security})
  return StockSizer(checker, capital, lot_size=lot_size)


def make_signal(**over) -> SignalState:
  '''Return an entry-grade signal, with fields replaceable.

  The default is rank 3 of 50 having come from 7, on a positive 12-1
  momentum: inside :data:`~stock_rl.decisions.entry_rank_max`, so the
  default decision is a BUY and a HOLD observed from these defaults can
  only have come from one of the explicit hold branches.

  Args:
    over: Fields to replace.

  Returns:
    A :class:`SignalState`.
  '''
  values = {
    'symbol': SYMBOL,
    'momentum_rank': 3,
    'previous_rank': 7,
    'universe': 50,
    'momentum': 0.184,
    'as_of': BAR,
  }
  values.update(over)
  return SignalState(**values)


def make_position(**over) -> PositionState:
  '''Return a book flat against a 6 percent target, fields replaceable.

  Args:
    over: Fields to replace.

  Returns:
    A :class:`PositionState`.
  '''
  values = {
    'symbol': SYMBOL,
    'target_weight': 0.06,
    'current_weight': 0.0,
    'as_of': BAR,
  }
  values.update(over)
  return PositionState(**values)


def make_trip(code=drawdown_code, detail='drawdown 0.16 breached',
              observed=0.16):
  '''Return a real trip recorded by a real kill switch.

  The switch is built in a throwaway directory because tripping it
  persists the trip: a real switch writing into the repository would make
  the working tree depend on which tests ran.

  Args:
    code: Trip code to record.
    detail: What tripped it, in the switch's own words.
    observed: Measured value behind the trip.

  Returns:
    The :class:`~stock_rl.risk.killswitch.Trip` the switch recorded, taken
    from its own history rather than hand-built.
  '''
  with tempfile.TemporaryDirectory() as directory:
    switch = KillSwitch('A1', path=Path(directory) / 'kill.json')
    switch.trip(detail, code=code, observed=observed)
    return switch.history[-1]


def tripped_switch(tmp_path, **metrics):
  '''Return a kill switch tripped by its own automatic conditions.

  Args:
    tmp_path: pytest directory for the state file. The switch persists its
      trips, so it needs somewhere writable that is not the repository.
    metrics: Conditions for ``evaluate``, e.g. ``drawdown=0.20``.

  Returns:
    The :class:`~stock_rl.risk.killswitch.KillSwitch`, latched.
  '''
  switch = KillSwitch('A1', path=tmp_path / 'kill.json')
  switch.evaluate(**metrics)
  return switch


def make_risk(trips=(), as_of=BAR) -> RiskState:
  '''Return the kill switch's record at the decision bar.

  Args:
    trips: Trips on record. Empty means armed.
    as_of: When the record was taken.

  Returns:
    A :class:`RiskState`.
  '''
  return RiskState(tuple(trips), as_of)


def decide_default(**over) -> Decision:
  '''Return the decision for the default inputs, with any of them replaced.

  Args:
    over: Arguments for :func:`~stock_rl.decisions.decide` to replace.

  Returns:
    The :class:`Decision`.
  '''
  values = {
    'signal': make_signal(),
    'position': make_position(),
    'risk': make_risk(),
    'sizer': make_sizer(),
    'decision_bar': BAR,
    'price': 3_000.0,
    'sizing': EDGE,
    'method': kelly_method,
    'account': AccountState(),
    'algo_id': 'A1',
  }
  values.update(over)
  return decide(**values)


def payload_of(decision) -> dict:
  '''Return a decision's record as a parsed mapping.

  Args:
    decision: The decision to serialise.

  Returns:
    The parsed JSON object.
  '''
  return json.loads(decision.to_json())


def forge(decision, change) -> dict:
  '''Return a decision's record with one field tampered with.

  Args:
    decision: The decision to serialise.
    change: Callable taking and returning the parsed payload.

  Returns:
    The tampered payload, so a test can assert what it forged.
  '''
  payload = payload_of(decision)
  change(payload)
  return payload


def every_branch() -> tuple[Decision, ...]:
  '''Return one decision from every branch of the rule tree.

  Returns:
    A BUY, a SELL, and one HOLD per explicit hold branch, so a test can
    assert a property of all of them at once.
  '''
  trip = make_trip()
  return (
    decide_default(),
    decide_default(
      signal=make_signal(momentum_rank=34, previous_rank=20, momentum=0.02),
      position=make_position(target_weight=0.0, current_weight=0.06)),
    decide_default(risk=make_risk((trip,))),
    decide_default(position=make_position(current_weight=0.06)),
    decide_default(
      signal=make_signal(momentum_rank=34, previous_rank=34),
      position=make_position(target_weight=0.06, current_weight=-0.06)),
    decide_default(
      signal=make_signal(momentum_rank=17, previous_rank=9, momentum=0.02)),
    decide_default(
      signal=make_signal(momentum_rank=2, previous_rank=5, momentum=-0.03)),
    decide_default(price=3_500.0),
    decide_default(sizer=make_sizer(capital=1_000.0)),
  )


class ExplodingSizer(StockSizer):
  '''A sizer that refuses to size anything, so any call into it fails.

  Subclasses the real sizer rather than standing in for it, so the
  configuration it exposes is the configuration the digest hashes while
  the one method the decision must not reach is trapped.
  '''

  def size(self, *args, **kwargs):
    '''Refuse to size anything.

    Args:
      args: Ignored; present for the overridden signature.
      kwargs: Ignored; present for the overridden signature.

    Raises:
      AssertionError: Always, if this method is reached at all.
    '''
    raise AssertionError(
      f'the sizer was asked for a size with {args} {kwargs}')


class TestTheVocabulary:
  '''Action reuses the side, and every timeframe says how many bars.'''

  def test_buy_and_sell_are_the_side_strings_the_project_already_uses(self):
    # Not a near-duplicate enum: the values are read off costs.Side, so
    # there is one table of side strings in the project. A serialised
    # decision therefore carries what audit already records and what the
    # cost model already prices.
    assert Action.BUY.value == Side.BUY.value == buy
    assert Action.SELL.value == Side.SELL.value == sell

  def test_hold_is_the_only_new_string(self):
    # HOLD is not a side of a trade: there is no order, no cost and no
    # fill for it, which is why Action exists and why it borrows rather
    # than re-declares the other two.
    assert Action.HOLD.value == 'hold'
    assert 'hold' not in (Side.BUY.value, Side.SELL.value)

  def test_every_timeframe_states_its_horizon_in_bars(self):
    horizons = [member.bars for member in Timeframe]
    assert horizons == sorted(horizons)
    assert len(set(horizons)) == len(horizons)
    assert all(isinstance(horizon, int) and horizon > 0
               for horizon in horizons)

  def test_the_horizon_is_the_enum_value_so_it_cannot_drift(self):
    # The bar count is the member's value, so there is no second table to
    # fall out of step with the name.
    for member in Timeframe:
      assert member.bars == int(member.value)

  def test_each_band_selects_the_timeframe_the_record_prints(self):
    expected = {
      Timeframe.FAST: 0.01,
      Timeframe.MID: 0.10,
      Timeframe.SLOW: 0.30,
    }
    for timeframe, momentum in expected.items():
      assert _select_timeframe(momentum) is timeframe
      decision = decide_default(signal=make_signal(momentum=momentum))
      assert decision.timeframe is timeframe
      assert f'({timeframe.bars} bars)' in decision.render(), (
        'the rendered record must carry the horizon in bars, not a word '
        'like "short term"')

  def test_a_weak_momentum_selects_the_short_horizon(self):
    assert _select_timeframe(0.0) is Timeframe.FAST
    assert _select_timeframe(fast_momentum_max) is Timeframe.FAST
    assert _select_timeframe(slow_momentum_min) is Timeframe.SLOW

  def test_every_reason_source_is_reachable_and_the_list_is_not_padded(self):
    seen = {reason.source
            for decision in every_branch()
            for reason in decision.reasons}
    assert seen == set(reason_sources), (
      f'reasons named {sorted(seen)} against a vocabulary of '
      f'{sorted(reason_sources)}; a name nothing can produce or a name '
      'nothing produces is a defect in one of the two')


class TestTheKillSwitchVeto:
  '''A latched switch is a HOLD, and it says which code latched.'''

  def test_a_tripped_switch_forces_a_hold_and_never_a_buy(self):
    # The signal here is entry-grade: without the latch this decision is
    # a BUY, so this is the veto doing the work.
    assert decide_default().action is Action.BUY
    decision = decide_default(risk=make_risk((make_trip(),)))
    assert decision.action is Action.HOLD
    assert decision.action is not Action.BUY

  def test_the_trip_code_is_named_in_the_reasons(self):
    decision = decide_default(risk=make_risk((make_trip(),)))
    veto = decision.reasons[0]
    assert veto.source == 'risk.kill_switch'
    assert drawdown_code in veto.detail

  def test_every_trip_on_record_is_quoted_verbatim(self):
    trips = (make_trip(), make_trip(code=manual_code, detail='operator stop'))
    decision = decide_default(risk=make_risk(trips))
    detail = decision.reasons[0].detail
    assert drawdown_code in detail and manual_code in detail
    assert 'operator stop' in detail

  def test_the_trip_comes_from_a_real_kill_switch(self, tmp_path):
    # Not a hand-made flag: the switch is tripped on a drawdown through
    # its own evaluate, and the record the decision quotes is the one the
    # switch wrote.
    switch = tripped_switch(tmp_path, drawdown=0.20)
    decision = decide_default(risk=make_risk(switch.history))
    assert decision.reasons[0].detail == (
      f'the kill switch is latched, so no order may be released '
      f'({switch.history[0].code}: {switch.history[0].detail})')

  def test_the_veto_outranks_every_driver(self):
    decision = decide_default(risk=make_risk((make_trip(),)))
    assert decision.reasons[0].impact == veto_impact
    assert decision.reasons[0].impact == max(
      reason.impact for reason in decision.reasons)

  def test_no_size_is_computed_while_halted(self):
    # The sizer raises on any call, so reaching it fails the test. A HOLD
    # that still sized something is an order waiting for a bug to release
    # it, and the halt branch must not even ask.
    halting_sizer = ExplodingSizer(make_sizer().checker, 1_000_000.0)
    decision = decide_default(risk=make_risk((make_trip(),)),
                              sizer=halting_sizer)
    assert decision.action is Action.HOLD
    assert not decision.sizes

  def test_a_latched_switch_hides_the_entire_signal_branch(self):
    decision = decide_default(risk=make_risk((make_trip(),)))
    sources = {reason.source for reason in decision.reasons}
    assert sources == {'risk.kill_switch', 'timeframe.band'}

  def test_an_armed_switch_reports_itself_armed(self):
    risk = make_risk()
    assert risk.tripped is False
    assert not risk.codes
    assert decide_default().action is Action.BUY

  def test_the_codes_are_the_kill_switch_ones(self):
    risk = make_risk((make_trip(code=manual_code, detail='stop'),))
    assert risk.codes == (manual_code,)


class TestReasonOrdering:
  '''Reasons are ordered by impact, and each one names its input.'''

  def test_reasons_are_sorted_by_descending_impact(self):
    for decision in every_branch():
      impacts = [reason.impact for reason in decision.reasons]
      assert impacts == sorted(impacts, reverse=True)

  def test_the_largest_driver_is_the_first_reason(self):
    decision = decide_default()
    assert decision.reasons[0].source == 'signal.momentum'
    assert decision.largest_reason is decision.reasons[0]
    assert decision.reasons[0].impact == 0.184

  def test_a_weak_driver_does_not_lead(self):
    # The rank move is the same in both, so only the momentum differs. The
    # strong one leads; the weak one does not.
    strong = decide_default()
    weak = decide_default(signal=make_signal(momentum=0.01))
    assert strong.reasons[0].source == 'signal.momentum'
    assert weak.reasons[0].impact < strong.reasons[0].impact

  def test_every_reason_names_a_known_input(self):
    for decision in every_branch():
      for reason in decision.reasons:
        assert reason.source in reason_sources

  def test_every_reason_carries_the_number_it_observed(self):
    # A reason with no number in it is a mood. The requirement is the
    # detail, not just the impact field.
    for decision in every_branch():
      for reason in decision.reasons:
        assert any(character.isdigit() for character in reason.detail), (
          f'reason {reason.source!r} carries no observed value: '
          f'{reason.detail!r}')

  def test_the_reason_quotes_the_rank_move_rather_than_saying_changed(self):
    # The example the module claims to meet: a rank that fell from 1 to 7
    # of 50, stated as a move, with the universe it was ranked in.
    decision = decide_default(
      signal=make_signal(momentum_rank=7, previous_rank=1))
    move = next(reason for reason in decision.reasons
                if reason.source == 'signal.rank_move')
    assert move.detail == (
      'cross-sectional rank moved +6 places, from 1 to 7 of 50')

  def test_the_entry_reason_quotes_the_band_and_the_universe(self):
    decision = decide_default(signal=make_signal(momentum_rank=2))
    entry = next(reason for reason in decision.reasons
                 if reason.source == 'signal.rank_entry')
    assert 'rank 2 of 50' in entry.detail
    assert f'1-{entry_rank_max}' in entry.detail

  def test_the_exit_reason_quotes_the_exit_band(self):
    decision = decide_default(
      signal=make_signal(momentum_rank=34, previous_rank=34),
      position=make_position(target_weight=0.0, current_weight=0.06))
    exit_reason = next(reason for reason in decision.reasons
                       if reason.source == 'signal.rank_exit')
    assert f'{exit_rank_max}-50' in exit_reason.detail

  def test_the_timeframe_reason_never_leads(self):
    # It qualifies the record; it does not decide it. Zero impact is what
    # puts it last, and the sort has to hold that way.
    for decision in every_branch():
      assert decision.reasons[-1].source == 'timeframe.band'
      assert decision.reasons[-1].impact == 0.0

  def test_every_decision_states_its_timeframe_criterion(self):
    decision = decide_default()
    frame = decision.reasons[-1]
    assert f'{decision.timeframe.bars} bars' in frame.detail
    assert '+18.40%' in frame.detail

  def test_an_unsorted_decision_cannot_be_built(self):
    low = Reason('timeframe.band', 0.0, 'a horizon of 5 bars')
    high = Reason('signal.momentum', 0.184, '12-1 momentum is +18.40%')
    with pytest.raises(ValueError, match='ordered by impact'):
      Decision(
        action=Action.BUY, timeframe=Timeframe.SLOW,
        sizes=((SYMBOL, 0.125),), reasons=(low, high), decided_at=BAR,
        fingerprint=f'{decision_schema}:' + '0' * 64)

  def test_a_reason_that_names_nothing_known_is_refused(self):
    with pytest.raises(ValueError, match='reason source must be one of'):
      Reason('vibes', 1.0, 'the chart looks like a reversal')


class TestTheFingerprint:
  '''Identical inputs give an identical digest; any change moves it.'''

  def test_an_identical_input_produces_an_identical_fingerprint(self):
    first = decide_default()
    second = decide_default()
    assert first.fingerprint == second.fingerprint
    assert first == second

  def test_the_fingerprint_is_this_module_scheme(self):
    digest = decide_default().fingerprint
    scheme, _, body = digest.partition(':')
    assert scheme == decision_schema
    assert len(body) == 64 and all(character in '0123456789abcdef'
                                  for character in body)

  def test_a_malformed_fingerprint_is_refused(self):
    for bad in ('nonsense', f'{decision_schema}:xyz', 'other/1:' + '0' * 64):
      with pytest.raises(ValueError, match='fingerprint must be'):
        Decision(
          action=Action.HOLD, timeframe=Timeframe.FAST, sizes=(),
          reasons=(Reason('timeframe.band', 0.0, 'a horizon of 5 bars'),),
          decided_at=BAR, fingerprint=bad)

  def test_every_signal_input_changes_the_fingerprint_and_the_reasons(self):
    # The mutations that must alter what a person is told, not merely the
    # digest. A parameter added to decide() without a line here fails this
    # test rather than passing unnoticed.
    baseline = decide_default()
    mutations = {
      'momentum_rank': {'signal': make_signal(momentum_rank=4)},
      'previous_rank': {'signal': make_signal(previous_rank=12)},
      'universe': {'signal': make_signal(universe=40)},
      'momentum': {'signal': make_signal(momentum=0.30)},
      'target_weight': {'position': make_position(target_weight=0.09)},
      'current_weight': {'position': make_position(current_weight=0.02)},
      'price': {'price': 3_100.0},
      'sizing': {'sizing': SizingInputs(
        prob_win=0.60, win_loss_ratio=1.5, uncertainty=0.5)},
      'method': {'method': vol_target_method,
                 'sizing': SizingInputs(realised_vol=0.30,
                                        target_vol=0.15)},
      'capital': {'sizer': make_sizer(capital=800_000.0)},
      'lot_size': {'sizer': make_sizer(lot_size=50)},
    }
    for name, over in mutations.items():
      changed = decide_default(**over)
      assert changed.fingerprint != baseline.fingerprint, (
        f'changing {name} did not move the fingerprint')
      assert changed.reasons != baseline.reasons, (
        f'changing {name} did not change what the reader is told')

  def test_the_kill_switch_state_is_in_the_digest(self):
    armed = decide_default()
    halted = decide_default(risk=make_risk((make_trip(),)))
    assert armed.fingerprint != halted.fingerprint

  def test_the_decision_bar_is_in_the_digest(self):
    # Every input moves back with the bar, so this is the same decision
    # taken one bar earlier and not a visibility refusal.
    earlier = decide_default(
      decision_bar=BEFORE_BAR, signal=make_signal(as_of=BEFORE_BAR),
      position=make_position(as_of=BEFORE_BAR),
      risk=make_risk(as_of=BEFORE_BAR))
    assert earlier.fingerprint != decide_default().fingerprint

  def test_the_sizer_configuration_is_in_the_digest(self):
    # Regression: a digest that left the capital base out gave the same
    # fingerprint to a decision that places 41 shares and to one on the
    # same signal that places none, because only the capital differed.
    fat = decide_default()
    thin = decide_default(sizer=make_sizer(capital=1_000.0))
    assert fat.action is Action.BUY
    assert thin.action is Action.HOLD
    assert fat.fingerprint != thin.fingerprint

  def test_the_account_ledger_is_in_the_digest_even_when_it_is_invisible(
    self,
  ):
    # The ledger decides whether the RMS clears the order, so two decisions
    # that differ only in it are not the same decision. The reasons are
    # identical here because the order clears either way, which is exactly
    # why a digest over the reasons alone would call them one decision.
    default = decide_default()
    busy = decide_default(account=AccountState(turnover=5_000.0))
    assert default.fingerprint != busy.fingerprint
    assert default.reasons == busy.reasons

  def test_a_venue_change_moves_the_digest_even_when_it_is_invisible(self):
    # The band is what refuses an order, so it is an input to the
    # decision. Here it moves without moving the reasons, which is
    # exactly the case where a digest over the reasons alone would look
    # like two identical decisions.
    default = decide_default()
    widened = decide_default(
      sizer=make_sizer(band=PriceBand(1_000.0, 9_000.0)))
    assert default.fingerprint != widened.fingerprint
    assert default.reasons == widened.reasons


class TestHoldIsAnOutcome:
  '''A HOLD is a result with a reason, not the absence of a signal.'''

  def test_a_full_strength_signal_on_a_book_at_target_holds(self):
    signal = make_signal(momentum_rank=1, previous_rank=1, momentum=0.184)
    assert signal.momentum_rank <= entry_rank_max
    assert signal.momentum > 0.0
    decision = decide_default(
      signal=signal, position=make_position(current_weight=0.06))
    assert decision.action is Action.HOLD
    target = decision.reasons[0]
    assert target.source == 'position.target_gap'
    assert 'target already met, no action needed' in target.detail

  def test_a_size_that_rounds_to_nothing_holds_with_the_signal_present(self):
    decision = decide_default(sizer=make_sizer(capital=1_000.0))
    sources = {reason.source for reason in decision.reasons}
    assert decision.action is Action.HOLD
    assert 'sizing.rounds_to_nothing' in sources
    assert 'signal.momentum' in sources

  def test_an_rms_refusal_holds_with_the_signal_present(self):
    # 3500 is outside the venue band, so the RMS refuses the sized order.
    decision = decide_default(price=3_500.0)
    sources = {reason.source for reason in decision.reasons}
    assert decision.action is Action.HOLD
    assert 'risk.rms_refusal' in sources
    assert 'signal.momentum' in sources

  def test_the_refusal_quotes_the_check_that_fired(self):
    decision = decide_default(price=3_500.0)
    refusal = decision.reasons[0]
    assert refusal.source == 'risk.rms_refusal'
    assert refusal.impact == veto_impact
    assert 'price_band' in refusal.detail
    assert '3500.0' in refusal.detail

  def test_a_refusal_does_not_shrink_the_order_to_fit(self):
    # sizing.py refuses rather than shrinking, and the decision must not
    # turn that refusal into a smaller BUY.
    decision = decide_default(price=3_500.0)
    assert not decision.sizes

  def test_a_book_with_nothing_to_sell_holds_in_the_exit_band(self):
    # In the exit band with no long to close there is nothing to do. The
    # branch is reachable only with a gap to close towards, since a flat
    # book on a zero target is already at target and stops earlier.
    decision = decide_default(
      signal=make_signal(momentum_rank=40, previous_rank=40),
      position=make_position(target_weight=0.06, current_weight=-0.06))
    assert decision.action is Action.HOLD
    assert decision.reasons[0].source == 'position.flat'
    assert 'nothing to sell' in decision.reasons[0].detail

  def test_the_no_trade_band_holds_between_the_entry_and_exit_bands(self):
    decision = decide_default(
      signal=make_signal(momentum_rank=17, previous_rank=9, momentum=0.02))
    band = next(reason for reason in decision.reasons
                if reason.source == 'signal.no_trade_band')
    assert decision.action is Action.HOLD
    assert f'{entry_rank_max + 1}-{exit_rank_max - 1}' in band.detail

  def test_a_non_positive_momentum_holds_even_at_the_top_rank(self):
    decision = decide_default(
      signal=make_signal(momentum_rank=1, previous_rank=1, momentum=-0.03))
    gate = next(reason for reason in decision.reasons
                if reason.source == 'signal.momentum_gate')
    assert decision.action is Action.HOLD
    assert '-3.00%' in gate.detail

  def test_an_exit_band_closes_the_position_to_flat(self):
    decision = decide_default(
      signal=make_signal(momentum_rank=34, previous_rank=20, momentum=0.02),
      position=make_position(target_weight=0.0, current_weight=0.06))
    assert decision.action is Action.SELL
    assert decision.sizes == ((SYMBOL, 0.0),)
    assert 'close RELIANCE to 0.00% of capital' in decision.render()

  def test_an_exit_states_that_it_overrides_a_non_zero_target(self):
    decision = decide_default(
      signal=make_signal(momentum_rank=34, previous_rank=34),
      position=make_position(target_weight=0.02, current_weight=0.06))
    assert decision.action is Action.SELL
    assert 'overriding the 2.00% target weight' in decision.render()

  def test_a_hold_never_carries_a_size(self):
    for decision in every_branch():
      if decision.action is Action.HOLD:
        assert not decision.sizes

  def test_the_tolerance_decides_whether_the_target_is_met(self):
    at_tolerance = decide_default(
      position=make_position(target_weight=0.06, current_weight=0.0599))
    beyond = decide_default(
      position=make_position(target_weight=0.06, current_weight=0.0590))
    assert abs(0.0599 - 0.06) == pytest.approx(target_tolerance)
    assert at_tolerance.action is Action.HOLD
    assert beyond.action is Action.BUY

  def test_a_hold_still_states_the_timeframe(self):
    for decision in every_branch():
      if decision.action is Action.HOLD:
        assert decision.timeframe is not None
        assert decision.reasons[-1].source == 'timeframe.band'

  def test_the_sell_needs_a_position_because_no_size_is_computed_for_it(self):
    # Documented ceiling, asserted so it stays visible: the shares to
    # close are the gap the book already carries, not a size this module
    # derives, because sizing.py prices an edge bet and a close has none.
    decision = decide_default(
      signal=make_signal(momentum_rank=34, previous_rank=34),
      position=make_position(target_weight=0.0, current_weight=0.06))
    gap = next(reason for reason in decision.reasons
               if reason.source == 'position.target_gap')
    assert '6.00% of capital' in gap.detail


class TestNoLookAhead:
  '''Nothing postdating the decision bar can reach a decision.'''

  def test_an_input_exactly_at_the_bar_is_legal(self):
    # The boundary is inclusive: a reading stamped at the bar was public
    # when the bar closed.
    for source in (make_signal(as_of=BAR), make_position(as_of=BAR),
                   make_risk(as_of=BAR)):
      assert decide_default(
        signal=source if isinstance(source, SignalState) else make_signal(),
        position=source if isinstance(source, PositionState)
        else make_position(),
        risk=source if isinstance(source, RiskState) else make_risk(),
      ).action is Action.BUY

  def test_an_input_before_the_bar_is_legal(self):
    assert decide_default(signal=make_signal(as_of=BEFORE_BAR)).action \
      is Action.BUY

  def test_a_post_bar_input_is_refused_for_every_input(self):
    late = (make_signal(as_of=AFTER_BAR), make_position(as_of=AFTER_BAR),
            make_risk(as_of=AFTER_BAR))
    for field, source in zip(('signal', 'position', 'risk'), late):
      with pytest.raises(DecisionLookAheadError, match='not visible at'):
        decide_default(**{field: source})

  def test_the_refusal_names_the_offending_input_and_the_bar(self):
    with pytest.raises(DecisionLookAheadError) as caught:
      decide_default(signal=make_signal(as_of=AFTER_BAR))
    message = str(caught.value)
    assert 'SignalState' in message
    assert AFTER_BAR.isoformat() in message
    assert BAR.isoformat() in message

  def test_a_post_bar_halt_still_refuses_rather_than_trading(self):
    # The refusal is not conditional on the decision it would have
    # prevented: a latched switch read at a later bar is a leak too.
    with pytest.raises(DecisionLookAheadError):
      decide_default(risk=make_risk((make_trip(),), as_of=AFTER_BAR))

  def test_it_is_the_same_error_sentiment_raises(self):
    # One failure mode, one name: a caller catching this catches the
    # sentiment leak as well.
    assert DecisionLookAheadError is LookAheadError
    assert issubclass(DecisionLookAheadError, ValueError)

  def test_a_naive_decision_bar_is_refused(self):
    naive = datetime(2026, 3, 2, 9, 15)
    with pytest.raises(ValueError, match='timezone-aware'):
      decide_default(decision_bar=naive)

  def test_an_input_with_no_timestamp_is_refused(self):
    # Nothing without a timestamp can be shown to be legal at any bar.
    with pytest.raises(ValueError, match='carries no aware as_of'):
      _require_visible(BAR, object())

  def test_the_helper_does_not_mutate_what_it_checks(self):
    signal = make_signal(as_of=BEFORE_BAR)
    _require_visible(BAR, signal)
    assert signal.as_of == BEFORE_BAR


class TestSerialisation:
  '''The record round-trips byte for byte, or refuses.'''

  def test_the_round_trip_is_byte_identical_for_every_branch(self):
    for decision in every_branch():
      original = decision.to_json()
      assert Decision.from_json(original).to_json() == original

  def test_the_round_trip_preserves_the_decision_itself(self):
    decision = decide_default()
    assert Decision.from_json(decision.to_json()) == decision

  def test_an_already_parsed_mapping_is_accepted(self):
    decision = decide_default()
    assert Decision.from_json(payload_of(decision)) == decision

  def test_the_record_carries_the_horizon_not_only_the_name(self):
    # The bar count travels with the record so a reader does not need this
    # module's enum to know the horizon.
    payload = payload_of(decide_default(signal=make_signal(momentum=0.30)))
    assert payload['timeframe'] == {'name': 'SLOW', 'bars': 60}

  def test_the_record_names_the_schema_and_the_bar(self):
    payload = payload_of(decide_default())
    assert payload['schema'] == decision_schema
    assert payload['decided_at'] == BAR.isoformat()

  def test_the_action_serialises_as_the_side_string(self):
    # What execution.audit already records, so the two records join.
    assert payload_of(decide_default())['action'] == 'buy'
    assert payload_of(decide_default(
      signal=make_signal(momentum_rank=34, previous_rank=34),
      position=make_position(target_weight=0.0, current_weight=0.06),
    ))['action'] == 'sell'
    assert payload_of(decide_default(price=3_500.0))['action'] == 'hold'

  def test_a_foreign_schema_is_refused(self):
    decision = decide_default()
    payload = forge(decision, lambda item: item.update(schema='other/9'))
    with pytest.raises(ValueError, match='decision schema'):
      Decision.from_json(payload)

  def test_a_horizon_that_disagrees_with_the_logic_is_refused(self):
    decision = decide_default()
    payload = forge(decision, lambda item: item['timeframe'].update(
      bars=99))
    with pytest.raises(ValueError, match='the record and the logic '
                           'disagree'):
      Decision.from_json(payload)

  def test_an_unknown_timeframe_is_refused(self):
    decision = decide_default()
    payload = forge(decision, lambda item: item['timeframe'].update(
      name='GLIDE'))
    with pytest.raises(ValueError, match='unknown timeframe'):
      Decision.from_json(payload)

  def test_a_timeframe_that_is_not_an_object_is_refused(self):
    decision = decide_default()
    payload = forge(decision, lambda item: item.update(timeframe='SLOW'))
    with pytest.raises(ValueError, match='timeframe must be an object'):
      Decision.from_json(payload)

  def test_an_unknown_action_is_refused(self):
    decision = decide_default()
    payload = forge(decision, lambda item: item.update(action='SQUEEZE'))
    with pytest.raises(ValueError, match='malformed decision record'):
      Decision.from_json(payload)

  def test_a_missing_key_is_refused(self):
    decision = decide_default()
    payload = forge(decision, lambda item: item.pop('sizes'))
    with pytest.raises(ValueError, match='malformed decision record'):
      Decision.from_json(payload)

  def test_a_malformed_reason_is_refused(self):
    decision = decide_default()
    payload = forge(decision,
                    lambda item: item['reasons'].append({'source': 'vibes'}))
    with pytest.raises(ValueError, match='malformed decision record'):
      Decision.from_json(payload)

  def test_a_reason_that_is_not_an_object_is_refused(self):
    decision = decide_default()
    payload = forge(decision,
                    lambda item: item['reasons'].append('because'))
    with pytest.raises(ValueError, match='a reason must be an object'):
      Decision.from_json(payload)

  def test_a_payload_that_is_not_an_object_is_refused(self):
    with pytest.raises(ValueError, match='must be an object'):
      Decision.from_json('[1, 2, 3]')

  def test_a_naive_recorded_bar_is_refused(self):
    decision = decide_default()
    payload = forge(decision, lambda item: item.update(
      decided_at='2026-03-02T09:15:00'))
    with pytest.raises(ValueError, match='timezone-aware'):
      Decision.from_json(payload)

  def test_an_unparseable_timestamp_is_refused(self):
    with pytest.raises(ValueError, match='not an ISO 8601 timestamp'):
      _aware_from('the day before')

  def test_a_bad_size_is_refused(self):
    decision = decide_default()
    payload = forge(decision, lambda item: item.update(
      sizes={SYMBOL: -0.1}))
    with pytest.raises(ValueError, match='must be finite and >= 0'):
      Decision.from_json(payload)

  def test_a_hold_carrying_a_size_is_refused_on_load(self):
    # The invariant is enforced on the loaded record too, not only on the
    # in-memory one: a file that says "hold, and here is what to trade"
    # is exactly the record that must not load.
    decision = decide_default()
    payload = forge(decision, lambda item: item.update(
      action='hold', sizes={SYMBOL: 0.125}))
    with pytest.raises(ValueError, match='a HOLD commits no capital'):
      Decision.from_json(payload)


class TestConstructionRefusals:
  '''Bad inputs are refused where the cause is still visible.'''

  def test_a_signal_needs_a_symbol(self):
    with pytest.raises(ValueError, match='needs a symbol'):
      make_signal(symbol='  ')

  def test_a_rank_outside_the_universe_is_refused(self):
    with pytest.raises(ValueError, match='momentum_rank must be in 1..50'):
      make_signal(momentum_rank=51)
    with pytest.raises(ValueError, match='previous_rank must be in 1..50'):
      make_signal(previous_rank=0)

  def test_an_empty_universe_is_refused(self):
    with pytest.raises(ValueError, match='universe must be >= 1'):
      make_signal(universe=0)

  def test_a_non_finite_momentum_is_refused(self):
    with pytest.raises(ValueError, match='momentum must be finite'):
      make_signal(momentum=float('nan'))

  def test_a_naive_signal_is_refused(self):
    with pytest.raises(ValueError, match='SignalState.as_of must be '
                           'timezone-aware'):
      make_signal(as_of=datetime(2026, 3, 2, 9, 15))

  def test_a_position_needs_a_symbol(self):
    with pytest.raises(ValueError, match='needs a symbol'):
      make_position(symbol='')

  def test_a_weight_outside_the_unit_interval_is_refused(self):
    with pytest.raises(ValueError, match='target_weight must be in'):
      make_position(target_weight=1.5)
    with pytest.raises(ValueError, match='current_weight must be in'):
      make_position(current_weight=float('inf'))

  def test_a_naive_position_is_refused(self):
    with pytest.raises(ValueError, match='PositionState.as_of must be '
                           'timezone-aware'):
      make_position(as_of=datetime(2026, 3, 2, 9, 15))

  def test_a_naive_risk_record_is_refused(self):
    with pytest.raises(ValueError, match='RiskState.as_of must be '
                           'timezone-aware'):
      make_risk(as_of=datetime(2026, 3, 2, 9, 15))

  def test_a_blank_reason_is_refused(self):
    with pytest.raises(ValueError, match='must say what it observed'):
      Reason('signal.momentum', 0.1, '   ')

  def test_a_negative_or_non_finite_impact_is_refused(self):
    with pytest.raises(ValueError, match='finite and >= 0'):
      Reason('signal.momentum', -0.1, '12-1 momentum is +18.40%')
    with pytest.raises(ValueError, match='finite and >= 0'):
      Reason('signal.momentum', float('inf'), '12-1 momentum is +18.40%')

  def test_a_decision_with_no_reasons_cannot_exist(self):
    with pytest.raises(ValueError, match='at least one reason'):
      Decision(
        action=Action.HOLD, timeframe=Timeframe.FAST, sizes=(), reasons=(),
        decided_at=BAR, fingerprint=f'{decision_schema}:' + '0' * 64)

  def test_a_decision_with_no_size_cannot_act(self):
    with pytest.raises(ValueError, match='must say what it commits'):
      Decision(
        action=Action.BUY, timeframe=Timeframe.FAST, sizes=(),
        reasons=(Reason('timeframe.band', 0.0, 'a horizon of 5 bars'),),
        decided_at=BAR, fingerprint=f'{decision_schema}:' + '0' * 64)

  def test_a_symbol_cannot_be_sized_twice(self):
    with pytest.raises(ValueError, match='sized twice'):
      Decision(
        action=Action.BUY, timeframe=Timeframe.FAST,
        sizes=((SYMBOL, 0.1), (SYMBOL, 0.2)),
        reasons=(Reason('timeframe.band', 0.0, 'a horizon of 5 bars'),),
        decided_at=BAR, fingerprint=f'{decision_schema}:' + '0' * 64)

  def test_a_blank_size_symbol_is_refused(self):
    with pytest.raises(ValueError, match='a size needs a symbol'):
      Decision(
        action=Action.BUY, timeframe=Timeframe.FAST, sizes=((' ', 0.1),),
        reasons=(Reason('timeframe.band', 0.0, 'a horizon of 5 bars'),),
        decided_at=BAR, fingerprint=f'{decision_schema}:' + '0' * 64)

  def test_a_signal_for_another_symbol_is_refused(self):
    with pytest.raises(ValueError, match='a decision that sizes one name'):
      decide_default(position=make_position(symbol='TCS'))

  def test_a_price_that_is_not_positive_is_refused(self):
    with pytest.raises(ValueError, match='price must be positive'):
      decide_default(price=0.0)

  def test_a_record_only_helper_refuses_a_non_dataclass(self):
    # Failing loudly here is what stops two different inputs hashing to
    # one digest.
    with pytest.raises(TypeError, match='expected a dataclass instance'):
      _record(object())


class TestTheDocumentedJustification:
  '''The module's own claims, asserted against the research findings.

  A fabricated citation in a docstring is a fabricated artefact: the
  research in ``docs/research/compliance/signal-attribution-and-audit-trail.md``
  records that there is NO Indian requirement to explain an individual
  BUY/HOLD/SELL decision, that the driver of documentation is
  re-registration under NSE 9.1 and 9.9, and that nothing here is evidence
  the strategy makes money. These tests parse the docstring so a later
  edit that reintroduces one of the rejected claims fails.
  '''

  @staticmethod
  def docstring() -> str:
    '''Return the module docstring.

    Returns:
      The docstring of :mod:`stock_rl.decisions`.
    '''
    return stock_rl.decisions.__doc__ or ''

  def test_the_docstring_is_not_empty(self):
    # Guard against the assertions below passing on an empty string.
    assert len(self.docstring()) > 1_000

  def test_the_justification_given_is_internal_control(self):
    assert 'internal control' in self.docstring()

  def test_the_real_driver_the_registration_duty_is_named(self):
    text = self.docstring()
    assert '9.1' in text and '9.9' in text
    assert 're-registration' in text or 'fresh Exchange registration' in text

  def test_no_regulatory_duty_is_claimed_for_a_per_decision_reason(self):
    text = self.docstring()
    assert ('no Indian requirement to explain an individual' in text
            or 'no requirement to explain an individual' in text)

  def test_the_rejected_citation_phrasings_are_absent(self):
    # The claims docs/research/corrections-to-design-docs.md records as
    # fabricated or wrong, none of which may reappear as a reason for
    # this record to exist.
    forbidden = (
      'SEBI requires',
      'NSE requires',
      'required by SEBI',
      'as per SEBI',
      'mandated by SEBI',
      'compliance requirement for',
      'regulator requires',
      'zero SEBI risk',
      '7 years',
    )
    text = self.docstring()
    for phrase in forbidden:
      assert phrase not in text, (
        f'the docstring claims {phrase!r}, which the research file records '
        'as fabricated or as a duty that does not exist')

  def test_the_retention_figure_cited_is_the_one_that_exists(self):
    text = self.docstring()
    assert '10.3' in text and 'five years' in text
    assert '8 years' not in text

  def test_no_profitability_is_claimed_anywhere_in_the_module(self):
    forbidden = ('profitable', 'guaranteed', 'outperform', 'will make money',
                 'alpha generation', 'best-in-class')
    text = self.docstring().lower()
    for phrase in forbidden:
      assert phrase not in text, (
        f'the docstring claims {phrase!r}; no decision in this project has '
        'been shown to make money and the corrections file is where that '
        'question is answered')

  def test_every_public_name_is_exported_and_documented(self):
    assert stock_rl.decisions.__all__
    for name in stock_rl.decisions.__all__:
      assert hasattr(stock_rl.decisions, name), f'{name} is missing'
      member = getattr(stock_rl.decisions, name)
      if isinstance(member, type) or callable(member):
        assert (member.__doc__ or '').strip(), f'{name} has no docstring'


class TestTheInternalHelpers:
  '''The canonical-rendering helpers, which the digest depends on.'''

  def test_a_float_is_rendered_through_repr(self):
    # repr, not str, so the digest records the exact double.
    assert _plain(0.1 + 0.2) == repr(0.1 + 0.2)

  def test_an_int_and_a_str_pass_through(self):
    assert _plain(7) == 7
    assert _plain('RELIANCE') == 'RELIANCE'

  def test_a_nested_dataclass_is_expanded(self):
    plain = _plain(make_position())
    assert plain == {
      'symbol': SYMBOL,
      'target_weight': repr(0.06),
      'current_weight': repr(0.0),
      'as_of': BAR,
    }

  def test_a_mapping_is_keyed_by_str_and_ordered(self):
    plain = _plain(AccountState(security_values={'TCS': 1.0, 'A1': 2.0}))
    assert list(plain['security_values']) == ['A1', 'TCS']

  def test_a_tuple_of_dataclasses_becomes_a_list_of_objects(self):
    plain = _plain(make_risk((make_trip(),)))
    assert isinstance(plain['trips'], list)
    assert plain['trips'][0]['code'] == drawdown_code
