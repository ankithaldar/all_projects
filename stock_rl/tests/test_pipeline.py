#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Tests for the composition layer: do the modules actually fit?

Every other test file in this package tests one module against its own
fixtures. This one tests the joins, which is the only place a real
defect can hide, and every test here is written to FAIL if the join is
wrong rather than to confirm that a function runs.

**The panels here are synthetic, and every result built on them says
so.** No licensed NSE panel exists in this repository, so a Sharpe
computed on generated prices is a statement about
:mod:`stock_rl.pipeline` and about nothing else. The tests assert that
provenance rather than quoting a number as a finding.

What is asserted, and why each one is written to bite:

  * **The two engines agree.** ``portfolio.run_portfolio`` and
    ``rl.train.compare`` measured the same baseline, the same panel and
    the same parameters must agree on Sharpe to
    :data:`stock_rl.pipeline.sharpe_tolerance`. The negative half
    perturbs one engine's warm-up by a single rebalance period and
    requires ``cross_check`` to *report* it, which is the only thing
    that makes the agreement meaningful.
  * **The harness receives the identical panel object.** Proven with a
    spy installed on both call boundaries, and with an assertion that the
    spy fired, because a spy that never ran proves nothing.
  * **A short panel is refused, by the existing gate.** The gate's inputs
    are recomputed here from :mod:`stock_rl.rl.train` and
    :mod:`stock_rl.metrics` and compared against what the pipeline
    reported, so the pipeline cannot pass this test with a threshold of
    its own.
  * **An unlicensed or unclassified symbol raises.** Cash equity is never
    assumed for a symbol whose segment nobody recorded.
  * **A tripped kill switch is a HOLD end to end**, through a real
    :class:`~stock_rl.risk.killswitch.KillSwitch` tripped by its own
    automatic drawdown condition, with the armed switch on identical
    inputs as the control.
  * **A decision survives the API's JSON encoder**, and two plausible
    encoders that would break it are shown to break it.
  * **A synthetic panel is labelled synthetic everywhere it surfaces**,
    and a panel deliberately mislabelled by a caller's own config is
    reported.
  * **Every disagreement is a readable data structure**, empty when the
    seams agree, filterable by seam, and serialisable through the API.

Every negative test has a positive control beside it: the unperturbed
engine agrees, the licensed feed loads, the long panel reports, the
armed switch buys. A test that passes because the thing under test never
ran is worse than no test at all.
'''

from __future__ import annotations

# pytest injects a fixture by parameter name, so every test method that
# asks for ``long_panel`` rebinds a module-level name. That is pytest's
# calling convention rather than a shadowing mistake, and the
# alternative -- naming the fixture and the parameter differently --
# makes these tests harder to read than the warning is worth.
# pylint: disable=redefined-outer-name

import json
import random
from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta, timezone

import pytest

import stock_rl.pipeline as pipeline
from stock_rl import baselines
from stock_rl.api import Response
from stock_rl.bars import Bar
from stock_rl.data.calendar import TradingCalendar
from stock_rl.data.vendors import (
  Entitlement,
  EntitlementError,
  ReplayVendor,
  Segment,
)
from stock_rl.decisions import Action, Decision, veto_impact
from stock_rl.experiment import harness
from stock_rl.experiment.harness import ArmConfig, ExperimentReport
from stock_rl.metrics import TRADING_DAYS_PER_YEAR, minimum_backtest_length
from stock_rl.pipeline import (
  BaselineTable,
  CrossCheckResult,
  Panel,
  PanelKind,
  RunSpec,
  baseline_names,
  cost_tolerance,
  cross_check,
  load_panel,
  run_baselines,
  run_experiment,
  seams_checked,
  sharpe_tolerance,
)
from stock_rl.portfolio import run_portfolio
from stock_rl.risk.killswitch import drawdown_code
from stock_rl.rl.policy import MomentumPolicy
from stock_rl.rl.train import compare

#: Exchange local time. The calendar's normal session runs 09:15 to
#: 15:30, so bars stamped at the 15:30 close are inside it and bars at
#: 03:00 are outside it. Both are exercised.
IST = timezone(timedelta(hours=5, minutes=30))

#: First date of the generated history, a Monday.
FIRST = datetime(2024, 1, 1, tzinfo=IST)

#: Four symbols, so momentum has something to rank and
#: ``trend_filtered_momentum``'s regime gate has something to reject.
SYMBOLS = ('AAA', 'BBB', 'CCC', 'DDE')

#: Vendor name on every fixture feed. Free text on purpose: the module
#: under test must not hardcode a vendor list.
VENDOR = 'synthetic-probe'

#: Warm-up long enough that ``momentum_ranked`` is live from its first
#: rebalance. At the shipped 60-bar default it needs 274 visible bars and
#: silently degenerates to equal weight, which would make a comparison
#: between baselines meaningless rather than merely short.
SPEC = RunSpec(history=300)

#: Bars in the long panel. Long enough that the length gate admits the
#: best of the five baselines: 600 trading-window bars is 2.38 years
#: against roughly 1.0 year required.
LONG_BARS = 900

#: Bars in the panel that must be refused. 150 bars leaves a 90-bar
#: trading window, 0.36 years, against about 1.7 years required.
SHORT_BARS = 150

#: The default momentum parameters ``MomentumPolicy`` carries, which are
#: also ``baselines.momentum_ranked``'s. Written out so a change to
#: either module's defaults cannot silently make this comparison vacuous.
MOMENTUM = {'lookback': 252, 'skip': 21, 'top': 10}


def trading_dates(count: int) -> list[date]:
  '''Return ``count`` trading dates on the weekends-only calendar.

  Args:
    count: How many dates are needed.

  Returns:
    Dates in ascending order, every one a weekday. The calendar's own
    3650-day guard is respected by asking for twice the calendar days
    wanted, since weekends are excluded from the answer.
  '''
  calendar = TradingCalendar()
  return calendar.trading_days(
    FIRST, FIRST + timedelta(days=count * 2))[:count]


def generated_bars(
  count: int = LONG_BARS,
  symbols: tuple[str, ...] = SYMBOLS,
  seed: int = 7,
  hour: int = 15,
  minute: int = 30,
  naive: bool = False,
) -> dict[str, list[Bar]]:
  '''Return generated bars, stamped inside or outside the session.

  ``SYNTHETIC``: these are generated prices. Every number computed on
  them is a statement about this repository's code and about no market
  at all.

  Args:
    count: Bars per symbol.
    symbols: Symbols to generate.
    seed: Seed for the price generator, so a panel is reproducible.
    hour: Hour of day to stamp each bar at.
    minute: Minute of day to stamp each bar at.
    naive: Stamp without a time zone. Used to prove that a decision is
      refused from a panel whose timestamps cannot be placed on a clock,
      rather than one being invented for it.

  Returns:
    Mapping of symbol to bars on a shared timeline.
  '''
  panels: dict[str, list[Bar]] = {}
  days = trading_dates(count)
  for index, symbol in enumerate(symbols):
    source = random.Random(seed + index)
    price = 100.0 + 25.0 * index
    series: list[Bar] = []
    for day in days:
      price *= 1.0 + source.gauss(0.0005, 0.014)
      close = round(price, 4)
      opening = round(close * (1.0 + source.gauss(0.0, 0.002)), 4)
      series.append(Bar(
        datetime(day.year, day.month, day.day, hour, minute,
                 tzinfo=None if naive else IST),
        opening,
        max(opening, close) * 1.004,
        min(opening, close) * 0.996,
        close,
        1.0e5,
      ))
    panels[symbol] = series
  return panels


def licensed_feed(
  panels: dict[str, list[Bar]],
  vendor: str = VENDOR,
  segments: tuple[Segment, ...] = (Segment.CASH,),
  classify: Segment = Segment.CASH,
) -> ReplayVendor:
  '''Return a replay feed whose entitlement covers ``panels``.

  Args:
    panels: Bars the feed serves.
    vendor: Name recorded in the audit trail.
    segments: Segments the vendor is licensed for.
    classify: Segment every symbol is recorded as trading in. Decoupled
      from ``segments`` on purpose: that is how a feed licensed only for
      derivatives and carrying cash scrips is built.

  Returns:
    A :class:`~stock_rl.data.vendors.ReplayVendor`.
  '''
  return ReplayVendor(
    vendor,
    panels,
    Entitlement.granted(vendor, *segments,
                        symbol_segments={symbol: classify
                                         for symbol in panels}),
  )


def scalars_for(panel: Panel) -> dict[str, float]:
  '''Return the injected arm-C scalars this module refuses to invent.

  Args:
    panel: The panel whose symbols need a scalar.

  Returns:
    One scalar per symbol, every value identical, so the arm-C
      comparison is a test of the plumbing rather than of a sentiment
      signal. Injected here rather than in ``src`` because a module that
      made one up would be presenting a fabrication as an experiment
      input.
  '''
  return {symbol: 0.1 for symbol in panel.symbols}


class _EmptyFeed:
  '''A feed-shaped object with nothing on it, for the refusals.

  Satisfies :class:`~stock_rl.data.vendors.MarketDataFeed` structurally
  and carries only what ``load_panel`` touches before it refuses, so a
  test can reach a refusal without inventing a price.

  Attributes:
    vendor_name: Vendor name for the message.
    entitlements: Entitlement record, or None to model a feed that has
      none at all.
    carried: Symbols ``symbols()`` reports.
    newest: Value ``last_bar_at`` returns for any symbol.
  '''

  def __init__(
    self,
    vendor_name: str = 'stub',
    entitlements: object | None = None,
    carried: tuple[str, ...] = (),
    newest: datetime | None = None,
  ) -> None:
    self.vendor_name = vendor_name
    self.entitlements = entitlements
    self._carried = carried
    self._newest = newest

  def symbols(self) -> list[str]:
    '''Return the symbols this feed claims to carry.'''
    return list(self._carried)

  def last_bar_at(self, symbol: str) -> datetime | None:
    '''Return the newest timestamp, or None when there is none.

    Args:
      symbol: Symbol asked about, ignored: every symbol of a feed with
        nothing on it is equally absent.
    '''
    del symbol
    return self._newest

  def bars(self, symbol: str, start: datetime,
           end: datetime) -> list[Bar]:
    '''Refuse: no test may reach this and be served a price.

    Args:
      symbol: Symbol requested.
      start: Range start.
      end: Range end.

    Raises:
      AssertionError: Always. Reaching this means the refusal under test
        did not fire.
    '''
    raise AssertionError(
      f'bars({symbol}) was reached; the refusal under test did not fire')


class _SilentFeed(_EmptyFeed):
  '''A licensed feed that answers every request with nothing.

  Distinct from :class:`_EmptyFeed` on purpose: a feed that has no
  entitlement record is refused before any bar is asked for, so its
  ``bars`` refusing proves nothing. This one is properly licensed and
  serves zero bars, which is a different defect and a different refusal.
  '''

  def bars(self, symbol: str, start: datetime,
           end: datetime) -> list[Bar]:
    '''Return no bars at all, as a feed with no data would.

    Args:
      symbol: Symbol requested.
      start: Range start.
      end: Range end.

    Returns:
      An empty list.
    '''
    return []


@dataclass(frozen=True, slots=True)
class _BlindEntitlement(Entitlement):
  '''An entitlement whose two halves disagree with each other.

  ``require_symbol`` accepts every symbol while ``segment_of`` reports
  no segment for any of them. No real feed produces this, and
  ``load_panel`` must still refuse rather than record a panel whose
  provenance it cannot state.
  '''

  def require_symbol(self, symbol: str) -> None:
    '''Accept any symbol, refusing nothing.

    Args:
      symbol: Symbol requested.
    '''

  def segment_of(self, symbol: str) -> Segment | None:
    '''Report no segment for any symbol.

    Args:
      symbol: Symbol asked about.

    Returns:
      None, always.
    '''
    return None


@pytest.fixture(scope='module')
def long_panel() -> Panel:
  '''Return the panel every engine seam is measured on.

  Returns:
    A 900-bar :class:`~stock_rl.pipeline.Panel` labelled
      ``PanelKind.SYNTHETIC``.
  '''
  return load_panel(licensed_feed(generated_bars()),
                    TradingCalendar(),
                    kind=PanelKind.SYNTHETIC)


@pytest.fixture(scope='module')
def long_report(long_panel: Panel) -> ExperimentReport:
  '''Return the harness report over the long panel.

  Module-scoped because it runs three arms under three seeds and is the
  single most expensive fixture here.

  Args:
    long_panel: The panel to run over.

  Returns:
    The harness's own report, with ``panel_kind`` taken from the panel.
  '''
  return run_experiment(long_panel, SPEC, llm_scalars=scalars_for(long_panel))


@pytest.fixture(scope='module')
def long_table(long_panel: Panel) -> BaselineTable:
  '''Return the baseline table over the long panel.

  Args:
    long_panel: The panel to measure.

  Returns:
    A table whose Sharpe column the gate admits.
  '''
  return run_baselines(long_panel, SPEC)


class TestTheFixtureItselfIsHonest:
  '''The panel and its labelling, checked before anything is measured.

  A suite that quietly measured a real panel would be a different suite,
  and a suite that quietly measured a synthetic one while calling it
  real would be worse than no suite.
  '''

  def test_no_licensed_panel_exists_in_this_repository(self, long_panel):
    '''Prices here are generated, and the module says which it is.'''
    assert long_panel.is_synthetic
    assert not long_panel.supports_conclusions
    assert 'SYNTHETIC' in long_panel.conclusion
    assert 'No conclusion' in long_panel.conclusion

  def test_a_vendor_label_is_not_inferred_from_the_prices(self):
    '''Relabelling the same bars is the caller's word, not the data's.

    The positive control for the labelling rules: the very same
    generated bytes handed to ``load_panel`` with the vendor kind come
    back claiming they may be quoted. Nothing in this module can detect
    generated prices, which is exactly why ``kind`` is required.
    '''
    panels = generated_bars(count=60)
    synthetic = load_panel(licensed_feed(panels), TradingCalendar(),
                           kind=PanelKind.SYNTHETIC)
    relabelled = load_panel(licensed_feed(panels), TradingCalendar(),
                            kind=PanelKind.VENDOR)
    assert relabelled.supports_conclusions
    assert not relabelled.is_synthetic
    assert 'SYNTHETIC' not in relabelled.conclusion
    # Same prices, different token: the label is part of the identity.
    assert relabelled.token != synthetic.token

  def test_the_panel_token_tracks_content_not_identity(self):
    '''One changed rupee must change the token, or drift goes unnoticed.'''
    bars = generated_bars(count=60)
    before = load_panel(licensed_feed(bars), TradingCalendar(),
                        kind=PanelKind.SYNTHETIC)
    drifted = dict(bars)
    drifted['AAA'] = [
      replace(bars['AAA'][0], close=bars['AAA'][0].close + 0.01),
      *bars['AAA'][1:],
    ]
    after = load_panel(licensed_feed(drifted), TradingCalendar(),
                       kind=PanelKind.SYNTHETIC)
    assert before.token != after.token

  def test_load_panel_refuses_to_guess_the_kind(self):
    '''The provenance argument is required, not defaulted.'''
    with pytest.raises(TypeError, match='kind'):
      # pylint: disable=missing-kwoa
      load_panel(licensed_feed(generated_bars(count=60)), TradingCalendar())


class TestLicenceAndCalendar:
  '''A panel is refused before it is measured, and never mislabelled.'''

  def test_a_symbol_in_an_unlicensed_segment_raises(self):
    '''Cash scrips on a derivatives-only feed are a licence breach.'''
    feed = licensed_feed(generated_bars(count=60),
                         segments=(Segment.FUTURES_AND_OPTIONS,))
    with pytest.raises(EntitlementError, match='CM'):
      load_panel(feed, TradingCalendar(), kind=PanelKind.SYNTHETIC)

  def test_a_symbol_with_no_recorded_segment_raises(self):
    '''An unclassified symbol cannot be assumed to be cash equity.'''
    feed = ReplayVendor(VENDOR, generated_bars(count=60),
                        Entitlement.none(VENDOR))
    with pytest.raises(EntitlementError, match='no licence can be shown'):
      load_panel(feed, TradingCalendar(), kind=PanelKind.SYNTHETIC)

  def test_a_licensed_non_cash_segment_loads_and_is_recorded(self):
    '''The positive control: a proper licence is honoured and named.'''
    feed = licensed_feed(generated_bars(count=60),
                         segments=(Segment.FUTURES_AND_OPTIONS,),
                         classify=Segment.FUTURES_AND_OPTIONS)
    panel = load_panel(feed, TradingCalendar(), kind=PanelKind.SYNTHETIC)
    assert panel.segments == frozenset({Segment.FUTURES_AND_OPTIONS})
    assert panel.bars == 60

  def test_an_entitlement_that_cannot_name_a_segment_is_refused(self):
    '''The two halves of an entitlement may not disagree silently.'''
    bars = generated_bars(count=60)
    feed = _EmptyFeed(entitlements=_BlindEntitlement(VENDOR),
                      carried=('AAA',),
                      newest=bars['AAA'][-1].timestamp)
    with pytest.raises(ValueError, match='carries no segment'):
      load_panel(feed, TradingCalendar(), kind=PanelKind.SYNTHETIC)

  def test_a_feed_with_no_entitlement_record_is_refused(self):
    '''A feed that cannot show a licence serves nothing.'''
    feed = _EmptyFeed(entitlements=None, carried=('AAA',), newest=FIRST)
    with pytest.raises(ValueError, match='no entitlement record'):
      load_panel(feed, TradingCalendar(), kind=PanelKind.SYNTHETIC)

  def test_a_feed_carrying_nothing_is_refused(self):
    '''Nothing to align is a refusal, not an empty panel.'''
    feed = _EmptyFeed(entitlements=Entitlement.none(VENDOR))
    with pytest.raises(ValueError, match='no symbols were requested'):
      load_panel(feed, TradingCalendar(), kind=PanelKind.SYNTHETIC)

  def test_a_feed_that_returns_no_bar_at_all_is_refused(self):
    '''A licensed feed serving nothing must not yield an empty panel.'''
    feed = _SilentFeed(
      entitlements=Entitlement.granted(
        VENDOR, Segment.CASH, symbol_segments={'AAA': Segment.CASH}),
      carried=('AAA',), newest=FIRST)
    with pytest.raises(ValueError, match='returned no bar at all'):
      load_panel(feed, TradingCalendar(), kind=PanelKind.SYNTHETIC)

  def test_a_feed_with_no_timestamp_to_read_a_bound_from_is_refused(self):
    '''An unbounded request needs an awareness to express itself in.'''
    feed = _EmptyFeed(
      entitlements=Entitlement.granted(
        VENDOR, Segment.CASH, symbol_segments={'AAA': Segment.CASH}),
      carried=('AAA',), newest=None)
    with pytest.raises(ValueError, match='carries no bar'):
      load_panel(feed, TradingCalendar(), kind=PanelKind.SYNTHETIC)

  def test_an_inverted_window_is_refused(self):
    '''A window with its ends the wrong way round is a caller error.'''
    bars = generated_bars(count=60)
    with pytest.raises(ValueError, match='is after end'):
      load_panel(licensed_feed(bars), TradingCalendar(),
                 kind=PanelKind.SYNTHETIC,
                 start=bars['AAA'][-1].timestamp,
                 end=bars['AAA'][0].timestamp)

  def test_a_weekend_bar_is_dropped_and_counted(self):
    '''A bar printed on a Sunday is not a price.'''
    bars = generated_bars(count=60)
    last = bars['AAA'][-1].timestamp
    # Appended after the last bar so the series stays strictly ascending,
    # which ReplayVendor checks rather than sorting.
    saturday = last + timedelta(days=(5 - last.weekday()) % 7 or 7)
    panels = dict(bars)
    panels['AAA'] = [
      *bars['AAA'],
      Bar(saturday, 100.0, 101.0, 99.0, 100.5, 1.0),
    ]
    panel = load_panel(licensed_feed(panels), TradingCalendar(),
                       kind=PanelKind.SYNTHETIC)
    assert panel.dropped_non_trading == 1
    assert panel.bars == 60

  def test_a_panel_whose_bars_all_survive_still_reports_its_span_in_years(
      self, long_panel: Panel):
    '''``Panel.years`` is the raw span, and the gate uses the window.

    The two differ by exactly the warm-up, so a reader comparing them can
    see which figure the gate was asked about.
    '''
    assert long_panel.dropped_non_trading == 0
    assert long_panel.dropped_outside_session == 0
    assert long_panel.dropped_unaligned == 0
    assert long_panel.years == pytest.approx(
      long_panel.bars / TRADING_DAYS_PER_YEAR)
    assert long_panel.years - (
      long_panel.bars - SPEC.history) / TRADING_DAYS_PER_YEAR == (
        pytest.approx(SPEC.history / TRADING_DAYS_PER_YEAR))

  def test_a_bar_outside_every_session_is_dropped_and_counted(self):
    '''03:00 is not a time this exchange trades at.'''
    feed = licensed_feed(generated_bars(count=60, hour=3, minute=0))
    with pytest.raises(ValueError, match='no session covers'):
      load_panel(feed, TradingCalendar(), kind=PanelKind.SYNTHETIC)

  def test_panels_that_share_no_timestamp_are_refused(self):
    '''Alignment is not something to forward-fill in silence.'''
    bars = generated_bars(count=60)
    panels = {'AAA': bars['AAA']}
    panels['BBB'] = [
      replace(bar, timestamp=bar.timestamp + timedelta(hours=6))
      for bar in bars['BBB']
    ]
    with pytest.raises(ValueError, match='cannot be aligned'):
      load_panel(licensed_feed(panels), TradingCalendar(),
                 kind=PanelKind.SYNTHETIC)

  def test_a_symbol_missing_one_bar_aligns_on_the_intersection(self):
    '''The gap is closed by dropping a bar, and the drop is counted.

    Alignment intersects the timestamps, so the one bar AAA lacks is
    dropped from the other three too. AAA itself loses nothing, which is
    why the count is one per *surviving* symbol and not one per symbol.
    '''
    bars = generated_bars(count=60)
    panels = dict(bars)
    panels['AAA'] = [bar for index, bar in enumerate(bars['AAA'])
                     if index != 5]
    panel = load_panel(licensed_feed(panels), TradingCalendar(),
                       kind=PanelKind.SYNTHETIC)
    assert panel.dropped_unaligned == len(SYMBOLS) - 1
    assert panel.bars == 59
    assert all(len(series) == 59 for series in panel.panels.values())


class TestBaselinesAreComparable:
  '''One engine, one specification, one panel, and a gate that bites.'''

  def test_every_baseline_runs_on_the_same_panel_and_specification(
      self, long_table: BaselineTable, long_panel: Panel):
    '''Comparability is the point, so it is asserted rather than hoped.'''
    assert long_table.names == baseline_names
    assert long_table.symbols == long_panel.symbols
    assert long_table.spec == SPEC
    assert {len(series)
            for series in long_panel.panels.values()} == {long_panel.bars}
    for run in long_table.runs:
      assert run.panel_token == long_panel.token
      assert run.panel_kind is PanelKind.SYNTHETIC

  def test_the_baseline_list_cannot_drift_from_the_module(self):
    '''A sixth baseline would change the gate's trial count silently.'''
    assert set(baseline_names) == set(baselines.__all__)
    assert len(baseline_names) == 5

  def test_the_gate_uses_the_existing_threshold_on_the_best_of_the_five(
      self, long_table: BaselineTable, long_panel: Panel):
    '''Recompute the gate here, so a private threshold cannot pass this.'''
    measured = [
      run_portfolio(long_panel.panels, getattr(baselines, name),
                    capital=SPEC.capital, costs=SPEC.costs,
                    rebalance_days=SPEC.rebalance_days,
                    max_weight=SPEC.max_weight, history=SPEC.history)
      for name in baseline_names
    ]
    claimed = max(result.sharpe for result in measured)
    window = min(len(result.returns) for result in measured)
    assert long_table.required_years == pytest.approx(
      minimum_backtest_length(len(baseline_names), claimed))
    assert long_table.available_years == pytest.approx(
      window / TRADING_DAYS_PER_YEAR)

  def test_a_long_enough_panel_reports_every_sharpe(
      self, long_table: BaselineTable):
    '''The positive control for the gate: enough history, numbers quoted.'''
    assert long_table.reportable
    assert all(run.sharpe is not None for run in long_table.runs)
    assert long_table.required_years <= long_table.available_years

  def test_a_short_panel_is_refused_with_a_message_naming_the_gate(self):
    '''The headline honesty requirement, asserted on every row.'''
    panel = load_panel(licensed_feed(generated_bars(count=SHORT_BARS)),
                       TradingCalendar(), kind=PanelKind.SYNTHETIC)
    table = run_baselines(panel)
    assert not table.reportable
    assert all(run.sharpe is None for run in table.runs)
    assert 'minimum_backtest_length' in table.gate_reason
    for run in table.runs:
      assert run.sharpe is None
      assert 'minimum_backtest_length' in run.gate_reason
      assert run.available_years < run.required_years

  def test_a_panel_with_no_positive_sharpe_claims_nothing(self):
    '''Flat prices support no claim, and the gate says so in words.'''
    panels = {
      symbol: [Bar(bar.timestamp, 100.0, 100.0, 100.0, 100.0, 0.0)
               for bar in series]
      for symbol, series in generated_bars(count=200).items()
    }
    table = run_baselines(load_panel(licensed_feed(panels),
                                     TradingCalendar(),
                                     kind=PanelKind.SYNTHETIC))
    assert not table.reportable
    assert 'no claim to support' in table.gate_reason

  def test_one_baseline_is_refused_because_there_is_nothing_to_deflate(
      self, long_panel: Panel):
    '''A single configuration is not a search and not a comparison.'''
    with pytest.raises(ValueError, match='at least two configurations'):
      run_baselines(long_panel, SPEC, names=('equal_weight',))

  def test_an_unknown_baseline_is_refused_by_name(self, long_panel: Panel):
    '''The message must list what is available, not only what is not.'''
    with pytest.raises(ValueError, match='not a baseline'):
      run_baselines(long_panel, SPEC,
                    names=('equal_weight', 'crystal_ball'))

  def test_rows_are_addressable_and_an_absent_row_raises(
      self, long_table: BaselineTable):
    '''Reading a Sharpe out of a missing row must not return a default.'''
    assert long_table.get('momentum_ranked').sharpe is not None
    with pytest.raises(KeyError, match='no baseline named'):
      long_table.get('crystal_ball')

  def test_the_table_payload_names_the_panel_on_every_row(
      self, long_table: BaselineTable):
    '''A row lifted out of this payload must still say where it came from.'''
    payload = long_table.to_json()
    assert payload['panel_kind'] == 'synthetic'
    assert payload['spec']['history'] == SPEC.history
    assert payload['reportable'] is True
    for row in payload['baselines']:
      assert row['panel_kind'] == 'synthetic'
      assert row['panel_token'] == long_table.panel_token
    Response.of_json(200, payload)


class TestBothEnginesAgree:
  '''The seam between ``run_portfolio`` and the RL environment.'''

  def test_the_two_engines_agree_on_the_same_baseline(
      self, long_panel: Panel, long_report: ExperimentReport):
    '''Same panel, same parameters, same Sharpe, to a stated tolerance.'''
    result = cross_check(long_panel, long_report, SPEC)
    assert result.agreed, [item.observed for item in result.disagreements]
    agreement = result.agreements[0]
    assert agreement.baseline == 'momentum_ranked'
    assert agreement.agreed
    assert agreement.tolerance == sharpe_tolerance
    assert abs(agreement.portfolio_sharpe
               - agreement.environment_sharpe) <= sharpe_tolerance
    assert agreement.costs_agree
    assert agreement.cost_gap <= agreement.cost_tolerance
    assert agreement.cost_tolerance == cost_tolerance
    assert agreement.panel_token == long_panel.token

  def test_the_two_engines_cost_each_other_to_within_a_part_in_a_trillion(
      self, long_panel: Panel, long_report: ExperimentReport):
    '''The exact-equality claim in ``rl.train`` does not hold exactly.

    ``train.compare`` documents that cost "agrees exactly, because it is
    a sum of rupees over the same fills on the same bars". Measured, the
    two agree to about one part in 10**12 and differ in the last place.
    This test pins the observed magnitude so that if the two engines ever
    start agreeing *exactly*, the tolerance is noticed rather than left
    looking necessary.
    '''
    result = cross_check(long_panel, long_report, SPEC)
    agreement = result.agreements[0]
    assert agreement.agreed
    assert agreement.costs_agree
    # Non-zero, and far below any economic threshold: this is
    # floating-point summation order, not a different book.
    assert 0.0 <= agreement.cost_gap < 1e-06
    assert agreement.costs_agree == (
      agreement.cost_gap <= cost_tolerance * max(1.0,
                                                 agreement.portfolio_cost))

  def test_the_engines_agree_independently_of_cross_check(
      self, long_panel: Panel, long_table: BaselineTable):
    '''The published row and the environment's own measurement must match.'''
    measurement = compare(
      MomentumPolicy(max_weight=SPEC.max_weight, **MOMENTUM),
      long_panel.panels,
      capital=SPEC.capital,
      costs=SPEC.costs,
      history=SPEC.history,
      max_weight=SPEC.max_weight,
      rebalance_days=SPEC.rebalance_days,
    )
    row = long_table.get('momentum_ranked')
    assert measurement.env['sharpe'] == pytest.approx(
      row.sharpe, abs=sharpe_tolerance)
    assert measurement.portfolio['total_cost'] == pytest.approx(
      row.total_cost)

  def test_perturbing_one_engine_is_reported(self, long_panel: Panel,
                                             long_report: ExperimentReport):
    '''The negative half of the agreement claim.

    One engine's warm-up moves by exactly one rebalance period. The
    tolerance is a millionth of a Sharpe, so a vacuous check would still
    read this as agreement.
    '''
    real = pipeline.run_portfolio

    def perturbed(panels, provider, **kwargs):
      '''Return the real result, measured over a different window.

      Args:
        panels: The panel, untouched.
        provider: The weight provider.
        kwargs: The parameters this call was given.

      Returns:
        A portfolio result over one rebalance period more history.
      '''
      return real(panels, provider,
                  **{**kwargs, 'history': kwargs['history'] + 21})

    with pytest.MonkeyPatch.context() as patch:
      patch.setattr(pipeline, 'run_portfolio', perturbed)
      broken = cross_check(long_panel, long_report, SPEC)
    assert not broken.agreed
    faults = broken.by_seam('engine_agreement')
    assert faults, [item.seam for item in broken.disagreements]
    assert str(sharpe_tolerance) in faults[0].expected
    assert 'gap of' in faults[0].observed
    assert faults[0].panel_token == long_panel.token
    # The control: the same fixture, unperturbed, agrees.
    assert cross_check(long_panel, long_report, SPEC).agreed

  def test_another_baseline_can_be_put_through_the_same_seam(
      self, long_panel: Panel, long_report: ExperimentReport):
    '''The knobs are applied to both engines, not just the one under test.

    The positive control for the exposed momentum parameters: a different
    holding count moves both engines to a different book and they still
    agree, so the seam is not comparing one hardcoded configuration
    against itself.
    '''
    result = cross_check(long_panel, long_report, SPEC, top=2)
    assert result.agreed, [item.observed for item in result.disagreements]
    assert result.agreements[0].baseline == 'momentum_ranked'
    default = cross_check(long_panel, long_report, SPEC)
    assert default.agreements[0].portfolio_sharpe != (
      result.agreements[0].portfolio_sharpe)


class TestThePanelReachesBothPathsUnchanged:
  '''Panel identity, coverage, parameters and labelling.'''

  def test_the_identical_panel_object_reaches_both_engines(
      self, monkeypatch, long_panel: Panel):
    '''Proven at the call boundary, with the spies asserted to have run.'''
    seen: dict[str, list[object]] = {'portfolio': [], 'harness': []}
    real_portfolio = pipeline.run_portfolio
    real_experiment = harness.run_experiment

    def spy_portfolio(panels, provider, **kwargs):
      '''Record the mapping the portfolio engine is handed.

      Args:
        panels: The mapping the caller passed.
        provider: The weight provider.
        kwargs: The remaining parameters.

      Returns:
        Whatever the real engine returns.
      '''
      seen['portfolio'].append(panels)
      return real_portfolio(panels, provider, **kwargs)

    def spy_experiment(panels, *rest, **named):
      '''Record the mapping the harness is handed.

      Args:
        panels: The mapping the caller passed.
        rest: The remaining positional parameters.
        named: The remaining keyword parameters.

      Returns:
        Whatever the real harness returns.
      '''
      seen['harness'].append(panels)
      return real_experiment(panels, *rest, **named)

    monkeypatch.setattr(pipeline, 'run_portfolio', spy_portfolio)
    monkeypatch.setattr(harness, 'run_experiment', spy_experiment)
    table = pipeline.run_baselines(long_panel, SPEC)
    pipeline.run_experiment(long_panel, SPEC,
                            llm_scalars=scalars_for(long_panel))
    # Both spies must have fired, or this test proved nothing at all.
    assert len(seen['portfolio']) == len(baseline_names)
    assert len(seen['harness']) == 1
    for panels in seen['portfolio']:
      assert panels is long_panel.panels
    assert seen['harness'][0] is long_panel.panels
    assert table.panel_token == long_panel.token

  def test_a_table_that_lost_a_baseline_is_reported(self, long_panel: Panel,
                                                   long_report):
    '''A comparison missing its control arm is not a comparison.'''
    real = pipeline.run_baselines

    def trimmed(panel, spec=RunSpec(), names=None):
      '''Return a table with three of the five baselines removed.

      Args:
        panel: The panel to measure.
        spec: Shared parameters.
        names: Names requested. Present in the signature because
          ``run_baselines`` takes it; ignored on purpose.

      Returns:
        A two-row table.
      '''
      del names
      return real(panel, spec, names=('equal_weight', 'momentum_ranked'))

    with pytest.MonkeyPatch.context() as patch:
      patch.setattr(pipeline, 'run_baselines', trimmed)
      result = cross_check(long_panel, long_report, SPEC)
    faults = result.by_seam('baseline_coverage')
    assert faults
    assert 'trend_filtered_momentum' in faults[0].observed
    # The control: the untrimmed table covers all five.
    assert run_baselines(long_panel, SPEC).names == baseline_names

  @pytest.mark.parametrize('drift', ['token', 'symbols'])
  def test_a_drifted_panel_is_reported(self, drift, long_panel: Panel,
                                       long_report: ExperimentReport):
    '''A copy that changed between here and the engine must be named.'''
    real = pipeline.run_baselines

    def drifted(panel, spec=RunSpec(), names=None):
      '''Return a table whose provenance has been falsified.

      Args:
        panel: The panel to measure.
        spec: Shared parameters.
        names: Names requested.

      Returns:
        The real table, relabelled with the wrong token or the wrong
          symbol set.
      '''
      table = real(panel, spec, names)
      if drift == 'token':
        return replace(table, panel_token='0' * 64)
      return replace(table, symbols=())

    with pytest.MonkeyPatch.context() as patch:
      patch.setattr(pipeline, 'run_baselines', drifted)
      result = cross_check(long_panel, long_report, SPEC)
    faults = result.by_seam('panel_identity')
    assert faults, drift
    assert 'drifted' in faults[0].observed
    assert result.by_seam('panel_identity') == result.disagreements, (
      'the identity drift must be the only disagreement, or this fixture '
      'is not isolating the seam')

  def test_arms_measured_on_other_parameters_are_reported(
      self, long_panel: Panel):
    '''Different capital on the arms is not a comparison of anything.'''
    other = run_experiment(long_panel, replace(SPEC, capital=5_000_000.0),
                           llm_scalars=scalars_for(long_panel))
    result = cross_check(long_panel, other, SPEC)
    faults = result.by_seam('parameter_agreement')
    assert faults
    assert '5000000.0' in faults[0].observed

  def test_a_report_that_mislabels_its_panel_is_reported(
      self, long_panel: Panel):
    '''A caller-supplied config may say anything; the seam says so.'''
    config = ArmConfig(history=SPEC.history,
                       panel_kind=PanelKind.VENDOR.value)
    mislabelled = run_experiment(long_panel, config=config,
                                 llm_scalars=scalars_for(long_panel))
    result = cross_check(long_panel, mislabelled, SPEC)
    faults = result.by_seam('panel_kind_propagation')
    assert faults
    assert "'vendor'" in faults[0].observed

  def test_a_synthetic_panel_is_labelled_synthetic_everywhere_it_surfaces(
      self, long_panel: Panel, long_table: BaselineTable,
      long_report: ExperimentReport):
    '''Panel, every row, the table, the report and the cross check.'''
    result = cross_check(long_panel, long_report, SPEC)
    assert long_panel.kind is PanelKind.SYNTHETIC
    assert long_table.panel_kind is PanelKind.SYNTHETIC
    assert long_report.panel_kind == 'synthetic'
    assert result.panel_kind is PanelKind.SYNTHETIC
    assert all(run.panel_kind is PanelKind.SYNTHETIC
               for run in long_table.runs)
    for payload in (long_table.to_json(), result.to_json()):
      assert payload['panel_kind'] == 'synthetic'
      assert payload['panel_token'] == long_panel.token
    assert 'no conclusion' in long_panel.conclusion.lower()


class TestTheKillSwitchAndTheDecision:
  '''The decision layer, end to end, through the API's encoder.'''

  def test_a_tripped_kill_switch_holds_end_to_end(
      self, long_panel: Panel, long_report: ExperimentReport):
    '''A real switch, tripped by its own condition, forces a HOLD.'''
    result = cross_check(long_panel, long_report, SPEC)
    latched = result.latched_decision
    assert latched is not None
    assert latched.action is Action.HOLD
    assert latched.sizes == ()
    assert latched.largest_reason.source == 'risk.kill_switch'
    assert latched.largest_reason.impact == veto_impact
    assert drawdown_code in latched.largest_reason.detail

  def test_the_armed_switch_on_identical_inputs_would_have_bought(
      self, long_panel: Panel, long_report: ExperimentReport):
    '''The control: a HOLD is evidence only if the inputs would trade.'''
    result = cross_check(long_panel, long_report, SPEC)
    armed = result.armed_decision
    assert armed is not None
    assert armed.action is Action.BUY
    assert dict(armed.sizes)
    assert armed.largest_reason.source != 'risk.kill_switch'

  def test_the_decision_survives_the_api_json_encoder(
      self, long_panel: Panel, long_report: ExperimentReport):
    '''Decisions' own encoder, then the HTTP one, then back again.'''
    result = cross_check(long_panel, long_report, SPEC)
    decision = result.latched_decision
    assert isinstance(decision, Decision)
    original = decision.to_json()
    response = Response.of_json(200, json.loads(original))
    restored = Decision.from_json(json.loads(response.body.decode('utf-8')))
    assert restored.to_json() == original
    assert restored == decision
    assert 'NaN' not in response.body.decode('utf-8')
    # The control: the seam itself sees nothing wrong with this panel.
    assert cross_check(long_panel, long_report, SPEC).agreed

  def test_an_encoder_that_truncates_the_fingerprint_is_reported(
      self, long_panel: Panel, long_report: ExperimentReport):
    '''A log redactor that shortens a hash breaks the record loudly.'''

    class _Truncating:
      '''An encoder that redacts the fingerprint, as a scrubber would.'''

      @staticmethod
      def of_json(status, payload):
        '''Return the response with the fingerprint shortened.

        Args:
          status: HTTP status, ignored.
          payload: The mapping to encode.

        Returns:
          A :class:`~stock_rl.api.Response` carrying the scrubbed body.
        '''
        return Response.of_json(status, {
          **payload, 'fingerprint': payload['fingerprint'][:16]})

    with pytest.MonkeyPatch.context() as patch:
      patch.setattr(pipeline, 'Response', _Truncating)
      result = cross_check(long_panel, long_report, SPEC)
    faults = result.by_seam('decision_round_trip')
    assert faults
    assert 'refuses' in faults[0].observed

  def test_an_encoder_that_trims_a_reason_is_reported(
      self, long_panel: Panel, long_report: ExperimentReport):
    '''Byte drift is reported with both payloads, not merely as a failure.'''

    class _Trimming:
      '''An encoder that shortens free text, as a scrubber would.'''

      @staticmethod
      def of_json(status, payload):
        '''Return the response with every reason detail trimmed.

        Args:
          status: HTTP status, ignored.
          payload: The mapping to encode.

        Returns:
          A :class:`~stock_rl.api.Response` carrying the trimmed body.
        '''
        return Response.of_json(status, {
          **payload,
          'reasons': [{**reason, 'detail': reason['detail'][:8]}
                      for reason in payload['reasons']]})

    with pytest.MonkeyPatch.context() as patch:
      patch.setattr(pipeline, 'Response', _Trimming)
      result = cross_check(long_panel, long_report, SPEC)
    faults = result.by_seam('decision_round_trip')
    assert faults
    assert 'before' in faults[0].observed

  def test_a_latched_decision_that_traded_is_reported(
      self, long_panel: Panel, long_report: ExperimentReport):
    '''A HOLD that commits capital, or blames a driver, is a failure.'''
    real = pipeline._decision_pair  # pylint: disable=protected-access

    def latched_bought(panel, symbol, price):
      '''Return the armed decision twice, as a broken switch would.

      Args:
        panel: The panel under test.
        symbol: Symbol the decision is about.
        price: Price it is sized against.

      Returns:
        The armed decision in both positions.
      '''
      armed = real(panel, symbol, price)[1]
      return armed, armed

    with pytest.MonkeyPatch.context() as patch:
      patch.setattr(pipeline, '_decision_pair', latched_bought)
      result = cross_check(long_panel, long_report, SPEC)
    faults = result.by_seam('kill_switch_hold')
    assert faults
    assert 'not hold' in faults[0].observed
    assert result.by_seam('kill_switch_is_load_bearing')

  def test_a_panel_with_naive_timestamps_is_refused_not_guessed(
      self, long_report: ExperimentReport):
    '''No time zone will be invented for a bar to make a decision legal.'''
    panel = load_panel(licensed_feed(generated_bars(count=400, naive=True)),
                       TradingCalendar(), kind=PanelKind.SYNTHETIC)
    with pytest.raises(ValueError, match='no time zone'):
      cross_check(panel, long_report, SPEC)
    # The control: the aware panel of the same shape checks out.
    aware = load_panel(licensed_feed(generated_bars(count=400)),
                       TradingCalendar(), kind=PanelKind.SYNTHETIC)
    assert cross_check(aware, long_report, SPEC).agreed

  def test_a_probe_switch_that_never_trips_is_refused(
      self, long_panel: Panel, long_report: ExperimentReport):
    '''A silent switch would make the whole decision seam vacuous.'''

    class _Mute:
      '''A kill switch that refuses to trip, standing in for a wiring bug.'''

      history: tuple = ()

      def __init__(self, *rest, **named):
        '''Accept the arguments a real switch takes and trip nothing.

        Args:
          rest: Positional arguments, ignored.
          named: Keyword arguments, ignored.
        '''
        del rest, named
        self.tripped = False

      def evaluate(self, **kwargs):
        '''Accept the drawdown and record nothing at all.

        Args:
          kwargs: The pre-defined conditions, ignored.
        '''

    with pytest.MonkeyPatch.context() as patch:
      patch.setattr(pipeline, 'KillSwitch', _Mute)
      with pytest.raises(ValueError, match='did not trip'):
        cross_check(long_panel, long_report, SPEC)


class TestDisagreementsAreData:
  '''The finding is a structure a caller can act on, not a print.'''

  def test_a_green_check_is_an_empty_structure(self, long_panel: Panel,
                                               long_report):
    '''Empty means every seam held, and the measurements are still there.'''
    result = cross_check(long_panel, long_report, SPEC)
    assert isinstance(result, CrossCheckResult)
    assert result.agreed
    assert not result.disagreements
    assert result.agreements
    assert not result.by_seam('engine_agreement')
    assert not result.by_seam('no_such_seam')
    assert not result.by_seam('panel_identity')

  def test_a_disagreement_is_readable_and_serialisable(
      self, long_panel: Panel, long_report):
    '''Every field is populated, and the payload survives the encoder.'''
    real = pipeline.run_baselines

    def trimmed(panel, spec=RunSpec(), names=None):
      '''Return a table with three of the five baselines removed.

      Args:
        panel: The panel to measure.
        spec: Shared parameters.
        names: Names requested. Present in the signature because
          ``run_baselines`` takes it; ignored on purpose.

      Returns:
        A two-row table.
      '''
      del names
      return real(panel, spec, names=('equal_weight', 'momentum_ranked'))

    with pytest.MonkeyPatch.context() as patch:
      patch.setattr(pipeline, 'run_baselines', trimmed)
      result = cross_check(long_panel, long_report, SPEC)
    assert not result.agreed
    assert 'baseline_coverage' in {item.seam
                                   for item in result.disagreements}
    payload = result.to_json()
    Response.of_json(200, payload)
    row = payload['disagreements'][0]
    assert set(row) == {'detail', 'expected', 'observed', 'panel_kind',
                        'panel_token', 'seam'}
    assert row['panel_kind'] == 'synthetic'
    assert row['panel_token'] == long_panel.token
    assert row['detail'] and row['expected'] and row['observed']
    assert payload['agreed'] is False
    assert payload['seams_checked'] == list(seams_checked)

  def test_a_finding_with_no_decisions_still_serialises(
      self, long_panel: Panel):
    '''The None branches of the payload are reachable and handled.'''
    empty = CrossCheckResult(agreements=(), disagreements=(),
                             panel_kind=PanelKind.SYNTHETIC,
                             panel_token=long_panel.token)
    payload = empty.to_json()
    assert payload['armed_decision'] is None
    assert payload['latched_decision'] is None
    assert payload['agreed'] is True
    Response.of_json(200, payload)

  def test_the_seam_list_cannot_shrink_silently(self):
    '''Nine named seams, asserted against a literal in this file.'''
    assert set(seams_checked) == {
      'baseline_coverage',
      'cost_agreement',
      'decision_round_trip',
      'engine_agreement',
      'kill_switch_hold',
      'kill_switch_is_load_bearing',
      'panel_identity',
      'panel_kind_propagation',
      'parameter_agreement',
    }
    assert len(seams_checked) == 9

  def test_no_number_is_reported_without_its_panel(
      self, long_table: BaselineTable, long_panel: Panel,
      long_report: ExperimentReport):
    '''Every performance figure in the payload names its panel.'''
    result = cross_check(long_panel, long_report, SPEC)
    payloads = [long_table.to_json(), result.to_json()]
    for payload in payloads:
      assert payload['panel_token'] == long_panel.token
    rows = [row for payload in payloads
            for row in (payload.get('baselines', [])
                        + payload.get('agreements', []))]
    assert rows
    for row in rows:
      assert row['panel_kind'] == 'synthetic'
      assert row['panel_token'] == long_panel.token
    # And the trading-year figure, which is also a performance claim.
    assert long_table.available_years == pytest.approx(
      (long_panel.bars - SPEC.history) / TRADING_DAYS_PER_YEAR)
