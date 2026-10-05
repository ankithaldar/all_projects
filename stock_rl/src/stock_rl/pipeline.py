#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''One path from a licensed feed to a number somebody may quote.

Thirty modules in this package are each tested against their own
fixtures. None of those tests proves that the modules fit together, and
the places they fit are exactly where a real defect hides: a panel built
one way and consumed another, two engines measuring the same strategy
and disagreeing, a decision that serialises in the decision layer and
then fails to survive the HTTP layer that ships it. This module is the
seam check. It composes the existing pieces and refuses to reimplement
any of them -- every number here comes out of
:mod:`stock_rl.portfolio`, :mod:`stock_rl.experiment.harness` or
:mod:`stock_rl.rl.train`, and every piece of arithmetic is theirs.

**What it is for, in one line.** :func:`cross_check` returns a data
structure naming every place two independently-tested modules disagree,
and that structure is empty when they agree.

* :func:`load_panel` builds a panel from a feed, honouring the
  calendar and the licence. A symbol whose segment the entitlement does
  not record is **refused**, not assumed to be cash equity: assuming a
  licence nobody showed is the failure that costs money, and
  :meth:`stock_rl.data.vendors.Entitlement.require_symbol` is where
  that refusal already lives.
* :func:`run_baselines` runs all five baselines over one panel through
  one engine, under one specification, and gates the Sharpe through
  :func:`stock_rl.metrics.minimum_backtest_length`.
* :func:`run_experiment` delegates to
  :func:`stock_rl.experiment.harness.run_experiment` and inherits its
  verdict untouched.
* :func:`cross_check` compares the two engines, the panel both paths
  measured, and the decision layer end to end.

**No licensed NSE panel ships with this repository.** There is no data
directory, no CSV and no vendor export anywhere beneath ``stock_rl``, so
nothing here can be run on real NSE prices until one is bought and
placed. :class:`PanelKind` exists so that when it arrives it is labelled
``'vendor'`` on every object it touches, and so that every panel built in
a test is labelled ``'synthetic'``. **A synthetic panel supports no
conclusion about any strategy, any market or any vendor.** What it does
support is the statement that the plumbing is wired correctly, which is
what this module asserts and the whole of what it asserts.
:func:`load_panel` takes ``kind`` as a required keyword with no default
precisely so that a panel cannot be mislabelled by omission, and it
generates no prices of its own: a fixture generator in ``src`` would be
one more thing that looks like data and is not.

**The gate is the existing one, never a second threshold.**
:func:`run_baselines` asks
:func:`stock_rl.metrics.minimum_backtest_length` whether the best of the
five baselines is supportable on the history available, and if it is
not, *every* row reports ``sharpe=None``. Gating one row and quoting
another would be the selection this project exists to refuse: five
baselines on one panel are one comparison, and a Sharpe quoted from a
subset of them is a Sharpe chosen after the fact.

PONYTAIL: the decision seam builds a one-symbol venue map and a kill
switch in a temporary directory, so it is a wiring check rather than a
deployment configuration. Ceiling: the RMS limits and the price band are
scaffolding values, so this says nothing about whether a real venue map
would clear a real order. Upgrade path: pass the account's
:class:`stock_rl.risk.checks.RmsChecker` into :func:`cross_check` once
one exists per Algo ID; the seam reads the limits for nothing else.

PONYTAIL: the engine seam compares ``momentum_ranked`` and nothing else.
Ceiling: :class:`stock_rl.rl.policy.MomentumPolicy` is the only policy in
this repository that wraps a baseline function rather than
reimplementing it, so the other four baselines have no second code path
to be compared against and cannot be cross-checked at all. Upgrade
path: wrap the remaining baselines in policies that delegate the same
way, in :mod:`stock_rl.rl.policy`, and widen this one call.

PONYTAIL: the momentum parameters :func:`cross_check` accepts exist so a
test can perturb one engine, not so a caller can search them. Ceiling:
nothing here is a Deflated Sharpe over a parameter sweep, and a sweep
here would be a multiple-testing problem dressed as tuning. Upgrade
path: none needed -- use :func:`stock_rl.rl.train.random_search`, which
logs every trial, and let :func:`stock_rl.rl.train.sharpe_report`
refuse the result if the history does not support it.

PONYTAIL: the panel identity seam compares content tokens and object
identity at the seams this module owns, which catches a copy that has
drifted between here and either engine. Ceiling: it cannot observe what
happens inside :mod:`stock_rl.experiment.harness` after the call.
Upgrade path: the harness could echo the mapping it was handed onto
:class:`stock_rl.experiment.harness.ArmConfig`, at the cost of every
report carrying a reference to a large object.
'''

from __future__ import annotations

import json
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from stock_rl import baselines
from stock_rl.api import Response
from stock_rl.bars import Bar
from stock_rl.costs import DELIVERY, CostModel
from stock_rl.data.calendar import TradingCalendar
from stock_rl.data.vendors import MarketDataFeed, Segment
from stock_rl.decisions import (
  Decision,
  PositionState,
  RiskState,
  SignalState,
  decide,
  veto_impact,
)
from stock_rl.execution.audit import hash_payload
from stock_rl.execution.sizing import SizingInputs, StockSizer
from stock_rl.experiment import harness
from stock_rl.experiment.harness import ArmConfig, ExperimentReport
from stock_rl.graph.edges import DependencyGraph
from stock_rl.metrics import TRADING_DAYS_PER_YEAR, minimum_backtest_length
from stock_rl.portfolio import run_portfolio
from stock_rl.risk.checks import (
  PriceBand,
  RmsChecker,
  RmsLimits,
  SecurityLimits,
)
from stock_rl.risk.killswitch import KillSwitch
from stock_rl.rl.policy import MomentumPolicy
from stock_rl.rl.train import compare
from stock_rl.sentiment.score import SentimentReading
from stock_rl.weights import (
  accepts_weight_cap,
  applied_weight_cap,
  bind_weight_cap,
)

__all__ = [
  'BaselineRun',
  'BaselineTable',
  'CrossCheckResult',
  'Disagreement',
  'EngineAgreement',
  'Panel',
  'PanelKind',
  'RunSpec',
  'baseline_names',
  'cost_tolerance',
  'cross_check',
  'load_panel',
  'panel_schema',
  'run_baselines',
  'run_experiment',
  'seams_checked',
  'sharpe_tolerance',
]

#: Schema tag carried in every token this module computes, so a token
#: written under a different hashing convention is recognisable as
#: foreign rather than compared against this one as though they matched.
panel_schema = 'stock_rl.pipeline/1'

#: Relative agreement tolerance between the two engines' total cost, as
#: a fraction of the rupee amount itself.
#:
#: **This is a real finding, and it contradicts a claim.**
#: :func:`stock_rl.rl.train.compare` documents that cost "agrees
#: exactly, because it is a sum of rupees over the same fills on the same
#: bars". Measured here, it agrees to about one part in 10**12 of the
#: total, not exactly: over a 400-bar panel at a 300-bar warm-up the two
#: engines reported 7846.813447074086 and 7846.813447074087 rupees, a
#: difference of one unit in the last place. The two books are the same
#: book; the floating-point order the charges are summed in is not. An
#: exact-equality seam would have failed on that panel and reported a
#: licence-scale disagreement that is not there, so this seam compares to
#: a tolerance and says so.
cost_tolerance = 1e-09

#: Agreement tolerance between the two engines' Sharpe, in Sharpe units.
#:
#: Measured as exactly equal on the panel this repository's tests build:
#: :func:`stock_rl.rl.train.compare` aligns the environment's fill bars
#: to :func:`stock_rl.portfolio.run_portfolio`'s, and the policy wraps
#: the same baseline function, so the two books are the same book. The
#: tolerance is therefore floating-point slack and nothing more.
#: :func:`stock_rl.rl.train.compare` names whole-share rounding as the
#: one thing that may legitimately separate the two engines, and
#: changing any single engine parameter moves the Sharpe by orders of
#: magnitude more than this.
sharpe_tolerance = 1e-06

#: Every seam :func:`cross_check` reports on, in the order it checks
#: them. A named tuple rather than a loop over a registry, because the
#: value of this module is that the set of checks is readable in one
#: place and a test can assert it has not shrunk.
seams_checked = (
  'baseline_coverage',
  'panel_identity',
  'parameter_agreement',
  'panel_kind_propagation',
  'engine_agreement',
  'cost_agreement',
  'decision_round_trip',
  'kill_switch_hold',
  'kill_switch_is_load_bearing',
)

#: The five baselines this module compares, written out rather than
#: discovered from :data:`stock_rl.baselines.__all__`. The count is the
#: ``trials`` argument to
#: :func:`stock_rl.metrics.minimum_backtest_length`, so a set discovered
#: at runtime would silently change how long a Sharpe must be believed.
#: A test asserts this equals ``baselines.__all__``, so adding a
#: baseline without adding it here fails rather than drifts.
baseline_names = (
  'buy_and_hold',
  'equal_weight',
  'low_volatility',
  'momentum_ranked',
  'trend_filtered_momentum',
)

#: The reason source a kill-switch veto must be recorded under, read from
#: the decisions module rather than spelled as a bare string here.
kill_switch_source = 'risk.kill_switch'


class PanelKind(StrEnum):
  '''What the prices in a panel are, said out loud on every object.

  Two members, because there are two questions a reader has and the
  second is not implied by the first: are these prices observed, and may
  anything be concluded from them?

  Attributes:
    VENDOR: Bars that came from a feed this project is licensed to use.
      A backtest on these may be quoted, subject to the length gate.
    SYNTHETIC: Generated prices. Every number computed on them is a
      statement about the code and about nothing else. This is the only
      kind any test in this repository may use, because no licensed
      panel is present.
  '''

  VENDOR = 'vendor'
  SYNTHETIC = 'synthetic'


@dataclass(frozen=True, slots=True)
class RunSpec:
  '''The parameters one measuring instrument is configured with.

  Exists so :func:`run_baselines` and :func:`run_experiment` are
  provably configured identically rather than happening to agree: the
  defaults here are the defaults of
  :func:`stock_rl.portfolio.run_portfolio`,
  :class:`stock_rl.experiment.harness.ArmConfig` and
  :class:`stock_rl.api.BacktestRequest`, so an unparameterised call in
  any of the three agrees by construction rather than by coincidence.

  Attributes:
    capital: Starting cash in rupees.
    costs: Transaction cost model, shared by every engine.
    history: Warm-up bars before the first decision.
    max_weight: Per-symbol weight cap.
    rebalance_days: Bars between rebalances.
  '''

  capital: float = 10_000_000.0
  costs: CostModel = DELIVERY
  history: int = 60
  max_weight: float = 0.10
  rebalance_days: int = 21


@dataclass(frozen=True, slots=True)
class Panel:
  '''One aligned, licensed, calendar-checked set of price panels.

  Carries its own provenance in every field a reader needs: which feed
  produced it, under which segments' licences, whether the prices are
  observed, and a content token that changes if any price changes. A
  panel is the thing a number is measured on, so the number and the
  panel travel together -- see :class:`BaselineRun` and
  :class:`EngineAgreement`, which each carry :attr:`kind` and
  :attr:`token` for exactly that reason.

  Attributes:
    panels: Symbol to bars, aligned on one shared timeline. This is the
      exact object handed to every engine; nothing in this module copies
      it, because a copy that has drifted is the defect this module
      exists to find.
    kind: Whether the prices are observed or generated.
    vendor: Feed name, for the audit trail.
    segments: Exchange segments the requested symbols were licensed
      under. Recorded rather than restricted: an unlicensed segment is
      refused outright, and a licensed non-cash segment is carried here
      so a cash-equity caller can see from the panel that it loaded one.
    symbols: Symbols in the panel, ascending.
    bars: Bars per symbol on the shared timeline.
    start: Timestamp of the first shared bar.
    end: Timestamp of the last shared bar.
    dropped_non_trading: Bars discarded because the calendar says the
      date did not trade.
    dropped_outside_session: Bars discarded because the calendar has no
      session covering their time of day.
    dropped_unaligned: Bars discarded because no other symbol carries
      the same timestamp.
    token: Content fingerprint: ``sha256`` over the vendor, the kind and
      every bar's timestamp and prices.
  '''

  panels: dict[str, list[Bar]]
  kind: PanelKind
  vendor: str
  segments: frozenset[Segment]
  symbols: tuple[str, ...]
  bars: int
  start: datetime
  end: datetime
  dropped_non_trading: int = 0
  dropped_outside_session: int = 0
  dropped_unaligned: int = 0
  token: str = ''

  @property
  def is_synthetic(self) -> bool:
    '''Return True when the prices are generated rather than observed.

    Returns:
      ``True`` only for :attr:`PanelKind.SYNTHETIC`.
    '''
    return self.kind is PanelKind.SYNTHETIC

  @property
  def supports_conclusions(self) -> bool:
    '''Return whether anything may be concluded from this panel.

    Returns:
      ``False`` for a synthetic panel, always. A generated price series
      is a fixture: it can show that two modules agree and it can show
      that a gate refuses, and it can show nothing whatever about a
      market.
    '''
    return not self.is_synthetic

  @property
  def conclusion(self) -> str:
    '''Return the sentence a reader must see beside any number.

    Returns:
      One line naming what the panel is and what may be concluded from
        it. Carried on the panel so a caller cannot report a number
        without the caveat being one attribute away.
    '''
    segments = sorted(segment.value for segment in self.segments)
    if self.supports_conclusions:
      return (f'{self.bars} bars of {self.vendor} data over {self.symbols}, '
              f'licensed under {segments}')
    return (f'SYNTHETIC panel: {self.bars} generated bars over '
            f'{self.symbols}. No conclusion about any strategy, any market '
            'or any vendor may be drawn from any number computed on it.')

  @property
  def years(self) -> float:
    '''Return the span of the panel in trading years.

    Returns:
      Bars divided by :data:`stock_rl.metrics.TRADING_DAYS_PER_YEAR`.
      The gate is applied to the *trading window* rather than to this,
      because the warm-up bars carry no decisions and no returns.
    '''
    return self.bars / TRADING_DAYS_PER_YEAR


@dataclass(frozen=True, slots=True)
class BaselineRun:
  '''One baseline's outcome on one panel, with its provenance attached.

  Every number here is a performance figure, so every one of them
  travels with :attr:`panel_kind` and :attr:`panel_token`: a Sharpe that
  does not name its panel is not reportable, and the cheapest way to
  make that true is for the object carrying it to be unable to travel
  alone.

  Attributes:
    name: Baseline name from :data:`baseline_names`.
    panel_kind: Whether the panel is observed or generated.
    panel_token: Content fingerprint of the panel measured.
    sharpe: Annualised Sharpe, or ``None`` when the length gate refuses.
      Never the ungated figure: a module that hands out the number it
      declined to report has not refused anything.
    total_return_multiple: Growth factor. Reported whether or not the
      gate refuses, because a drawdown and a cost are descriptions of
      what happened rather than claims about skill.
    max_drawdown: Deepest peak-to-trough decline, as a fraction.
    turnover: Total absolute weight traded.
    total_cost: Transaction cost charged, in rupees.
    rebalances: Rebalances performed.
    bars: Bars in the trading window the Sharpe would cover.
    funded_symbols: Names carrying a non-zero weight at the last
      rebalance. This is what makes two rows comparable at a glance: a
      baseline that degenerated to equal weight says so here.
    required_years: Years the claimed Sharpe needs.
    available_years: Years of returns the panel actually offers.
    gate_reason: Why the gate did or did not refuse, verbatim.
    cap_bound: Whether the provider was handed the per-symbol cap. False
      for a provider this module cannot introspect, whose book therefore
      carries a cap of its own choosing.
    applied_max_weight: Heaviest single-symbol weight across every
      rebalance, read back off the book's own snapshots rather than
      echoed from :attr:`RunSpec.max_weight`. Equal to the requested cap
      whenever the cap is the binding constraint and lower when it is
      not -- a cap is a ceiling, not a target -- and never above it.
  '''

  name: str
  panel_kind: PanelKind
  panel_token: str
  sharpe: float | None = None
  total_return_multiple: float = 1.0
  max_drawdown: float = 0.0
  turnover: float = 0.0
  total_cost: float = 0.0
  rebalances: int = 0
  bars: int = 0
  funded_symbols: int = 0
  required_years: float = 0.0
  available_years: float = 0.0
  gate_reason: str = ''
  cap_bound: bool = True
  applied_max_weight: float | None = None


@dataclass(frozen=True, slots=True)
class BaselineTable:
  '''Every baseline on one panel under one specification.

  The comparability is the deliverable, so the specification is carried
  here rather than left to the caller's memory: one capital, one cost
  model, one warm-up, one cadence and one cap for all five rows. Two
  rows measured under different parameters are not a comparison.

  Attributes:
    runs: One row per baseline, in :data:`baseline_names` order.
    spec: The shared specification.
    panel_kind: Whether the panel is observed or generated.
    panel_token: Content fingerprint of the panel measured.
    symbols: Symbols every row was measured over.
    panel_bars: Bars per symbol on the shared timeline.
    gate_reason: Why the gate did or did not refuse, verbatim.
    required_years: Years the claimed Sharpe needs.
    available_years: Years of returns available on the shared window.
  '''

  runs: tuple[BaselineRun, ...]
  spec: RunSpec
  panel_kind: PanelKind
  panel_token: str
  symbols: tuple[str, ...] = ()
  panel_bars: int = 0
  gate_reason: str = ''
  required_years: float = 0.0
  available_years: float = 0.0

  @property
  def names(self) -> tuple[str, ...]:
    '''Return the baseline names measured, in order.'''
    return tuple(run.name for run in self.runs)

  @property
  def reportable(self) -> bool:
    '''Return True when a Sharpe may be quoted from this table.

    Returns:
      ``True`` only when every row carries one. The gate is applied to
        the table rather than to a row, so this is all-or-nothing.
    '''
    return bool(self.runs) and all(run.sharpe is not None
                                   for run in self.runs)

  def get(self, name: str) -> BaselineRun:
    '''Return one row by baseline name.

    Args:
      name: Baseline name.

    Returns:
      The matching :class:`BaselineRun`.

    Raises:
      KeyError: If no row carries that name. Raised rather than
        returning a default, because a caller reading a Sharpe out of an
        absent row would be reading nothing at all.
    '''
    for run in self.runs:
      if run.name == name:
        return run
    raise KeyError(
      f'no baseline named {name!r} in this table; measured: {self.names}')

  def to_json(self) -> dict[str, Any]:
    '''Return the table as JSON-safe primitives.

    Returns:
      A mapping, not a string, so it drops straight into
        :meth:`stock_rl.api.Response.of_json` under the API's own
        encoder. Every row carries ``panel_kind`` and ``panel_token``,
        and the payload carries the shared specification, so a reader
        cannot quote a row without the panel and the parameters it came
        from.
    '''
    return {
      'available_years': self.available_years,
      'baselines': [{
        'applied_max_weight': run.applied_max_weight,
        'bars': run.bars,
        'cap_bound': run.cap_bound,
        'funded_symbols': run.funded_symbols,
        'max_drawdown': run.max_drawdown,
        'name': run.name,
        'panel_kind': run.panel_kind.value,
        'panel_token': run.panel_token,
        'rebalances': run.rebalances,
        'required_years': run.required_years,
        'sharpe': run.sharpe,
        'total_cost': run.total_cost,
        'total_return_multiple': run.total_return_multiple,
        'turnover': run.turnover,
      } for run in self.runs],
      'gate_reason': self.gate_reason,
      'panel_bars': self.panel_bars,
      'panel_kind': self.panel_kind.value,
      'panel_token': self.panel_token,
      'reportable': self.reportable,
      'required_years': self.required_years,
      'spec': {
        'capital': self.spec.capital,
        'costs': asdict(self.spec.costs),
        'history': self.spec.history,
        'max_weight': self.spec.max_weight,
        'rebalance_days': self.spec.rebalance_days,
      },
      'symbols': list(self.symbols),
    }


@dataclass(frozen=True, slots=True)
class EngineAgreement:
  '''One strategy measured by both engines, over one panel.

  The measured pair is carried whether or not it agreed, because a cross
  check whose findings are empty tells a reader that something was
  compared and not what. Each side's Sharpe comes with the panel it came
  from, for the same reason as everywhere else in this module.

  Attributes:
    baseline: Strategy name both engines were asked to run.
    panel_kind: Whether the panel is observed or generated.
    panel_token: Content fingerprint of the panel both engines saw.
    portfolio_sharpe: Sharpe from :func:`stock_rl.portfolio.run_portfolio`.
    environment_sharpe: Sharpe from the environment side of
      :func:`stock_rl.rl.train.compare`.
    tolerance: Agreement tolerance applied, in Sharpe units.
    agreed: Whether the two Sharpes are within tolerance.
    portfolio_cost: Total cost charged by
      :func:`stock_rl.portfolio.run_portfolio`, in rupees.
    environment_cost: Total cost charged by the environment, in rupees.
    cost_tolerance: Relative agreement tolerance applied to the two
      costs. See :data:`cost_tolerance` for why this is not equality.
    costs_agree: Whether the two costs are within that tolerance.
    cost_gap: The absolute difference, carried so a caller can see the
      size of what the tolerance allowed rather than only that it passed.
    fingerprint: Deployment fingerprint of the policy the environment
      side measured.
  '''

  baseline: str
  panel_kind: PanelKind
  panel_token: str
  portfolio_sharpe: float
  environment_sharpe: float
  tolerance: float
  agreed: bool
  portfolio_cost: float
  environment_cost: float
  cost_tolerance: float = cost_tolerance
  costs_agree: bool = True
  cost_gap: float = 0.0
  fingerprint: str = ''


@dataclass(frozen=True, slots=True)
class Disagreement:
  '''One named seam where two modules did not agree.

  A structure rather than a log line, so a caller can act on it: filter
  by :attr:`seam`, serialise it, fail a build on it. Every field is
  readable without this module's help, and the panel's identity and kind
  are on it, because a disagreement that does not say which panel it is
  about cannot be reproduced.

  Attributes:
    seam: One of :data:`seams_checked`.
    detail: What was compared and why it matters.
    expected: What the seam requires, as text.
    observed: What was actually found, as text.
    panel_kind: Whether the panel is observed or generated.
    panel_token: Content fingerprint of the panel involved.
  '''

  seam: str
  detail: str
  expected: str
  observed: str
  panel_kind: PanelKind
  panel_token: str


@dataclass(frozen=True, slots=True)
class CrossCheckResult:
  '''What :func:`cross_check` found, as data.

  An empty :attr:`disagreements` means every seam in
  :data:`seams_checked` held. It says nothing about the strategy: the
  panel decides that, and :attr:`panel_kind` says which panel this was.

  Attributes:
    agreements: What the two engines measured, carried whether or not it
      agreed.
    disagreements: One entry per seam that did not hold.
    panel_kind: Whether the panel is observed or generated.
    panel_token: Content fingerprint of the panel every seam used.
    latched_decision: The decision taken with the kill switch tripped.
    armed_decision: The decision taken from identical inputs with the
      switch armed. Carried because a HOLD is only evidence when the
      same inputs would otherwise have traded.
  '''

  agreements: tuple[EngineAgreement, ...]
  disagreements: tuple[Disagreement, ...]
  panel_kind: PanelKind
  panel_token: str
  latched_decision: Decision | None = None
  armed_decision: Decision | None = None

  @property
  def agreed(self) -> bool:
    '''Return True when no seam disagreed.

    Returns:
      ``True`` when :attr:`disagreements` is empty.
    '''
    return not self.disagreements

  def by_seam(self, seam: str) -> tuple[Disagreement, ...]:
    '''Return the disagreements on one named seam.

    Args:
      seam: One of :data:`seams_checked`.

    Returns:
      Matching disagreements, possibly empty. An unknown seam returns
        empty rather than raising, so a caller can ask about a seam this
        version does not have and learn the answer from the emptiness.
    '''
    return tuple(item for item in self.disagreements if item.seam == seam)

  def to_json(self) -> dict[str, Any]:
    '''Return the result as JSON-safe primitives.

    Returns:
      A mapping, not a string, so it drops straight into
        :meth:`stock_rl.api.Response.of_json`. The panel kind and token
        sit at the top level and on every row, and the two decisions are
        carried as their own canonical JSON, so a dashboard cannot render
        one of these without also rendering where it came from.
    '''
    return {
      'agreed': self.agreed,
      'agreements': [{
        'agreed': item.agreed,
        'baseline': item.baseline,
        'cost_gap': item.cost_gap,
        'cost_tolerance': item.cost_tolerance,
        'costs_agree': item.costs_agree,
        'environment_cost': item.environment_cost,
        'environment_sharpe': item.environment_sharpe,
        'fingerprint': item.fingerprint,
        'panel_kind': item.panel_kind.value,
        'panel_token': item.panel_token,
        'portfolio_cost': item.portfolio_cost,
        'portfolio_sharpe': item.portfolio_sharpe,
        'tolerance': item.tolerance,
      } for item in self.agreements],
      'armed_decision': (self.armed_decision.to_json()
                         if self.armed_decision else None),
      'disagreements': [{
        'detail': item.detail,
        'expected': item.expected,
        'observed': item.observed,
        'panel_kind': item.panel_kind.value,
        'panel_token': item.panel_token,
        'seam': item.seam,
      } for item in self.disagreements],
      'latched_decision': (self.latched_decision.to_json()
                           if self.latched_decision else None),
      'panel_kind': self.panel_kind.value,
      'panel_token': self.panel_token,
      'seams_checked': list(seams_checked),
    }


def load_panel(
  feed: MarketDataFeed,
  calendar: TradingCalendar,
  *,
  kind: PanelKind,
  symbols: Iterable[str] | None = None,
  start: datetime | None = None,
  end: datetime | None = None,
) -> Panel:
  '''Build one panel from a feed, under a licence and a calendar.

  Three checks, in this order, because each one can invalidate the
  next. **The licence first**: every requested symbol must be shown to
  be in a segment this feed is licensed for, through
  :meth:`stock_rl.data.vendors.Entitlement.require_symbol`. A symbol the
  entitlement does not classify is refused rather than assumed to be
  cash equity, which is the whole reason that method exists. **The
  calendar second**: a bar stamped on a date the exchange did not trade,
  or at a time no session covers, is dropped and counted, because a bar
  printed on a Sunday is not a price. **Alignment third**: only
  timestamps every symbol carries survive, because
  :func:`stock_rl.portfolio.run_portfolio` refuses unaligned panels and a
  backtest that quietly forward-fills a gap is measuring something the
  market did not do.

  ``kind`` is required and has no default. That is the whole point of
  the parameter: a panel must say whether its prices were observed or
  generated, and a default is a value a caller forgets.

  PONYTAIL: ``start`` and ``end`` are forwarded to the feed verbatim and
  default to the whole history it carries, expressed in the awareness of
  the feed's own timestamps. Ceiling: symbols whose timestamps disagree
  about awareness raise from the comparison inside the feed rather than
  being coerced, because choosing a time zone for a bar is a guess.
  Upgrade path: normalise awareness once at ingestion in
  :mod:`stock_rl.bars`, where the vendor's convention is documented.

  Args:
    feed: Any :class:`stock_rl.data.vendors.MarketDataFeed`. Its
      ``entitlements`` are consulted, not assumed: a feed carrying no
      entitlement record can show no licence for anything, so it is
      refused here rather than trusted to check itself.
    calendar: Trading calendar deciding which dates and times traded.
    kind: Whether the prices are observed or generated. Carried onto
      every object this panel reaches.
    symbols: Symbols to load. Every symbol the feed carries when
      omitted.
    start: First timestamp to include, inclusive.
    end: Last timestamp to include, inclusive.

  Returns:
    A :class:`Panel`.

  Raises:
    ValueError: If no symbol was requested, the feed carries no
      entitlement record, the window is inverted, the feed carries no
      bar to read a bound from, or nothing survives the calendar and the
      alignment. Each message names which check emptied the panel and
      how many bars each one removed.
    KeyError: From the feed, if it does not carry a requested symbol.
    EntitlementError: From the entitlement, if a symbol's segment is
      unrecorded or unlicensed.
  '''
  requested = (tuple(sorted(set(symbols))) if symbols
               else tuple(feed.symbols()))
  if not requested:
    raise ValueError(
      'no symbols were requested and the feed carries none, so there is '
      'no panel to build')
  entitlements = getattr(feed, 'entitlements', None)
  require = getattr(entitlements, 'require_symbol', None)
  if not callable(require):
    raise ValueError(
      f'{feed.vendor_name} carries no entitlement record with a '
      'require_symbol method, so no licence can be shown for any of its '
      'prices; a feed that cannot show a licence serves nothing')
  segments: set[Segment] = set()
  for symbol in requested:
    require(symbol)
    segment = entitlements.segment_of(symbol)
    if segment is None:
      raise ValueError(
        f'{symbol} passed the licence check but carries no segment, so '
        'the panel cannot record what it was licensed under')
    segments.add(segment)
  floor, ceiling = _bounds(feed, requested[0], start, end)
  kept: dict[str, list[Bar]] = {}
  fetched = 0
  dropped_closed = 0
  dropped_timed = 0
  for symbol in requested:
    series = feed.bars(symbol, floor, ceiling)
    fetched += len(series)
    survivors: list[Bar] = []
    for bar in series:
      if not calendar.is_trading_day(bar.timestamp):
        dropped_closed += 1
        continue
      if not calendar.is_market_open(bar.timestamp.date(),
                                     bar.timestamp.time()):
        dropped_timed += 1
        continue
      survivors.append(bar)
    kept[symbol] = survivors
  if not fetched:
    raise ValueError(
      f'{feed.vendor_name} returned no bar at all for {list(requested)}, so '
      f'there is no window to align and no panel to build')
  survivors = sum(len(series) for series in kept.values())
  if not survivors:
    raise ValueError(
      f'not one of the {fetched} bars {feed.vendor_name} returned for '
      f'{list(requested)} survives the calendar: {dropped_closed} fell on '
      f'a date the exchange did not trade and {dropped_timed} at a time '
      f'no session covers')
  shared = _shared_timestamps(kept)
  if not shared:
    raise ValueError(
      'no timestamp is carried by every requested symbol, so the panels '
      'cannot be aligned; run_portfolio refuses unaligned panels and '
      'forward-filling the gap would invent bars that never printed')
  panels: dict[str, list[Bar]] = {}
  dropped_unaligned = 0
  for symbol in requested:
    series = [bar for bar in kept[symbol] if bar.timestamp in shared]
    dropped_unaligned += len(kept[symbol]) - len(series)
    panels[symbol] = series
  return Panel(
    panels=panels,
    kind=kind,
    vendor=feed.vendor_name,
    segments=frozenset(segments),
    symbols=requested,
    bars=len(panels[requested[0]]),
    start=panels[requested[0]][0].timestamp,
    end=panels[requested[0]][-1].timestamp,
    dropped_non_trading=dropped_closed,
    dropped_outside_session=dropped_timed,
    dropped_unaligned=dropped_unaligned,
    token=_panel_token(panels, kind, feed.vendor_name),
  )


def run_baselines(
  panel: Panel,
  spec: RunSpec = RunSpec(),
  names: Sequence[str] | None = None,
) -> BaselineTable:
  '''Run every baseline over one panel through one engine.

  One engine, one specification, one panel for all five rows. That is the
  entire content of this function: five rows measured differently are
  five unrelated backtests, and the comparison the project needs --
  equal weight against everything else, on the identical instrument --
  only exists if the instrument is identical.

  The gate is :func:`stock_rl.metrics.minimum_backtest_length` asked the
  only question worth asking. The claim a reader could make from this
  table is the best of these five, so the gate is asked whether *that*
  is supportable, and if it is not, no row reports a Sharpe. Gating one
  row and quoting another would be selecting a Sharpe after seeing it.

  **The cap is bound to each provider, not just to the engine.**
  :attr:`RunSpec.max_weight` reaches ``run_portfolio`` as its ceiling and
  reaches the provider by keyword through
  :func:`stock_rl.weights.bind_weight_cap`, because every baseline
  carries its own ``max_weight`` default of 0.10. Passing the provider
  unbound made 0.10 the binding constraint under all three settings, so
  ``spec.max_weight`` was decoration on the rows that report it: a caller
  asking for 0.30 was handed a 0.10 book and three different requested
  caps returned one identical Sharpe. Each row therefore also reports
  :func:`stock_rl.weights.applied_weight_cap`, measured off the book's
  own snapshots, so the reported cap can be compared against the applied
  one. See :class:`BaselineRun`.

  Args:
    panel: The panel every row is measured over. Its ``panels`` mapping
      is handed to the engine unchanged; this function never copies it.
    spec: Shared parameters. Defaults are
      :func:`stock_rl.portfolio.run_portfolio`'s own defaults.
    names: Baselines to run, or every name in :data:`baseline_names`.

  Returns:
    A :class:`BaselineTable` carrying the gate's verdict and the panel's
      kind and token on every row.

  Raises:
    ValueError: If no names are given, fewer than two are given (a
      comparison of one strategy supports no claim to deflate against), a
      name is not a baseline, or the engine refuses the panel or the
      specification.
  '''
  chosen = tuple(names) if names else baseline_names
  unknown = [name for name in chosen if name not in baselines.__all__]
  if unknown:
    raise ValueError(
      f'not a baseline in stock_rl.baselines: {unknown}; known: '
      f'{sorted(baselines.__all__)}')
  if len(chosen) < 2:
    raise ValueError(
      f'the length gate needs at least two configurations to deflate '
      f'against, got {list(chosen)}')
  measured: list[tuple[str, Any, Any]] = []
  for name in chosen:
    provider = getattr(baselines, name)
    result = run_portfolio(
      panel.panels,
      # Bound by keyword: the five signatures disagree about where the cap
      # sits, and a positional call has already fed one to a moving
      # average window.
      bind_weight_cap(provider, spec.max_weight),
      capital=spec.capital,
      costs=spec.costs,
      rebalance_days=spec.rebalance_days,
      max_weight=spec.max_weight,
      history=spec.history,
    )
    measured.append((name, result, provider))
  claimed = max(result.sharpe for _, result, _ in measured)
  window = min(len(result.returns) for _, result, _ in measured)
  available = window / TRADING_DAYS_PER_YEAR
  required = (minimum_backtest_length(len(measured), claimed)
              if claimed > 0.0 else 0.0)
  reportable = required <= available and claimed > 0.0
  gate_reason = _gate_reason(len(measured), claimed, required, available)
  runs = tuple(
    BaselineRun(
      name=name,
      panel_kind=panel.kind,
      panel_token=panel.token,
      sharpe=result.sharpe if reportable else None,
      total_return_multiple=result.total_return_multiple,
      max_drawdown=result.max_drawdown,
      turnover=result.turnover,
      total_cost=result.total_cost,
      rebalances=result.rebalances,
      bars=len(result.returns),
      # run_portfolio refuses a window with no room to trade, so a result
      # that came back always rebalanced at least once and snapshots its
      # last target. Indexing rather than guarding: an empty snapshot
      # list would be a change to the engine's contract, and failing
      # loudly beats reporting a funded count of zero for a book that
      # was never built.
      funded_symbols=sum(1 for weight in result.weights[-1].values()
                         if weight > 0.0),
      required_years=required,
      available_years=available,
      gate_reason=gate_reason,
      # Stamped from the provider itself, so a table row cannot claim a
      # cap was applied when the provider never saw one.
      cap_bound=accepts_weight_cap(provider),
      applied_max_weight=applied_weight_cap(result.weights),
    )
    for name, result, provider in measured
  )
  return BaselineTable(
    runs=runs,
    spec=spec,
    panel_kind=panel.kind,
    panel_token=panel.token,
    symbols=panel.symbols,
    panel_bars=panel.bars,
    gate_reason=gate_reason,
    required_years=required,
    available_years=available,
  )


def run_experiment(
  panel: Panel,
  spec: RunSpec | None = None,
  config: ArmConfig | None = None,
  readings: list[SentimentReading] | None = None,
  graph: DependencyGraph | None = None,
  shock: str | None = None,
  llm_scalars: dict[str, float] | None = None,
) -> ExperimentReport:
  '''Run the three-arm experiment over the panel, by delegation.

  Delegates to :func:`stock_rl.experiment.harness.run_experiment` and
  returns its report untouched, verdict and all. The only thing decided
  here is the configuration, and only when the caller did not supply
  one: the shared :class:`RunSpec` is folded into the harness's own
  :class:`stock_rl.experiment.harness.ArmConfig` together with the
  panel's kind, so the arms are measured on the parameters the baselines
  were and the report says on its face which panel produced it.

  A caller-supplied ``config`` wins, including its own ``panel_kind``.
  :func:`cross_check`'s ``panel_kind_propagation`` seam reports it when
  that disagrees with the panel, which is the correct outcome: a report
  that mislabels its data is a defect, and the seam must be able to see
  one rather than being shielded from it.

  Args:
    panel: The panel every arm is measured over. Its ``panels`` mapping
      is handed to the harness unchanged.
    spec: Shared parameters, used only when ``config`` is None.
    config: A complete arm configuration, overriding ``spec``.
    readings: Sentiment readings for the context arms.
    graph: Dependency graph for the graph half of the context block.
    shock: Node key the graph traversal starts from.
    llm_scalars: One injected scalar per symbol. **Required by arm C**,
      and this module calls no model and invents no scalar: a synthetic
      sentiment value would be fabricated evidence presented as an
      experiment input.

  Returns:
    The harness's :class:`stock_rl.experiment.harness.ExperimentReport`.

  Raises:
    ValueError: From the harness, if the panels or the configuration
      cannot support the experiment, or arm C has no injected scalar.
  '''
  chosen = config
  if chosen is None:
    settings = spec or RunSpec()
    chosen = ArmConfig(
      capital=settings.capital,
      costs=settings.costs,
      history=settings.history,
      max_weight=settings.max_weight,
      rebalance_days=settings.rebalance_days,
      panel_kind=panel.kind.value,
    )
  return harness.run_experiment(
    panel.panels, readings, graph, shock, llm_scalars, chosen)


def cross_check(
  panel: Panel,
  report: ExperimentReport,
  spec: RunSpec = RunSpec(),
  lookback: int = 252,
  skip: int = 21,
  top: int = 10,
  symbol: str | None = None,
) -> CrossCheckResult:
  '''Check every seam and return what disagreed, as data.

  Nine seams, named in :data:`seams_checked`, each of which can fail for
  a reason no single module's own tests would catch:

  ``baseline_coverage``
    That all five baselines ran. A table of four is a comparison that
    quietly lost its control arm.
  ``panel_identity``
    That the panel every row was measured on is this panel: same
    symbols, same content token on the table and on every row, and the
    identical mapping object handed to the engines rather than a copy
    that has drifted.
  ``parameter_agreement``
    The harness's configuration against this function's
    :class:`RunSpec`, field by field. A report measured on different
    capital, costs, warm-up, cadence or cap than the baselines is not a
    comparison of anything.
  ``panel_kind_propagation``
    That the panel's kind reaches the baselines' rows, the table and the
    harness report. A synthetic panel that loses its label one layer
    down is the failure this whole module is arranged to prevent.
  ``engine_agreement``
    ``momentum_ranked`` through :func:`stock_rl.portfolio.run_portfolio`
    and through the environment side of
    :func:`stock_rl.rl.train.compare`, over the same panel and the same
    parameters, agreeing on Sharpe to :data:`sharpe_tolerance`. Three
    Sharpes from three code paths would be unfalsifiable; this is what
    makes the two-path claim checkable. **Momentum, and only momentum**,
    because :class:`stock_rl.rl.policy.MomentumPolicy` is the one policy
    in this repository that wraps a baseline function rather than
    reimplementing it, and a policy that reimplemented its strategy
    could not settle this question in either direction.
  ``cost_agreement``
    The same two engines on total cost, to :data:`cost_tolerance` rather
    than to equality. See that constant: the two engines agree to about
    one part in 10**12 and not exactly, so an equality seam would report
    a disagreement that is not one. Checked because a claim in a
    docstring that nothing checks is a comment, and because this one
    turned out to need a tolerance.
  ``decision_round_trip``
    That a decision from :func:`stock_rl.decisions.decide` survives the
    API's own JSON encoder and
    :meth:`stock_rl.decisions.Decision.from_json` byte for byte. A
    record that survives its own encoder and dies in the HTTP one is a
    blank cell in a dashboard and no error anywhere.
  ``kill_switch_hold``
    That a decision taken with a tripped kill switch is a HOLD, commits
    no capital, and names the trip code.
  ``kill_switch_is_load_bearing``
    That the *armed* switch, on identical inputs, produces a different
    action. Without this the HOLD above would prove nothing, because a
    weak signal produces a HOLD too.

  **What this does not prove.** The panel identity seam compares what
  this module holds and what it hands over, which catches a copy that
  drifted on the way to either engine. It cannot observe what happens
  inside :mod:`stock_rl.experiment.harness` after the call; proving the
  harness received the identical mapping object needs a spy at that call
  boundary, which ``tests/test_pipeline.py`` installs.

  Args:
    panel: The panel both paths must have measured.
    report: The harness report over that panel, from
      :func:`run_experiment`.
    spec: The specification both paths must have used.
    lookback: Momentum lookback in bars, applied to both engines.
    skip: Most recent bars excluded from the momentum, both engines.
    top: Names held, both engines. Exposed so a test can perturb one
      engine; it is not a tuning knob.
    symbol: Symbol the decision seams act on. Defaults to the panel's
      first symbol.

  Returns:
    A :class:`CrossCheckResult` whose ``agreed`` is True when every seam
      held.

  Raises:
    ValueError: If the panel's bar timestamps are naive, which no
      decision may be taken from.
  '''
  chosen = symbol or panel.symbols[0]
  disagreements: list[Disagreement] = []
  table = run_baselines(panel, spec)
  for seam in (_coverage_seam(panel, table),
               _identity_seam(panel, table),
               _parameters_seam(panel, table, report, spec),
               _kind_seam(panel, table, report)):
    if seam is not None:
      disagreements.append(seam)
  agreement = _measure_engines(panel, spec, lookback, skip, top)
  if not agreement.agreed:
    gap = abs(agreement.environment_sharpe - agreement.portfolio_sharpe)
    disagreements.append(Disagreement(
      seam='engine_agreement',
      detail=(f'{agreement.baseline} measured by run_portfolio and by the '
              f'RL environment over {panel.symbols}, {panel.kind.value} '
              f'panel {panel.token[:12]}'),
      expected=f'Sharpe within {sharpe_tolerance} of each other',
      observed=(f'run_portfolio {agreement.portfolio_sharpe!r} against '
                f'environment {agreement.environment_sharpe!r}, a gap of '
                f'{gap:.3e}'),
      panel_kind=panel.kind,
      panel_token=panel.token,
    ))
  if not agreement.costs_agree:
    disagreements.append(Disagreement(
      seam='cost_agreement',
      detail=(f'{agreement.baseline} cost charged by run_portfolio against '
              f'the environment, over {panel.symbols}, '
              f'{panel.kind.value} panel {panel.token[:12]}'),
      expected=(f'total cost within {cost_tolerance} of each other, the '
                'two engines trading the same book on the same bars'),
      observed=(f'run_portfolio {agreement.portfolio_cost!r} rupees '
                f'against environment {agreement.environment_cost!r} '
                f'rupees, a gap of {agreement.cost_gap:.3e}'),
      panel_kind=panel.kind,
      panel_token=panel.token,
    ))
  price = panel.panels[chosen][-1].close
  latched, armed = _decision_pair(panel, chosen, price)
  round_trip = _round_trip_seam(panel, latched)
  if round_trip is not None:
    disagreements.append(round_trip)
  veto = _veto_seam(panel, latched)
  if veto is not None:
    disagreements.append(veto)
  if armed.action is latched.action:
    disagreements.append(Disagreement(
      seam='kill_switch_is_load_bearing',
      detail=(f'decision for {chosen} on {panel.kind.value} panel '
              f'{panel.token[:12]}, identical inputs, armed and latched '
              f'switch'),
      expected=('a latched switch changes the action, so the HOLD is '
                'attributable to the switch and not to a weak signal'),
      observed=(f'both states produced {armed.action.value}, so this '
                f'panel proves nothing at all about the kill switch'),
      panel_kind=panel.kind,
      panel_token=panel.token,
    ))
  return CrossCheckResult(
    agreements=(agreement,),
    disagreements=tuple(disagreements),
    panel_kind=panel.kind,
    panel_token=panel.token,
    latched_decision=latched,
    armed_decision=armed,
  )


def _bounds(
  feed: MarketDataFeed,
  symbol: str,
  start: datetime | None,
  end: datetime | None,
) -> tuple[datetime, datetime]:
  '''Return the inclusive window to request from a feed.

  Args:
    feed: The feed being read.
    symbol: A symbol known to be carried, used to read the feed's own
      awareness from its timestamps.
    start: First timestamp wanted, or None for the whole history.
    end: Last timestamp wanted, or None for the whole history.

  Returns:
    Tuple of ``(floor, ceiling)``, both in the awareness of the feed's
      own timestamps, so an unbounded request never compares a naive
      bound against an aware bar.

  Raises:
    ValueError: If the feed carries no bar for ``symbol``, so there is no
      awareness to match, or the requested window is inverted.
  '''
  newest = feed.last_bar_at(symbol)
  if newest is None:
    raise ValueError(
      f'{feed.vendor_name} carries no bar for {symbol}, so its timestamps '
      'cannot be read and no window can be expressed in their awareness')
  floor = (start if start is not None
           else datetime.min.replace(tzinfo=newest.tzinfo))
  ceiling = (end if end is not None
             else datetime.max.replace(tzinfo=newest.tzinfo))
  if floor > ceiling:
    raise ValueError(f'start {floor.isoformat()} is after end '
                     f'{ceiling.isoformat()}')
  return floor, ceiling


def _shared_timestamps(kept: Mapping[str, list[Bar]]) -> frozenset[datetime]:
  '''Return the timestamps every symbol carries.

  Args:
    kept: Calendar-checked bars per symbol.

  Returns:
    The intersection of every symbol's timestamps, empty when there is
      none.
  '''
  shared: frozenset[datetime] | None = None
  for series in kept.values():
    stamps = frozenset(bar.timestamp for bar in series)
    shared = stamps if shared is None else shared & stamps
  return frozenset() if shared is None else shared


def _panel_token(
  panels: Mapping[str, list[Bar]],
  kind: PanelKind,
  vendor: str,
) -> str:
  '''Return the content fingerprint of a panel.

  Hashed through :func:`stock_rl.execution.audit.hash_payload`, the
  project's one canonical-JSON hashing convention, so a panel token and a
  decision fingerprint cannot be two different definitions of "the same
  record". Every price is in the payload, so a copy that has drifted by
  one rupee carries a different token and
  :func:`cross_check`'s identity seam says so.

  Args:
    panels: Aligned bars per symbol.
    kind: The panel's kind, so a relabelled panel is a different panel.
    vendor: The feed name, for the audit trail.

  Returns:
    Lowercase hex digest.
  '''
  return hash_payload({
    'schema': panel_schema,
    'kind': kind.value,
    'vendor': vendor,
    'symbols': {
      symbol: [[bar.timestamp.isoformat(), bar.open, bar.high, bar.low,
                bar.close, bar.volume] for bar in panels[symbol]]
      for symbol in sorted(panels)
    },
  })


def _gate_reason(
  trials: int,
  claimed: float,
  required: float,
  available: float,
) -> str:
  '''Return the sentence the gate stands behind.

  Args:
    trials: Configurations compared.
    claimed: Best Sharpe across them.
    required: Years that Sharpe needs.
    available: Years on offer.

  Returns:
    One line naming what is being claimed, what it needs, what exists,
      or why there is nothing to support.
  '''
  if claimed <= 0.0:
    return (f'none of the {trials} baselines reached a positive Sharpe, so '
            'there is no claim to support and no Sharpe is reportable')
  if required > available:
    return (f'{trials} baselines claiming Sharpe {claimed:.3f} need '
            f'{required:.2f} years of returns and only {available:.2f} are '
            f'available, so stock_rl.metrics.minimum_backtest_length '
            f'refuses every row; gating one and quoting another would be '
            f'selecting a Sharpe after seeing it')
  return (f'{available:.2f} years available against {required:.2f} '
          f'required by stock_rl.metrics.minimum_backtest_length for the '
          f'best of {trials} baselines at Sharpe {claimed:.3f}')


def _coverage_seam(
  panel: Panel,
  table: BaselineTable,
) -> Disagreement | None:
  '''Return a disagreement unless every baseline ran.

  Args:
    panel: The panel measured.
    table: The table produced.

  Returns:
    None when all five baselines are present, else the disagreement.
  '''
  missing = [name for name in baseline_names if name not in table.names]
  if not missing:
    return None
  return Disagreement(
    seam='baseline_coverage',
    detail=(f'baselines run over {panel.symbols}, {panel.kind.value} '
            f'panel {panel.token[:12]}'),
    expected=f'all of {list(baseline_names)} measured on one panel',
    observed=f'missing {missing}; measured {list(table.names)}',
    panel_kind=panel.kind,
    panel_token=panel.token,
  )


def _identity_seam(
  panel: Panel,
  table: BaselineTable,
) -> Disagreement | None:
  '''Return a disagreement unless the table measured this exact panel.

  Compares content rather than trust: the content token on the table and
  on every one of its rows, and the symbol set. A panel copied and
  mutated on the way to the engine carries a different token, and a table
  measured over a different universe names different symbols.

  Args:
    panel: The panel the caller holds.
    table: The table produced from it.

  Returns:
    None when the table names this panel, else the disagreement.
  '''
  mismatched = [
    name for name, token in (
      [('table', table.panel_token)]
      + [(f'row {run.name}', run.panel_token) for run in table.runs]
    ) if token != panel.token
  ]
  if not mismatched and table.symbols == panel.symbols:
    return None
  return Disagreement(
    seam='panel_identity',
    detail=(f'panel handed to the baselines against the panel loaded from '
            f'the feed, {panel.kind.value}'),
    expected=(f'{panel.symbols} over {panel.bars} bars, token '
              f'{panel.token[:12]}, and the identical mapping handed to '
              f'every engine'),
    observed=(f'{list(table.symbols)} over {table.panel_bars} bars, token '
              f'{table.panel_token[:12]}, mismatched on {mismatched}, so a '
              f'copy has drifted or a different panel was measured'),
    panel_kind=panel.kind,
    panel_token=panel.token,
  )


def _parameters_seam(
  panel: Panel,
  table: BaselineTable,
  report: ExperimentReport,
  spec: RunSpec,
) -> Disagreement | None:
  '''Return a disagreement unless both paths used the same parameters.

  Args:
    panel: The panel measured.
    table: The baselines' table, carrying its specification.
    report: The harness report, carrying its own configuration.
    spec: The specification the caller asked for.

  Returns:
    None when every shared parameter agrees, else the disagreement.
  '''
  from_report = RunSpec(
    capital=report.config.capital,
    costs=report.config.costs,
    history=report.config.history,
    max_weight=report.config.max_weight,
    rebalance_days=report.config.rebalance_days,
  )
  if spec == table.spec and spec == from_report:
    return None
  return Disagreement(
    seam='parameter_agreement',
    detail=(f'the parameters the two paths measured over {panel.symbols}, '
            f'{panel.kind.value} panel {panel.token[:12]}'),
    expected=f'baselines and arms both on {spec}',
    observed=f'baselines on {table.spec}, arms on {from_report}',
    panel_kind=panel.kind,
    panel_token=panel.token,
  )


def _kind_seam(
  panel: Panel,
  table: BaselineTable,
  report: ExperimentReport,
) -> Disagreement | None:
  '''Return a disagreement unless the panel's kind reached every layer.

  A panel labelled synthetic in one place and unknown in the next is the
  failure this module is arranged to prevent, so the label is checked at
  every layer it passes through.

  Args:
    panel: The panel loaded.
    table: The baselines' table.
    report: The harness report.

  Returns:
    None when the kind agrees everywhere, else the disagreement.
  '''
  found = {panel.kind.value, table.panel_kind.value, report.panel_kind}
  found.update(run.panel_kind.value for run in table.runs)
  if found == {panel.kind.value}:
    return None
  return Disagreement(
    seam='panel_kind_propagation',
    detail=(f'what the panel is called, at each layer it passes through, '
            f'for token {panel.token[:12]}'),
    expected=(f'{panel.kind.value} on the panel, every row, the table and '
              'the report'),
    observed=(f'{sorted(found)}, so a reader could see a number from this '
              f'panel without seeing that it is generated'),
    panel_kind=panel.kind,
    panel_token=panel.token,
  )


def _measure_engines(
  panel: Panel,
  spec: RunSpec,
  lookback: int,
  skip: int,
  top: int,
) -> EngineAgreement:
  '''Measure momentum through both engines and return the pair.

  The portfolio side is :func:`stock_rl.portfolio.run_portfolio` over
  :attr:`Panel.panels` with :func:`stock_rl.baselines.momentum_ranked`;
  the environment side is :func:`stock_rl.rl.train.compare` over the
  same mapping with :class:`stock_rl.rl.policy.MomentumPolicy`, which
  calls that same baseline function. Same panels, same capital, same
  cost model, same warm-up, same cadence and same cap, so any difference
  in the two Sharpes belongs to the engines rather than to the inputs.

  The knobs are passed **by keyword**, not positionally. Positionally
  they are not interchangeable across this package: the fifth parameter
  of ``momentum_ranked`` is ``max_weight`` and the fifth of
  ``trend_filtered_momentum`` is ``ma_window``, so a positional call
  silently handed the weight cap to a moving-average window and produced
  ``sma(closes, 0.1)`` rather than an error the caller could read.

  Args:
    panel: The panel both engines read.
    spec: The shared measurement parameters.
    lookback: Momentum lookback in bars.
    skip: Most recent bars excluded from the momentum.
    top: Names held.

  Returns:
    An :class:`EngineAgreement` carrying both sides.

  Raises:
    ValueError: If either engine refuses the panel or the parameters,
      including the guard :func:`stock_rl.rl.train.compare` raises when
      the two engines report different windows. That guard is the reason
      this seam is worth running: it is the failure that made the two
      Sharpes incomparable once already.
  '''
  result = run_portfolio(
    panel.panels,
    lambda visible: baselines.momentum_ranked(
      visible, lookback=lookback, skip=skip, top=top,
      max_weight=spec.max_weight),
    capital=spec.capital,
    costs=spec.costs,
    rebalance_days=spec.rebalance_days,
    max_weight=spec.max_weight,
    history=spec.history,
  )
  policy = MomentumPolicy(
    lookback=lookback, skip=skip, top=top, max_weight=spec.max_weight)
  measurement = compare(
    policy,
    panel.panels,
    capital=spec.capital,
    costs=spec.costs,
    history=spec.history,
    max_weight=spec.max_weight,
    rebalance_days=spec.rebalance_days,
  )
  environment = measurement.env
  cost_gap = abs(environment['total_cost'] - result.total_cost)
  return EngineAgreement(
    baseline='momentum_ranked',
    panel_kind=panel.kind,
    panel_token=panel.token,
    portfolio_sharpe=result.sharpe,
    environment_sharpe=environment['sharpe'],
    tolerance=sharpe_tolerance,
    agreed=abs(environment['sharpe'] - result.sharpe) <= sharpe_tolerance,
    portfolio_cost=result.total_cost,
    environment_cost=environment['total_cost'],
    costs_agree=cost_gap <= cost_tolerance * max(1.0, result.total_cost),
    cost_gap=cost_gap,
    fingerprint=measurement.fingerprint,
  )


def _decision_pair(
  panel: Panel,
  symbol: str,
  price: float,
) -> tuple[Decision, Decision]:
  '''Return one latched and one armed decision from identical inputs.

  Both decisions come from :func:`stock_rl.decisions.decide` with the
  same signal, position, price and sizing inputs; only the kill switch
  differs. That is what makes the pair a control rather than two
  unrelated examples: if the armed decision does not trade, the HOLD in
  the latched one is not evidence of anything.

  The switch is a real :class:`stock_rl.risk.killswitch.KillSwitch`,
  tripped through its own automatic drawdown condition and persisted to a
  temporary state file, because the control under test is specifically "a
  latched switch, written to disk by the module that would really trip
  it". The venue map is one symbol wide; see the module docstring for
  what that costs and what to do about it.

  Args:
    panel: The panel the decision bar comes from.
    symbol: Symbol the decision is about.
    price: Price the decision sizes against.

  Returns:
    Tuple of ``(latched, armed)``.

  Raises:
    ValueError: If the panel's timestamps are naive, so no decision may
      be taken from it and no time zone will be invented for one; or if
      the probe switch did not trip, in which case everything downstream
      of it would prove nothing.
  '''
  bar = panel.panels[symbol][-1]
  if bar.timestamp.tzinfo is None:
    raise ValueError(
      f'the last bar of {symbol} is stamped {bar.timestamp.isoformat()} '
      'with no time zone, so no decision may be taken from this panel; '
      'stock_rl.decisions requires an aware decision bar and this module '
      'will not guess the exchange time zone')
  sizer = _probe_sizer(symbol, price)
  signal = SignalState(
    symbol=symbol,
    momentum_rank=3,
    previous_rank=7,
    universe=50,
    momentum=0.184,
    as_of=bar.timestamp,
  )
  position = PositionState(
    symbol=symbol,
    target_weight=0.06,
    current_weight=0.0,
    as_of=bar.timestamp,
  )
  sizing = SizingInputs(prob_win=0.55, win_loss_ratio=1.5, uncertainty=0.5)
  with tempfile.TemporaryDirectory(prefix='stock_rl_probe_') as folder:
    switch = KillSwitch(f'PROBE-{symbol}',
                        path=Path(folder) / f'kill_{symbol}.json')
    armed = decide(signal, position, RiskState((), bar.timestamp), sizer,
                   bar.timestamp, price, sizing=sizing)
    # 30 percent against the switch's own 15 percent limit, so the
    # automatic condition fires rather than the manual path being taken.
    switch.evaluate(drawdown=0.30)
    if not switch.tripped:
      raise ValueError(
        'the probe kill switch did not trip on a 30 percent drawdown '
        'against its own 15 percent limit, so every decision below it '
        'would be taken with an armed switch and the seam would prove '
        'nothing')
    latched = decide(signal, position,
                     RiskState(switch.history, bar.timestamp), sizer,
                     bar.timestamp, price, sizing=sizing)
  return latched, armed


def _probe_sizer(symbol: str, price: float) -> StockSizer:
  '''Return a sizer over a one-symbol venue map, for the decision seams.

  Args:
    symbol: The symbol the venue map covers.
    price: Price the band is built around.

  Returns:
    A :class:`stock_rl.execution.sizing.StockSizer` whose limits are
      scaffolding: wide bands, generous order caps, and capital scaled so
      the sizing rule's own output clears them. Nothing here reads the
      limits for any purpose other than letting
      :func:`stock_rl.decisions.decide` reach its sizing branch at all.
  '''
  capital = 1_000_000.0
  security = SecurityLimits(
    symbol=symbol,
    band=PriceBand(price * 0.8, price * 1.2),
    mwpl=PriceBand(price * 0.1, price * 10.0),
    max_order_quantity=100_000,
    max_order_value=capital,
  )
  limits = RmsLimits(
    cumulative_open_order_value=capital * 5.0,
    max_position=1_000_000,
    max_trading_value=capital * 10.0,
    max_exposure=capital * 10.0,
    max_turnover=capital * 20.0,
    max_security_value=capital * 2.0,
  )
  return StockSizer(RmsChecker(limits, {symbol: security}), capital)


def _round_trip_seam(
  panel: Panel,
  decision: Decision,
) -> Disagreement | None:
  '''Return a disagreement unless the decision survives the API encoder.

  The full path a decision really takes: the decision layer's own
  :meth:`stock_rl.decisions.Decision.to_json`, then
  :meth:`stock_rl.api.Response.of_json` -- which is what actually ships
  it, and which substitutes ``None`` for a non-finite float and sorts its
  keys -- then back through
  :meth:`stock_rl.decisions.Decision.from_json`. A record that survives
  its own encoder and dies in the HTTP one is a blank cell in a
  dashboard and no error anywhere.

  A refused reconstruction is reported rather than raised. A layer that
  trimmed a field would otherwise turn this seam into an exception in the
  middle of a check, which tells the caller nothing about which of the
  nine seams failed.

  Args:
    panel: The panel the decision came from, for the message.
    decision: The decision to round-trip.

  Returns:
    None when the bytes are identical, else the disagreement.
  '''
  original = decision.to_json()
  response = Response.of_json(200, json.loads(original))
  try:
    restored = Decision.from_json(json.loads(response.body.decode('utf-8')))
  except (ValueError, TypeError) as exc:
    return Disagreement(
      seam='decision_round_trip',
      detail=(f'a decision on {panel.kind.value} panel '
              f'{panel.token[:12]} through decisions.to_json, the API JSON '
              f'encoder and decisions.from_json'),
      expected='byte-identical JSON before and after',
      observed=f'the API encoder emitted {response.body!r}, which '
               f'decisions.from_json refuses: {exc}',
      panel_kind=panel.kind,
      panel_token=panel.token,
    )
  if restored.to_json() == original:
    return None
  return Disagreement(
    seam='decision_round_trip',
    detail=(f'a decision on {panel.kind.value} panel {panel.token[:12]} '
            f'through decisions.to_json, the API JSON encoder and '
            f'decisions.from_json'),
    expected='byte-identical JSON before and after',
    observed=f'before {original!r}, after {restored.to_json()!r}',
    panel_kind=panel.kind,
    panel_token=panel.token,
  )


def _veto_seam(
  panel: Panel,
  decision: Decision,
) -> Disagreement | None:
  '''Return a disagreement unless the latched decision is a real HOLD.

  Four things are checked, because a HOLD that gets one of them wrong is
  worse than no HOLD at all: the action, the absence of a size, the
  largest reason naming the switch, and that veto outranking every other
  driver. The size check matters most --
  :class:`stock_rl.decisions.Decision` refuses to construct a HOLD that
  carries one, so a HOLD with a size would mean the invariant was
  bypassed rather than upheld.

  Args:
    panel: The panel the decision came from, for the message.
    decision: The latched decision.

  Returns:
    None when the decision is a proper veto, else the disagreement.
  '''
  largest = decision.largest_reason
  faults = [fault for fault, failed in (
    (f'the action is {decision.action.value}, not hold',
     decision.action.value != 'hold'),
    (f'it commits {dict(decision.sizes)} of capital while halted',
     bool(decision.sizes)),
    (f'the largest reason is {largest.source!r}, not the kill switch',
     largest.source != kill_switch_source),
    (f'the veto carries impact {largest.impact} rather than {veto_impact}, '
     f'so it does not outrank every other driver',
     largest.impact != veto_impact),
    ('no reason names the kill switch at all',
     not any(reason.source == kill_switch_source
             for reason in decision.reasons)),
  ) if failed]
  if not faults:
    return None
  return Disagreement(
    seam='kill_switch_hold',
    detail=(f'the decision taken with the kill switch latched, on '
            f'{panel.kind.value} panel {panel.token[:12]}'),
    expected=('a HOLD committing nothing, whose largest reason names the '
              'kill switch and outranks every driver'),
    observed='; '.join(faults),
    panel_kind=panel.kind,
    panel_token=panel.token,
  )
