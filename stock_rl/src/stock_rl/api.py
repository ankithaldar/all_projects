#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Read-only HTTP API over the backtest core, standard library only.

Why there is no web framework here
----------------------------------

The design doc proposed FastAPI on uvicorn for inference and risk. This
module refuses both, for reasons that are about correctness rather than
taste.

One: ``http.server.ThreadingHTTPServer`` and ``BaseHTTPRequestHandler``
are already in the standard library, and this API is seven JSON
endpoints plus a document, over structures that already exist in this
repository. A framework would buy schema validation on the wire format;
:func:`_parse_request` does that by hand in about sixty lines, and the
hand-written version can be stricter than a coerced one -- an
unrecognised key is rejected here, because a silently ignored
``max_weight`` is exactly how a backtest ends up reporting performance it
did not earn.

Two: the project carries zero runtime dependencies as a rule rather than
an accident (see ``stock_rl/__init__.py``). A web framework would be the
largest dependency in the repository by an order of magnitude, in a
process that already computes Sharpe ratios by hand.

Three, and decisively: the project's own research concludes that risk and
execution must not be separated by a network boundary, because a
partition between them must fail either open -- unguarded orders reach
the exchange -- or closed, and neither is acceptable. Every additional
process is another boundary that can partition. One process has no such
failure mode by construction, and this module is the same process.

What this API is
----------------

An observation surface, plus one trigger that runs a backtest. It places
no orders, contacts no broker, and can neither trip nor clear the kill
switch. That asymmetry is deliberate. The design doc's kill switch is a
React button, and SEBI CIR/MRD/DP/09/2012 requires a switch that "is
expected to automatically trigger a halt on trading activity based on
pre-defined conditions". A button is a thing an operator has to think to
press, and the failure it exists for is the moment nobody is looking, so
no route here can halt or resume trading.
:mod:`stock_rl.risk.killswitch` is the control; this module reads its
state and cannot write it.

Known gap: no authentication, by deferral rather than by oversight
-------------------------------------------------------------------

There is no auth, no token and no TLS on this server, and that is a real
gap, not a considered posture. Everything the endpoints return is
readable by anyone who can open a socket to the port, which for a
trading desk means the book, the positions and the audit-relevant risk
state.

It is deferred for three specific reasons, and each would be a bad
reason on its own:

  1. Loopback binding is enforced, not merely defaulted (see
     :data:`default_host` and :func:`is_loopback`), so the exposure is
     "anything that can already run code as this user".
  2. SEBI requires the strategy servers to be located in India with "no
     interlink with any system or ID located/linked outside India", so a
     remote identity provider would have to be an Indian one, and picking
     one is a decision for whoever owns the deployment rather than for a
     library module.
  3. An in-process check that nobody has deployed is not a control. A
     half-built auth layer that fails open is worse than an honest
     absence, because it reads as protection.

What replaces it, before this ever listens on anything but loopback: put
a TLS-terminating reverse proxy in front, require a client certificate,
and have the proxy set the Algo ID header that
:mod:`stock_rl.compliance.algo_tag` already exists to record. Until that
exists, the correct reading of this module is "local developer tool", and
``GET /api/health`` says so in its ``advisory`` field.

Signals, honestly labelled
--------------------------

``GET /api/signals`` reports a momentum construction and nothing else,
because the other four families are measured rather than asserted: see
``GET /api/baselines``, which runs every provider in
:mod:`stock_rl.baselines` on the identical backtester, costs and splits.
That endpoint exists so the RL claim is checkable rather than asserted,
and a signal endpoint that quietly omitted the control arm would make it
un-checkable.

The ``confidence`` field is an ordinal midrank percentile of the
cross-sectional momentum score. It is not a calibrated probability and
must not be presented as one: nothing here has been fitted to outcome
frequency, so a "0.9 confidence" means "ranked near the top of this
panel", which is a much weaker statement than "90 percent likely to work".

Every JSON body this module emits carries :data:`not_advice`, error
responses included. A disclaimer attached to the page but not to the
payload stops being a disclaimer the moment someone reads the payload.
'''

from __future__ import annotations

import argparse
import ipaddress
import json
import logging
import math
import socket
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import partial
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs
from typing import Any, Protocol

from stock_rl import __version__, baselines
from stock_rl.bars import Bar, load_csv
from stock_rl.indicators import momentum
from stock_rl.metrics import max_drawdown, sharpe_ratio, total_return
from stock_rl.portfolio import WeightProvider, run_portfolio
from stock_rl.web import index_html, script, stylesheet

__all__ = [
  'ApiHandler',
  'ApiServer',
  'ApiServer6',
  'ApiService',
  'BacktestRequest',
  'BacktestRunner',
  'BadRequest',
  'PortfolioState',
  'ProviderFailed',
  'Refused',
  'Response',
  'Signal',
  'SignalSource',
  'admitted_origin',
  'bound_address',
  'build_server',
  'control_arm',
  'default_host',
  'default_port',
  'dispatch',
  'historical_var',
  'is_loopback',
  'load_panels',
  'main',
  'momentum_signals',
  'not_advice',
  'routes',
  'run_baseline',
  'serve',
  'strategies',
]

_log = logging.getLogger('stock_rl.api')

#: Interface the API binds to. Loopback only, and the check is enforced
#: in :func:`is_loopback` rather than left to the operator's discipline.
#: See the module docstring for why this surface serves trading
#: decisions and audit state.
default_host = '127.0.0.1'

#: Default port. Not a privileged port, so the server can run unprivileged.
default_port = 8765

#: Largest accepted request body, 1 MiB. A backtest trigger is a few
#: hundred bytes; anything larger is a mistake or an attack, and reading
#: it would let one client pin the process.
max_body_bytes = 1 << 20

#: How much of an over-sized body is read and thrown away so the client
#: can finish writing and then read the refusal. Separate from
#: :data:`max_body_bytes` on purpose: the limit says what is accepted,
#: the drain says how politely the rejection is delivered. Answering 413
#: while the peer is still writing drops the connection mid-handshake,
#: and the peer sees a reset instead of the answer.
max_drain_bytes = 8 << 20

#: Bars in the trailing momentum window. 252 NSE trading days, matching
#: :data:`stock_rl.metrics.TRADING_DAYS_PER_YEAR`.
signal_lookback = 252

#: Most recent bars excluded from the momentum window. The 12-1
#: construction of Jegadeesh and Titman, applied because Sehgal and
#: Balakrishnan (2002) found Indian short-horizon returns continue rather
#: than reverse, so an unskipped lookback mixes two opposing effects.
signal_skip = 21

#: Symbols the momentum signal holds when the ranking is usable.
signal_top = 10

#: Cap on any single symbol's weight, matching ``env.nse.MAX_WEIGHT``.
signal_max_weight = 0.10

#: Confidence reported as the historical one-period VaR level.
var_confidence = 0.95

#: Weight difference below which a rebalance is not worth sending.
weight_epsilon = 1e-6

#: Attached to every payload, because a screen that renders Sharpe next
#: to a number without this line is an advertisement.
not_advice = (
  'Research output from a backtested model. Not financial advice, not a '
  'recommendation, and not a solicitation to buy or sell anything.'
)

#: Injected monotonic-seconds source. Tests supply a stub so uptime is a
#: known number and no test waits for time to pass.
Clock = Callable[[], float]

#: Injected wall-clock source, so a recorded timestamp is deterministic.
WallClock = Callable[[], datetime]


class BadRequest(ValueError):
  '''The request could not be understood or was not safe to run.

  Reported as HTTP 400. It covers malformed JSON, a non-object body, an
  unrecognised field and an out-of-range value. All four are client
  errors, and all four are worth a 400 rather than a 500 because a
  traceback in an HTTP response is a leak, not a diagnosis.
  '''


class Refused(ValueError):
  '''The request was well-formed but cannot be run right now.

  Reported as HTTP 422. An empty panel set is the common case: the
  server is healthy, the request was valid, and there is nothing to
  backtest. That is not the same as being asked for something invalid,
  and it is not a server fault either.
  '''


class ProviderFailed(RuntimeError):
  '''A weight provider raised instead of returning weights.

  Reported as HTTP 500, deliberately distinct from :class:`Refused`. A
  422 says "change your request"; a 500 says "this needs an engineer".
  Collapsing the two would have a caller retry a request that was never
  the problem.

  The distinction is enforced at the call site by :func:`_guarded`, which
  wraps the provider, rather than by inspecting the exception where it is
  caught. :func:`stock_rl.portfolio.run_portfolio` raises ``ValueError``
  for its own refusals and lets a provider's exception through
  unchanged, so the two are indistinguishable by type at the catch site;
  a provider raising ``ValueError`` would be reported to the caller as a
  422, which tells them to edit a request that was never the problem.

  PONYTAIL: the catch around a provider is deliberately broad. A
  provider is a callable owned by another module, and one that raises
  anything at all must not take down the endpoint whose whole purpose is
  to measure every provider. Ceiling: the failure is reported, not
  diagnosed -- the message carries the exception repr. Upgrade path:
  narrow the catch to the provider's own error type once ``baselines.py``
  declares one.
  '''


class TripLike(Protocol):
  '''Structural type for one recorded kill-switch trip.

  Declared rather than imported so this module has no import-time
  coupling to :mod:`stock_rl.risk.killswitch`. The two modules are built
  separately, and a dashboard that cannot start because the risk module
  moved is a dashboard nobody uses.
  '''

  code: str
  detail: str
  at: str
  observed: float


class ThresholdsLike(Protocol):
  '''Structural type for the pre-defined trip conditions.'''

  index_fall: float
  max_drawdown: float
  ack_latency: float
  fno_ban: bool


class KillSwitchLike(Protocol):
  '''Structural type for the subset of the kill switch this API reads.

  Read-only by construction: every member here is a getter, so the API
  has no vocabulary for halting or resuming trading even if a future
  change tried to give it one.
  '''

  algo_id: str

  @property
  def tripped(self) -> bool:
    '''Return whether trading is halted.'''

  @property
  def trading_enabled(self) -> bool:
    '''Return whether an order may be released.'''

  @property
  def thresholds(self) -> ThresholdsLike:
    '''Return the pre-defined trip conditions.'''

  def breached(
    self,
    index_change: float,
    drawdown: float,
    fno_banned: bool,
    ack_latency: float | None,
  ) -> tuple[str, ...]:
    '''Return the conditions currently breached, in detail text.'''


#: A backtest trigger. Receives a validated request, returns a
#: JSON-safe mapping. Injected so a test needs neither panels nor a
#: backtest to exercise the HTTP layer.
BacktestRunner = Callable[['BacktestRequest'], Mapping[str, Any]]

#: A signal source. Receives the panels and the held weights, returns one
#: row per symbol. Injected so a test can pin the action vocabulary
#: without constructing price history.
SignalSource = Callable[[Mapping[str, Sequence[Bar]],
                         Mapping[str, float]], list['Signal']]


def _utcnow() -> datetime:
  '''Return the current time as an aware UTC datetime.

  Isolated so the module's only wall-clock read is a single function and
  a test can replace it.

  Returns:
    Timezone-aware datetime in UTC.
  '''
  return datetime.now(timezone.utc)


@dataclass(frozen=True, slots=True)
class Signal:
  '''One per-symbol recommendation, with the evidence behind it.

  Attributes:
    symbol: Venue symbol.
    action: One of ``'BUY'``, ``'HOLD'``, ``'SELL'``.
    confidence: Midrank percentile of the momentum score across the
      panel, in ``[0, 1]``. An ordering, not a probability; see the
      module docstring.
    reasons: Human-readable evidence, one clause per observable fact.
      Never empty, because a recommendation with no stated reason is the
      artefact this project's audit research is against.
    target_weight: Weight the signal asks for.
    held_weight: Weight currently held.
    score: Trailing 12-1 momentum, or None when history is short.
    rank: One-based rank by score, or None when unscorable.
  '''

  symbol: str
  action: str
  confidence: float
  reasons: tuple[str, ...]
  target_weight: float
  held_weight: float
  score: float | None = None
  rank: int | None = None

  def to_json(self) -> dict[str, Any]:
    '''Return the signal as JSON-safe primitives.

    Returns:
      Mapping with string keys and JSON-safe values.
    '''
    return {
      'symbol': self.symbol,
      'action': self.action,
      'confidence': round(self.confidence, 4),
      'reasons': list(self.reasons),
      'target_weight': round(self.target_weight, 6),
      'held_weight': round(self.held_weight, 6),
      'score': None if self.score is None else round(self.score, 6),
      'rank': self.rank,
    }


@dataclass(frozen=True, slots=True)
class BacktestRequest:
  '''A validated backtest trigger.

  Attributes:
    symbols: Symbols to run. Empty means every loaded symbol.
    capital: Starting cash in rupees.
    strategy: Name of a provider in :mod:`stock_rl.baselines`.
    rebalance_days: Bars between rebalances.
    max_weight: Cap on any single symbol's weight.
    history: Bars required before the first rebalance.
  '''

  symbols: tuple[str, ...] = ()
  capital: float = 10_000_000.0
  strategy: str = 'momentum_ranked'
  rebalance_days: int = 21
  max_weight: float = signal_max_weight
  history: int = 60

  def to_json(self) -> dict[str, Any]:
    '''Return the request as JSON-safe primitives.

    Returns:
      Mapping describing what was asked for, echoed in the response so a
      result can be attributed to the parameters that produced it.
    '''
    return {
      'symbols': list(self.symbols),
      'capital': self.capital,
      'strategy': self.strategy,
      'rebalance_days': self.rebalance_days,
      'max_weight': self.max_weight,
      'history': self.history,
    }


#: Configuration ``GET /api/baselines`` measures the control arm under
#: when the caller does not ask for another one.
#:
#: It is a :class:`BacktestRequest` rather than four loose constants so
#: that the object ``run_baseline`` is handed is the very object the
#: endpoint measured with, which is what stops one named strategy
#: carrying two Sharpes on one dashboard. The values are
#: :class:`BacktestRequest`'s own defaults, which are the same numbers
#: :mod:`stock_rl.portfolio`, :mod:`stock_rl.baselines` and
#: :mod:`stock_rl.experiment.harness` already use, so an
#: unparameterised control arm and an unparameterised
#: ``POST /api/backtest`` agree by construction rather than by
#: coincidence.
#:
#: A caller that means to trigger a run at other settings can ask for the
#: arm at those settings too: ``GET /api/baselines?rebalance_days=5``
#: measures under exactly the request the caller will POST, and the
#: payload says which one it used.
#:
#: PONYTAIL: one configuration for every provider, no per-strategy
#: override. Ceiling: a provider whose own paper uses different settings
#: is measured at these settings, so the row is comparable across
#: providers but not identical to that paper. Upgrade path: accept a
#: per-name override mapping validated by :func:`_parse_request`, and
#: echo the override in the row so the disclosure stays complete.
control_arm = BacktestRequest()


@dataclass(slots=True)
class PortfolioState:
  '''The book the read endpoints report.

  Defaults describe an empty portfolio rather than a zero-valued one: a
  fresh server has no positions and no history, and reporting an equity
  curve of ``[1.0]`` for that would be an assertion that performance
  was exactly zero rather than an absence of any.

  Attributes:
    capital: Starting or reference capital in rupees.
    value: Marked value of the book in rupees.
    cash: Uninvested cash in rupees. Only a live book or a backtest's
      implied undeployed capital populates this; the backtester does not
      report a cash balance, and inventing one would be a fabrication.
    weights: Held weight per symbol.
    targets: Target weight per symbol from the last signal or rebalance.
    equity: Equity curve normalised to 1.0.
    returns: Per-bar returns aligned with ``equity``.
    turnover: Total absolute weight traded, as recorded by the backtester.
    total_cost: Total transaction cost in rupees.
  '''

  capital: float = 0.0
  value: float = 0.0
  cash: float = 0.0
  weights: dict[str, float] = field(default_factory=dict)
  targets: dict[str, float] = field(default_factory=dict)
  equity: list[float] = field(default_factory=list)
  returns: list[float] = field(default_factory=list)
  turnover: float = 0.0
  total_cost: float = 0.0

  @property
  def invested(self) -> float:
    '''Return the sum of held weights, i.e. gross exposure.'''
    return sum(abs(weight) for weight in self.weights.values())

  @property
  def drawdown(self) -> float:
    '''Return the deepest drawdown on the equity curve.

    Computed by :func:`stock_rl.metrics.max_drawdown` rather than
    tracked alongside the curve, so the reported number cannot drift
    away from the series it describes.
    '''
    return max_drawdown(self.equity).depth


@dataclass(frozen=True, slots=True)
class Response:
  '''One HTTP response, built without touching a socket.

  Attributes:
    status: HTTP status code.
    body: Encoded response body.
    content_type: Value of the ``Content-Type`` header.
  '''

  status: int
  body: bytes
  content_type: str = 'application/json; charset=utf-8'

  @classmethod
  def of_json(cls, status: int, payload: Any) -> 'Response':
    '''Return a JSON response for ``payload``.

    Non-finite floats become ``null`` rather than the bare ``NaN`` and
    ``Infinity`` tokens Python emits by default. Those tokens are not
    valid JSON, and a browser's ``JSON.parse`` rejects the whole
    response -- so one undefined metric would blank the dashboard rather
    than show one empty cell.

    Args:
      status: HTTP status code.
      payload: Any structure ``json.dumps`` accepts.

    Returns:
      The response, body encoded as UTF-8.
    '''
    # Keys are sorted so two runs over identical state serialise
    # identically, which makes a response diffable in a log.
    text = json.dumps(
      _jsonable(payload), allow_nan=False, sort_keys=True)
    return cls(status, text.encode('utf-8'))

  @classmethod
  def of_html(cls, status: int, document: str) -> 'Response':
    '''Return an HTML response.

    Args:
      status: HTTP status code.
      document: HTML text.

    Returns:
      The response.
    '''
    return cls(
      status, document.encode('utf-8'), 'text/html; charset=utf-8')


class ApiService:
  '''Everything the read endpoints serve, and the seams they read it
  through.

  Three seams exist so that tests neither sleep nor need price history:
  ``clock`` for monotonic seconds, ``now`` for wall-clock stamps, and
  ``backtest_runner`` and ``signal_source`` for the two derived answers.
  Everything else is a plain attribute.

  Args are documented on ``__init__``.
  '''

  def __init__(
    self,
    panels: Mapping[str, Sequence[Bar]] | None = None,
    state: PortfolioState | None = None,
    kill_switch: KillSwitchLike | None = None,
    backtest_runner: BacktestRunner | None = None,
    signal_source: SignalSource | None = None,
    clock: Clock = time.monotonic,
    now: WallClock = _utcnow,
    top: int = signal_top,
  ) -> None:
    '''Build a service.

    Args:
      panels: Price panels keyed by symbol. Copied, so a later mutation
        by the caller cannot change what the API reports between two
        requests without appearing in the audit trail.
      state: The book to report. A fresh empty state by default.
      kill_switch: Read-only handle on a latching kill switch. ``None``
        means "not wired in", which is reported as such rather than as
        a healthy untripped switch: conflating the two would let a
        dashboard show a green light for a control that is not
        connected.
      backtest_runner: Runs a validated request. Defaults to
        :func:`run_baseline`.
      signal_source: Produces signals. Defaults to :func:`momentum_signals`.
      clock: Monotonic-seconds source for uptime.
      now: Wall-clock source for recorded timestamps.
      top: Symbols the default momentum signal holds.

    Raises:
      ValueError: If ``top`` is not positive.
    '''
    if top < 1:
      raise ValueError(f'top must be >= 1, got {top}')
    self.panels: dict[str, list[Bar]] = {
      symbol: list(bars) for symbol, bars in (panels or {}).items()}
    self.state = state if state is not None else PortfolioState()
    self.kill_switch = kill_switch
    self.backtest_runner = backtest_runner or partial(run_baseline, self)
    self.signal_source = signal_source or partial(momentum_signals,
                                                 top=top)
    self.clock = clock
    self.now = now
    self.top = top
    self.started_at = clock()

  @property
  def uptime(self) -> float:
    '''Return seconds since this service was built.

    Non-negative by construction: the clock is read once at
    construction, so a monotonic source can only move it forward. A
    non-monotonic injection is clamped to zero rather than reported
    negative, because a negative uptime is a bug report, not a metric.
    '''
    return max(0.0, self.clock() - self.started_at)

  @property
  def symbols(self) -> list[str]:
    '''Return the loaded symbols, in ascending order.'''
    return sorted(self.panels)

  def panels_for(
    self,
    symbols: Iterable[str] = (),
  ) -> dict[str, list[Bar]]:
    '''Return the subset of panels named by ``symbols``.

    Args:
      symbols: Symbols to keep. Empty keeps every loaded symbol.

    Returns:
      Mapping of symbol to its bars, in ascending symbol order.
    '''
    wanted = tuple(sorted(set(symbols)))
    if not wanted:
      return {symbol: self.panels[symbol] for symbol in self.symbols}
    return {symbol: self.panels[symbol] for symbol in wanted
            if symbol in self.panels}

  def trading_enabled(self) -> bool:
    '''Return whether trading is currently permitted.

    Returns:
      False whenever a kill switch is wired in and latched, True when
      no switch is wired in. The last case is a statement about this
      process's wiring, not about a market: ``GET /api/risk`` names it.
    '''
    if self.kill_switch is None:
      return True
    return bool(self.kill_switch.trading_enabled)

  def health(self) -> dict[str, Any]:
    '''Return the liveness payload.

    Reports what is loaded rather than only that the process is up,
    because "alive" and "useful" are different questions and an operator
    watching a dashboard wants both.

    Returns:
      Mapping with ``status``, ``version``, ``uptime_seconds`` and the
      loaded-universe summary.
    '''
    bars = sum(len(panel) for panel in self.panels.values())
    return {
      'status': 'ok',
      'version': __version__,
      'uptime_seconds': round(self.uptime, 3),
      'symbols': len(self.panels),
      'bars': bars,
      'trading_enabled': self.trading_enabled(),
      'kill_switch': _switch_status(self.kill_switch),
      'not_advice': not_advice,
      'advisory': (
        'Local developer tool with no authentication. Loopback only. '
        'Not financial advice.'),
    }

  def signals(self) -> dict[str, Any]:
    '''Return the per-symbol signal table.

    ``as_of`` names the bar the ranking was measured on, which
    ``generated_at`` does not: the wall clock says when the payload was
    built, and the signal is computed from the whole panel including its
    final bar, so the two are different facts. Without the bar date a
    signal built on a close that has not yet settled is indistinguishable
    from one built on yesterday's, and ``held_weight`` is compared
    against it either way.

    Returns:
      Mapping with the generated timestamp, the bar the ranking was
      measured on, the ranking parameters that produced it, and one row
      per symbol carrying action, confidence and reasons. ``as_of`` is
      None with no panels loaded, because there is no bar to name.
    '''
    rows = self.signal_source(self.panels, self.state.weights)
    return {
      'count': len(rows),
      'generated_at': _stamp(self.now()),
      'as_of': _last_bar(self.panels),
      'trading_enabled': self.trading_enabled(),
      'lookback': signal_lookback,
      'skip': signal_skip,
      'top': self.top,
      'signals': [row.to_json() for row in rows],
      'not_advice': not_advice,
    }

  def equity(self) -> dict[str, Any]:
    '''Return the equity curve and its headline statistics.

    ``sharpe`` and ``max_drawdown`` come from :mod:`stock_rl.metrics` and
    are not recomputed here. A second implementation of either would be
    a second definition of the same word, and the two would disagree
    eventually -- most likely because only one of them got the
    annualisation convention right.

    A metric with no series behind it is ``None``, not its degenerate
    value. :func:`stock_rl.metrics.sharpe_ratio` answers 0.0 for an empty
    return series, ``max_drawdown`` answers 0.0 for an empty curve and
    ``total_return`` answers 1.0, so a book that has never run would
    otherwise report a Sharpe of exactly zero, a drawdown of exactly zero
    and a growth multiple of exactly 1.0 -- byte for byte the numbers a
    real backtest that returned nothing reports, and indistinguishable
    from them. ``PortfolioState`` exists to report absence, so the
    absence has to survive the arithmetic.

    Returns:
      Mapping with the curve, its length, and Sharpe, drawdown, turnover
      and total cost. Sharpe and the growth multiple are None with no
      return series; the drawdown is None with no equity curve.
    '''
    state = self.state
    curve = list(state.equity)
    series = list(state.returns)
    return {
      'points': len(curve),
      'equity': curve,
      'sharpe': sharpe_ratio(series) if series else None,
      'max_drawdown': max_drawdown(curve).depth if curve else None,
      'turnover': state.turnover,
      'total_cost': state.total_cost,
      'total_return_multiple': total_return(series) if series else None,
      'capital': state.capital,
      'final_value': state.value,
      'source': 'backtest' if curve else 'none',
      'not_advice': not_advice,
    }

  def positions(self) -> dict[str, Any]:
    '''Return held and target weights with cash and drawdown.

    Every rupee figure in the payload shares one denominator: the book's
    marked ``value``. They used to share two -- each position was valued
    against ``value`` while cash was a share of ``capital`` -- so on a
    book that rose from ten million to fifty-one million the payload's own
    three numbers disagreed by twenty-five million, and the dashboard
    prints them side by side. Cash is now the residual of ``value`` after
    the holdings, so the rows and the cash always sum to the value.

    Returns:
      Mapping with one row per symbol and the book's cash, exposure and
      drawdown.
    '''
    state = self.state
    total = state.value or state.capital
    rows = []
    for symbol in sorted(set(state.weights) | set(state.targets)):
      weight = float(state.weights.get(symbol, 0.0))
      rows.append({
        'symbol': symbol,
        'weight': weight,
        'target_weight': float(state.targets.get(symbol, 0.0)),
        'value': weight * total,
      })
    held_value = sum(row['value'] for row in rows)
    return {
      'count': len(rows),
      'as_of': _stamp(self.now()),
      'positions': rows,
      # state.cash when a caller supplied one (a live book knows its own
      # cash balance); otherwise the residual, so a backtest's implied
      # undeployed capital is measured against the same value the rows are.
      'cash': state.cash if state.cash else total - held_value,
      'capital': state.capital,
      'value': state.value,
      'invested': state.invested,
      'drawdown': state.drawdown,
      'total_cost': state.total_cost,
      'not_advice': not_advice,
    }

  def baselines(self, request: BacktestRequest | None = None) -> dict[str, Any]:
    '''Return every baseline's metrics on the identical backtester.

    The endpoint exists so the learned policy's claim is checkable. A
    strategy compared against nothing is an assertion; compared against
    equal weight, buy and hold, momentum, low volatility and trend-filtered
    momentum on one backtester with one cost model, it is a measurement.
    Equal weight is the control DeMiguel, Garlappi and Uppal (2009) found
    no optimisation reliably beating, so a row that cannot beat it has
    not demonstrated anything.

    The control arm is measured under one declared configuration, and
    that configuration is echoed in the payload. Two rows carrying the
    same strategy name and two different Sharpes on one dashboard are
    the finding: a reader cannot attribute a number to the settings that
    produced it, and the default configuration used to be the library's
    own, which no caller had asked for and no payload disclosed. A
    caller that wants the arm measured at other settings passes a
    request, and then ``POST /api/backtest`` under those same settings
    reproduces the row exactly.

    Args:
      request: Configuration to measure at, or None for
        :data:`control_arm`.

    Returns:
      Mapping with one row per provider in :mod:`stock_rl.baselines`,
      each carrying Sharpe, drawdown, turnover and cost, or the reason it
      could not run. The settings the rows were measured under are
      disclosed at the top level as well as under ``request``.
    '''
    settings = request or control_arm
    rows = [self._baseline_row(name, provider, settings)
            for name, provider in strategies().items()]
    scored = [row for row in rows if row['status'] == 'ok']
    best = max(scored, key=lambda row: row['sharpe']) if scored else None
    return {
      'count': len(rows),
      'strategies': rows,
      'best_sharpe': None if best is None else best['name'],
      'symbols': len(self.panels),
      'capital': settings.capital,
      'rebalance_days': settings.rebalance_days,
      'max_weight': settings.max_weight,
      'history': settings.history,
      'request': settings.to_json(),
      'not_advice': not_advice,
    }

  def _baseline_row(
    self,
    name: str,
    provider: WeightProvider,
    request: BacktestRequest | None = None,
  ) -> dict[str, Any]:
    '''Run one baseline and return its row, or the reason it failed.

    A provider that cannot run -- most often because there are no panels
    -- gets a ``skipped`` row rather than being dropped from the table.
    An absent row is indistinguishable from an untried one, and an
    untried control arm is the failure mode this endpoint is built to
    prevent.

    ``skipped`` and ``error`` are therefore different claims and are not
    interchangeable. ``skipped`` says the backtester refused, which is a
    property of the data and would happen again identically.
    ``error`` says a provider raised, which is a defect in another
    module. Filing a crash as ``skipped`` tells an operator the control
    arm was never attempted, when in fact it was attempted and blew up.

    ``request`` carries the configuration the row is measured under, and
    it is the same object ``run_baseline`` is handed by
    ``POST /api/backtest``, so one named strategy cannot carry two
    Sharpes for one set of settings.

    Args:
      name: Provider name as registered.
      provider: The weight provider to run.
      request: Configuration to measure at, or None for
        :data:`control_arm`.

    Returns:
      Mapping with the strategy name and either its metrics or an error.
    '''
    settings = request or control_arm
    row: dict[str, Any] = {'name': name}
    panels = self.panels_for(settings.symbols)
    if not panels:
      row['status'] = 'skipped'
      # Named apart from "no symbols matched", because the two are
      # different operator problems: one is an empty server, the other is
      # a request naming symbols this server never loaded.
      row['error'] = (
        'no price panels matched the requested symbols'
        if settings.symbols else 'no price panels are loaded')
      return row
    try:
      result = run_portfolio(
        panels,
        _guarded(name, provider),
        capital=settings.capital,
        rebalance_days=settings.rebalance_days,
        max_weight=settings.max_weight,
        history=settings.history,
      )
    except ValueError as exc:
      # Only the backtester's own refusals reach here as ValueError: a
      # provider is wrapped by _guarded, so its exceptions arrive as
      # ProviderFailed and are filed as an error below.
      row['status'] = 'skipped'
      row['error'] = str(exc)
      return row
    except Exception as exc:  # pylint: disable=broad-exception-caught
      # One provider misbehaving must not hide the other four rows: this
      # endpoint exists so the control arm is complete, and a missing
      # row is indistinguishable from an untried one.
      row['status'] = 'error'
      row['error'] = repr(exc)
      return row
    row.update({
      'status': 'ok',
      'sharpe': result.sharpe,
      'max_drawdown': result.max_drawdown,
      'turnover': result.turnover,
      'total_cost': result.total_cost,
      'total_return_multiple': result.total_return_multiple,
      'rebalances': result.rebalances,
      'points': len(result.equity),
    })
    return row

  def risk(self) -> dict[str, Any]:
    '''Return kill-switch state, one-period VaR and breached conditions.

    The ``breached`` list is the operator's answer to "why will this not
    clear", and it is computed by asking the switch with live metrics
    rather than by re-deriving the thresholds here. Duplicating the
    threshold comparison would create a second definition of a safety
    condition, which is precisely the kind of drift a kill switch cannot
    tolerate.

    Every read of the switch is contained, including the ones that fail
    with a type no clause names: a state file on a dead mount raises
    ``OSError``, and a ``/api/risk`` that raised would leave
    ``/api/health`` able to print ``TRIPPED`` with nothing on the
    dashboard repeating it. A switch that cannot be read is reported as
    unreadable, never as untripped.

    Returns:
      Mapping with the switch state, the VaR estimate, the breached
      conditions and the pre-defined limits. The VaR fraction and rupees
      are None with no return history behind them.
    '''
    state = self.state
    fraction, observed = historical_var(state.returns)
    exposure = state.value or state.capital
    return {
      'as_of': _stamp(self.now()),
      'kill_switch': _switch_view(self.kill_switch, state.drawdown),
      'var': {
        'confidence': var_confidence,
        'one_period_fraction': fraction,
        # None rather than 0.0 with no returns behind it: zero rupees of
        # VaR on a book that has never run is a claim of exactly zero
        # risk, which is the one thing this module does not know.
        'rupees': None if fraction is None else fraction * exposure,
        'observations': observed,
        'method': 'historical_percentile',
        'note': (
          'Historical percentile of realised per-bar returns. VaR is '
          'not among the 16 pre-trade RMS checks NSE 11.1 lists, and '
          'post-trade surveillance is the exchange obligation, so this '
          'is reported as context rather than as a control.'),
      },
      'breaches': _breached(self.kill_switch, state.drawdown),
      'thresholds': _thresholds(self.kill_switch),
      'drawdown': state.drawdown,
      'total_cost': state.total_cost,
      'not_advice': not_advice,
    }

  def backtest(self, request: BacktestRequest) -> dict[str, Any]:
    '''Run a backtest and fold the result into the reported book.

    Updating the state is what makes the dashboard useful: ``POST
    /api/backtest`` is the only write here, and it writes observation
    state, not a position. A run over different symbols replaces the
    reported weights rather than blending into them, so two backtests can
    never be mistaken for one portfolio.

    Args:
      request: Validated trigger.

    Returns:
      JSON-safe mapping with the request echoed, the equity curve and the
      headline metrics.

    Raises:
      Refused: If the runner cannot run the request.
    '''
    payload = dict(self.backtest_runner(request))
    payload.setdefault('request', request.to_json())
    self._absorb(payload, request)
    return payload

  def _absorb(self, payload: Mapping[str, Any],
              request: BacktestRequest) -> None:
    '''Copy the parts of a result the read endpoints report.

    ``cash`` is the undeployed share of the book's *value*, not a
    marked-to-market cash balance: the cross-sectional backtester reports
    a cost and a target, and the difference between those and a broker's
    cash position is exactly the kind of detail worth not inventing. It
    is a share of value rather than of capital for the same reason every
    other rupee figure in :meth:`positions` is -- on a book that rose
    from ten million to fifty-one million, a cash balance that is a share
    of capital and holdings that are a share of value do not add up to
    the value they are printed beside.

    Args:
      payload: Result mapping from the runner.
      request: The request that produced it.
    '''
    equity = _floats(payload.get('equity'))
    returns = _floats(payload.get('returns'))
    weights = _weights(payload.get('weights'))
    self.state.capital = request.capital
    self.state.equity = equity
    self.state.returns = returns
    self.state.turnover = _number(payload.get('turnover'), 0.0)
    self.state.total_cost = _number(payload.get('total_cost'), 0.0)
    self.state.targets = dict(weights)
    self.state.weights = {key: value for key, value in weights.items()
                          if value != 0.0}
    self.state.value = request.capital * (
      equity[-1] if equity else 1.0)
    self.state.cash = self.state.value * (1.0 - self.state.invested)


def _guarded(name: str, provider: WeightProvider) -> WeightProvider:
  '''Return ``provider`` wrapped so its own failures are distinguishable.

  :func:`stock_rl.portfolio.run_portfolio` raises ``ValueError`` for its
  own refusals -- unaligned panels, a history that leaves no room -- and
  lets whatever a provider raises pass straight through. That makes a
  ``ValueError`` ambiguous at every catch site above it: an off-by-one
  slice inside a provider is not a bad request, and reporting it as a 422
  tells the caller to change a request that was never the problem.

  The distinction cannot be recovered afterwards from the exception type
  alone, so it is made here, at the only point where the two are still
  separable: the wrapper is this module's, the exception it raises is
  :class:`ProviderFailed`, and a refusal from the backtester itself still
  arrives as the ``ValueError`` it always was.

  PONYTAIL: the wrapper converts rather than diagnoses. Ceiling: the 500
  body carries the exception repr and nothing about where inside the
  provider it came from. Upgrade path: let ``baselines.py`` declare its
  own error type and narrow this wrapper to it, so a genuine
  ``ValueError`` from a provider is reported as itself.

  Args:
    name: Provider name, for the message.
    provider: The callable to wrap.

  Returns:
    A callable with the same signature that reports a provider failure as
    :class:`ProviderFailed`.
  '''
  def guarded(visible: Mapping[str, Sequence[Bar]]) -> dict[str, float]:
    '''Return the provider's weights, or raise :class:`ProviderFailed`.

    Args:
      visible: Price panels visible at the decision bar.

    Returns:
      Target weight per symbol.

    Raises:
      ProviderFailed: Whatever the provider raised, renamed.
    '''
    try:
      return provider(visible)
    except Exception as exc:  # pylint: disable=broad-exception-caught
      raise ProviderFailed(
        f'{name!r} raised instead of returning weights: {exc!r}') from exc
  return guarded


def strategies() -> dict[str, WeightProvider]:
  '''Return every provider in :mod:`stock_rl.baselines`, by name.

  The table is discovered from the module's own ``__all__`` rather than
  written out here, so adding a baseline cannot leave it missing from
  the control arm. The whole point of the endpoint is that the control
  arm is complete.

  Returns:
    Mapping of provider name to the provider callable.
  '''
  found: dict[str, WeightProvider] = {}
  for name in baselines.__all__:
    found[name] = getattr(baselines, name)
  return found


def run_baseline(
  service: ApiService,
  request: BacktestRequest,
) -> dict[str, Any]:
  '''Run the requested baseline over the requested panels.

  The default runner. It reuses
  :func:`stock_rl.portfolio.run_portfolio`, so the API, the RL
  environment and the published baselines are measured by one
  instrument with one cost model -- three different Sharpes from three
  code paths would be unfalsifiable.

  Args:
    service: Service supplying the panels.
    request: Validated trigger.

  Returns:
    JSON-safe mapping with the equity curve, returns, final weights and
    headline metrics.

  Raises:
    Refused: If the strategy is unknown, no panels were selected, or the
      backtester refuses the parameter combination.
    ProviderFailed: If the provider itself raises. Distinct from
      :class:`Refused` on purpose: a 422 says "change your request", and
      the request is not what is wrong here.
  '''
  table = strategies()
  if request.strategy not in table:
    raise Refused(
      f'unknown strategy {request.strategy!r}; known: {sorted(table)}')
  panels = service.panels_for(request.symbols)
  if not panels:
    raise Refused('no price panels matched the requested symbols')
  try:
    result = run_portfolio(
      panels,
      # Wrapped so a provider's own ValueError is not mistaken for the
      # backtester refusing the parameter combination below.
      _guarded(request.strategy, table[request.strategy]),
      capital=request.capital,
      rebalance_days=request.rebalance_days,
      max_weight=request.max_weight,
      history=request.history,
    )
  except ValueError as exc:
    # The backtester refusing a parameter combination the caller chose
    # is a 422, not a crash. Letting it out would put a traceback in an
    # HTTP response for an input the caller could have been told about.
    raise Refused(str(exc)) from exc
  except Exception as exc:  # pylint: disable=broad-exception-caught
    raise ProviderFailed(
      f'{request.strategy!r} raised instead of returning weights: '
      f'{exc!r}') from exc
  return {
    'strategy': request.strategy,
    'symbols': sorted(panels),
    'capital': request.capital,
    'equity': list(result.equity),
    'returns': list(result.returns),
    'weights': dict(result.weights[-1]) if result.weights else {},
    'sharpe': sharpe_ratio(result.returns),
    'max_drawdown': max_drawdown(result.equity).depth,
    'turnover': result.turnover,
    'total_cost': result.total_cost,
    'total_return_multiple': result.total_return_multiple,
    'rebalances': result.rebalances,
    'points': len(result.equity),
    'not_advice': not_advice,
  }


def momentum_signals(
  panels: Mapping[str, Sequence[Bar]],
  held: Mapping[str, float] | None = None,
  top: int = signal_top,
  max_weight: float = signal_max_weight,
  lookback: int = signal_lookback,
  skip: int = signal_skip,
) -> list[Signal]:
  '''Return one signal per symbol from the 12-1 cross-sectional ranking.

  Both the rank and the target weight come from the same score, so they
  cannot disagree. The score is
  :func:`stock_rl.indicators.momentum` over ``lookback`` bars with the
  most recent ``skip`` bars excluded.

  ``confidence`` is the midrank percentile of that score. It is an
  ordering, not a probability: nothing here has been calibrated against
  outcome frequency, so a high value means "near the top of this panel"
  and nothing more. When every score is identical there is no
  dispersion to rank, so every symbol is ``HOLD`` at confidence 0.0 --
  a flat ranking is the honest answer and manufacturing a ladder out of
  ties would invent conviction.

  Args:
    panels: Price panels keyed by symbol.
    held: Currently held weight per symbol.
    top: How many leaders to hold.
    max_weight: Cap on any single weight.
    lookback: Bars in the momentum window.
    skip: Most recent bars excluded from it.

  Returns:
    One :class:`Signal` per symbol, ordered by rank then symbol.
  '''
  held_weights = dict(held or {})
  scores = {symbol: _score(panel, lookback, skip)
            for symbol, panel in panels.items()}
  scored = {symbol: value for symbol, value in scores.items()
            if value is not None}
  ordered = sorted(scored, key=lambda name: (-scored[name], name))
  dispersed = len(set(scored.values())) > 1
  targets = _targets(ordered, panels, top, max_weight)
  rows = []
  for symbol in sorted(panels):
    rows.append(_signal_for(
      symbol, scores[symbol], ordered, scored, targets,
      held_weights.get(symbol, 0.0), dispersed, lookback, skip))
  return rows


def _last_bar(panels: Mapping[str, Sequence[Bar]]) -> str | None:
  '''Return the timestamp of the most recent bar across every panel.

  The bar a signal is measured on, which is a property of the data rather
  than of the clock. Panels are aligned by
  :func:`stock_rl.portfolio.run_portfolio`, so in practice they share a
  final bar and the maximum is that one; taking the maximum rather than
  the first panel's last bar keeps the answer truthful if they ever do
  not.

  Args:
    panels: Loaded price panels.

  Returns:
    ISO 8601 timestamp of the latest bar, or None with no panels.
  '''
  stamps = [panel[-1].timestamp for panel in panels.values() if panel]
  return _stamp(max(stamps)) if stamps else None


def _score(panel: Sequence[Bar], lookback: int, skip: int) -> float | None:
  '''Return the 12-1 trailing return of one panel.

  Args:
    panel: Bars in ascending time order.
    lookback: Bars in the momentum window.
    skip: Most recent bars excluded from it.

  Returns:
    Fractional return, or None when the panel is too short to score.
  '''
  closes = [bar.close for bar in panel]
  window = closes[:len(closes) - skip] if skip else closes
  return momentum(window, lookback)


def _targets(
  ordered: list[str],
  panels: Mapping[str, Sequence[Bar]],
  top: int,
  max_weight: float,
) -> dict[str, float]:
  '''Return target weights: the leaders share the cap, the rest to cash.

  Args:
    ordered: Scored symbols, best momentum first.
    panels: Loaded panels, which name the symbols that cannot be scored.
    top: How many leaders to hold.
    max_weight: Cap on any single weight.

  Returns:
    Weight per symbol across every loaded symbol, including unscorable
    ones at zero.
  '''
  chosen = ordered[:top]
  if not chosen:
    return {symbol: 0.0 for symbol in panels}
  weight = min(1.0 / len(chosen), max_weight)
  return {symbol: weight if symbol in chosen else 0.0
          for symbol in panels}


def _window_clause(lookback: int, skip: int) -> str:
  '''Return the momentum window as one clause, whatever the arguments.

  ``skip`` is stated even at zero. "Skipping the most recent 0 bars" reads
  oddly, but it is the number the caller passed and the number the score
  used, and a reader comparing two payloads needs to see that one excluded
  nothing rather than infer it from a sentence that changed shape.

  Args:
    lookback: Bars in the momentum window.
    skip: Most recent bars excluded from it.

  Returns:
    A clause naming the window, e.g. ``'252 bars, skipping the most
    recent 21'``.
  '''
  return f'{lookback} bars, skipping the most recent {skip}'


def _signal_for(
  symbol: str,
  score: float | None,
  ordered: list[str],
  scored: Mapping[str, float],
  targets: Mapping[str, float],
  held: float,
  dispersed: bool,
  lookback: int,
  skip: int,
) -> Signal:
  '''Build one signal, choosing the action and stating the evidence.

  The window named in each reason is the ``lookback`` and ``skip`` the
  score was actually built from, never the module defaults. A reason is
  the evidence for the action beside it, and one that cites a 252-bar
  window for a score computed over five bars describes a computation
  that did not happen -- which is precisely the sort of drift this
  project's audit research is against.

  Args:
    symbol: Symbol the signal is about.
    score: Its 12-1 momentum, or None when unscorable.
    ordered: Scored symbols, best momentum first.
    scored: Momentum per scored symbol.
    targets: Target weight per symbol.
    held: Currently held weight.
    dispersed: Whether the panel had any dispersion to rank.
    lookback: Bars in the momentum window actually used.
    skip: Most recent bars excluded from it.

  Returns:
    The signal.
  '''
  target = float(targets.get(symbol, 0.0))
  window = _window_clause(lookback, skip)
  if score is None:
    return Signal(
      symbol=symbol,
      action='HOLD',
      confidence=0.0,
      reasons=(f'fewer than {lookback + skip + 1} bars, so the {window} '
               'momentum window is not yet available',),
      target_weight=0.0,
      held_weight=held,
    )
  rank = ordered.index(symbol) + 1
  total = len(scored)
  confidence = _midrank(symbol, scored) if dispersed else 0.0
  reasons = [
    f'momentum {score:+.2%} over {window}',
    f'rank {rank} of {total} by momentum',
  ]
  if not dispersed:
    reasons.append('every symbol scored identically, so the ranking '
                   'carries no information')
  if target - held > weight_epsilon:
    action = 'BUY'
    reasons.append(f'target {target:.2%} against {held:.2%} held, so the '
                   'signal asks to add')
  elif held - target > weight_epsilon:
    action = 'SELL'
    reasons.append(f'target {target:.2%} against {held:.2%} held, so the '
                   'signal asks to reduce')
  else:
    action = 'HOLD'
    reasons.append(f'target {target:.2%} matches the {held:.2%} held, so '
                   'no trade is implied')
  return Signal(
    symbol=symbol,
    action=action,
    confidence=confidence,
    reasons=tuple(reasons),
    target_weight=target,
    held_weight=held,
    score=score,
    rank=rank,
  )


def _midrank(symbol: str, scored: Mapping[str, float]) -> float:
  '''Return the midrank percentile of one score within the panel.

  Ties share the midpoint of the ranks they span, which is why an
  identical panel produces one shared value instead of an arbitrary
  ladder.

  Args:
    symbol: Symbol being ranked.
    scored: Momentum per scored symbol.

  Returns:
    Percentile in ``[0, 1]``, or 0.0 for a panel of one symbol, where
    no comparison exists.
  '''
  total = len(scored)
  score = scored[symbol]
  lower = sum(1 for other in scored.values() if other < score)
  tied = sum(1 for other in scored.values() if other == score)
  # The denominator is floored at one so a single-symbol panel scores
  # 0.0 rather than dividing by zero, which keeps this callable from
  # anywhere without a guard the caller would have to remember.
  return (lower + 0.5 * (tied - 1)) / max(1, total - 1)


def historical_var(
  returns: Sequence[float],
  confidence: float = var_confidence,
) -> tuple[float | None, float]:
  '''Return the historical one-period VaR and the sample size behind it.

  The empirical quantile of realised returns, which is the only VaR the
  standard library can compute without assuming a distribution. No
  normal assumption is made, because assuming one for a book of Indian
  mid-caps understates the tail those books actually produce.

  Args:
    returns: Per-period returns as fractions.
    confidence: Confidence level, e.g. 0.95.

  Returns:
    Tuple of (loss as a positive fraction or None, observation count).
    With no usable return the fraction is None rather than 0.0: a VaR of
    0.0 means "the worst of the returns we have was a gain", which is a
    measurement, whereas a count of 0 means there is no history at all,
    and that is the absence of a measurement. A caller that cannot tell
    those apart will print "VaR Rs 0" beside a book that has never run.

  Raises:
    ValueError: If ``confidence`` is outside ``(0, 1)``.
  '''
  if not 0.0 < confidence < 1.0:
    raise ValueError(f'confidence must be in (0, 1), got {confidence}')
  usable = sorted(value for value in returns if math.isfinite(value))
  if not usable:
    return None, 0.0
  index = int(math.floor((1.0 - confidence) * len(usable)))
  index = max(0, min(index, len(usable) - 1))
  return max(0.0, -usable[index]), float(len(usable))


def _switch_view(
  switch: KillSwitchLike | None,
  drawdown: float,
) -> dict[str, Any]:
  '''Describe the kill switch, including a refusal to read it.

  A switch whose state cannot be read -- typically a corrupt state file,
  which :class:`stock_rl.risk.killswitch.KillSwitch` raises on purpose --
  is reported as an error with ``tripped: null``. Treating that as
  untripped would be the dangerous reading: the class exists precisely to
  refuse rather than re-arm, and a dashboard that smoothed the refusal
  away would restore the defect at the last layer.

  The catch is broad on purpose and it is the whole point of this
  function: the switch reads a file, a socket or a lock on every call,
  and ``OSError``, ``KeyError`` and ``TypeError`` are all things a real
  deployment raises there. Naming a subset would mean the one that fires
  in production is the one this endpoint does not catch, and a safety
  surface that raises is a safety surface that is simply absent.

  Args:
    switch: The switch, or None when none is wired in.
    drawdown: Current drawdown, fed to the threshold comparison.

  Returns:
    Mapping describing the switch. ``status`` is ``'unreadable'`` when any
    read raised, which is not the same answer as ``'armed'``.
  '''
  if switch is None:
    return {
      'wired': False,
      'tripped': None,
      'trading_enabled': None,
      'algo_id': None,
      'status': 'not_wired',
      'trips': [],
    }
  try:
    tripped = bool(switch.tripped)
    return {
      'wired': True,
      'tripped': tripped,
      'trading_enabled': bool(switch.trading_enabled),
      'algo_id': switch.algo_id,
      'status': 'TRIPPED' if tripped else 'armed',
      'trips': [_trip_view(trip) for trip in switch.history]
      if hasattr(switch, 'history') else [],
    }
  except Exception as exc:  # pylint: disable=broad-exception-caught
    # KillSwitchError derives from RuntimeError, and a refusal here is
    # the state worth surfacing rather than swallowing.
    return {
      'wired': True,
      'tripped': None,
      'trading_enabled': None,
      'algo_id': getattr(switch, 'algo_id', None),
      'status': 'unreadable',
      'error': str(exc),
      'drawdown': drawdown,
      'trips': [],
    }


def _trip_view(trip: TripLike) -> dict[str, Any]:
  '''Return one recorded trip as JSON-safe primitives.

  Args:
    trip: Recorded trip.

  Returns:
    Mapping with the code, the detail, the timestamp and the observation.
  '''
  return {
    'code': getattr(trip, 'code', ''),
    'detail': getattr(trip, 'detail', ''),
    'at': getattr(trip, 'at', ''),
    'observed': getattr(trip, 'observed', 0.0),
  }


def _switch_status(switch: KillSwitchLike | None) -> str:
  '''Return a one-word kill-switch state for the health payload.

  Args:
    switch: The switch, or None.

  Returns:
    ``'armed'``, ``'TRIPPED'``, ``'unreadable'`` or ``'not_wired'``.
  '''
  return str(_switch_view(switch, 0.0)['status'])


def _thresholds(switch: KillSwitchLike | None) -> dict[str, Any]:
  '''Return the pre-defined trip limits, or an empty mapping.

  The limits are supplementary detail: ``breached`` is the answer that
  decides whether trading halts, so a limits read that raises costs the
  caller four numbers and must not cost them the whole payload. The catch
  is broad for the same reason :func:`_switch_view`'s is, and the same
  cost applies in reverse: the switch that cannot be read is reported as
  unreadable by its own view, so an empty mapping here never becomes a
  claim that there are no limits.

  Args:
    switch: The switch, or None.

  Returns:
    Mapping of limit name to value, empty when no switch is wired in or
    the limits could not be read.
  '''
  if switch is None:
    return {}
  try:
    limits = switch.thresholds
    return {
      'index_fall': limits.index_fall,
      'max_drawdown': limits.max_drawdown,
      'ack_latency': limits.ack_latency,
      'fno_ban': limits.fno_ban,
    }
  except Exception:  # pylint: disable=broad-exception-caught
    return {}


def _breached(switch: KillSwitchLike | None,
              drawdown: float) -> list[str]:
  '''Return the conditions the switch considers breached.

  The switch is asked rather than re-derived, because a second copy of
  the threshold comparison is a second definition of a safety condition
  and the two will drift. The live metrics it needs are the ones this
  service holds; the session index change and the acknowledgement
  latency belong to a live feed and are not held here, so they are not
  invented.

  A read that raises yields an empty list, and that is the one place this
  module accepts a silent answer. It does so because the alternative is
  worse than silence: ``breached`` is supplementary detail on a payload
  whose ``kill_switch`` block already refuses loudly, and a partially
  readable switch must still be reportable. The empty list is a list of
  conditions this process was able to read, next to a status that says
  the state could not be.

  Args:
    switch: The switch, or None.
    drawdown: Current drawdown as a positive fraction.

  Returns:
    Detail text per breached condition, empty when clear, unwired or
    unreadable.
  '''
  if switch is None:
    return []
  try:
    return list(switch.breached(0.0, drawdown, False, None))
  except Exception:  # pylint: disable=broad-exception-caught
    return []


def _stamp(moment: datetime) -> str:
  '''Return an ISO 8601 UTC timestamp.

  A :class:`datetime.date` is accepted because :class:`stock_rl.bars.Bar`
  carries whatever the vendor CSV gave it, and daily exports routinely
  hold a bare date. Such a bar has no time of day to report and is
  stamped at midnight UTC rather than refused, so a daily panel's last bar
  can be named in the payload.

  Args:
    moment: Timezone-aware or naive datetime, or a date.

  Returns:
    ISO 8601 string, assumed UTC when the value is naive.
  '''
  if not isinstance(moment, datetime):
    moment = datetime(moment.year, moment.month, moment.day)
  if moment.tzinfo is None:
    moment = moment.replace(tzinfo=timezone.utc)
  return moment.isoformat()


def _jsonable(value: Any) -> Any:
  '''Return ``value`` with every non-finite float replaced by ``None``.

  ``json.dumps`` emits bare ``NaN`` and ``Infinity``, which are not
  JSON and which ``JSON.parse`` rejects outright. One undefined metric
  would then blank the whole dashboard instead of showing one empty
  cell, so the substitution happens before serialisation.

  Args:
    value: Any structure.

  Returns:
    An equivalent structure containing only JSON-safe values.
  '''
  if isinstance(value, float):
    return value if math.isfinite(value) else None
  if isinstance(value, dict):
    return {str(key): _jsonable(item) for key, item in value.items()}
  if isinstance(value, (list, tuple)):
    return [_jsonable(item) for item in value]
  return value


def _floats(value: Any) -> list[float]:
  '''Return ``value`` as a list of finite floats.

  Args:
    value: Candidate sequence.

  Returns:
    The finite floats in it, or an empty list when it is not a sequence
    of numbers. Anything unusable becomes no state rather than a
    partially-populated series.
  '''
  if not isinstance(value, (list, tuple)):
    return []
  found = [float(item) for item in value
           if isinstance(item, (int, float))
           and not isinstance(item, bool)]
  return [item for item in found if math.isfinite(item)]


def _weights(value: Any) -> dict[str, float]:
  '''Return ``value`` as a symbol-to-weight mapping.

  Args:
    value: Candidate mapping.

  Returns:
    Finite weights keyed by symbol, empty when ``value`` is unusable.
  '''
  if not isinstance(value, dict):
    return {}
  found: dict[str, float] = {}
  for symbol, weight in value.items():
    if not isinstance(weight, (int, float)) or isinstance(weight, bool):
      continue
    number = float(weight)
    if math.isfinite(number):
      found[str(symbol)] = number
  return found


def _number(value: Any, fallback: float) -> float:
  '''Return ``value`` as a finite float.

  Args:
    value: Candidate number.
    fallback: Returned when ``value`` is not a finite number.

  Returns:
    The float, or ``fallback``.
  '''
  if isinstance(value, bool) or not isinstance(value, (int, float)):
    return fallback
  number = float(value)
  return number if math.isfinite(number) else fallback


def _parse_request(body: bytes, known: Iterable[str]) -> BacktestRequest:
  '''Validate a JSON body into a :class:`BacktestRequest`.

  Rejects rather than coerces, and rejects unknown keys rather than
  ignoring them. A silently dropped ``max_weight`` is the specific
  hazard: the run would succeed, report a Sharpe, and describe a
  portfolio nobody asked for.

  Args:
    body: Raw request body.
    known: Symbols the service has loaded.

  Returns:
    The validated request.

  Raises:
    BadRequest: If the body is not a JSON object, carries unknown keys,
      or carries a value outside its permitted range.
  '''
  text = body.decode('utf-8', errors='replace').strip()
  if not text:
    raise BadRequest('request body is empty; POST a JSON object')
  try:
    payload = json.loads(text)
  except json.JSONDecodeError as exc:
    raise BadRequest(f'body is not valid JSON: {exc}') from exc
  if not isinstance(payload, dict):
    raise BadRequest(
      f'body must be a JSON object, got {type(payload).__name__}')
  allowed = {
    'symbols', 'capital', 'strategy', 'rebalance_days', 'max_weight',
    'history',
  }
  unknown = sorted(set(payload) - allowed)
  if unknown:
    raise BadRequest(
      f'unrecognised field(s) {unknown}; accepted: {sorted(allowed)}')
  defaults = BacktestRequest()
  table = strategies()
  strategy = _text(payload.get('strategy', defaults.strategy),
                   'strategy')
  if strategy not in table:
    raise BadRequest(
      f'unknown strategy {strategy!r}; known: {sorted(table)}')
  symbols = _symbol_list(payload.get('symbols'), known)
  return BacktestRequest(
    symbols=symbols,
    capital=_positive_number(payload.get('capital', defaults.capital),
                            'capital'),
    strategy=strategy,
    rebalance_days=_count(payload.get('rebalance_days',
                                      defaults.rebalance_days),
                          'rebalance_days'),
    max_weight=_capped_number(payload.get('max_weight',
                                          defaults.max_weight),
                              'max_weight'),
    history=_count(payload.get('history', defaults.history), 'history'),
  )


def _symbol_list(value: Any, known: Iterable[str]) -> tuple[str, ...]:
  '''Validate a ``symbols`` field.

  Args:
    value: Candidate list of symbols.
    known: Symbols the service has loaded.

  Returns:
    Symbols in ascending order, deduplicated.

  Raises:
    BadRequest: If the field is not a list of non-empty strings, or
      names a symbol that is not loaded. An unknown symbol is a 400 and
      not a silent omission, because a backtest over fewer names than
      were requested is not the backtest that was asked for.
  '''
  if value is None:
    return ()
  if not isinstance(value, list):
    raise BadRequest(f'symbols must be a list, got {type(value).__name__}')
  loaded = set(known)
  chosen: set[str] = set()
  for item in value:
    if not isinstance(item, str) or not item.strip():
      raise BadRequest(f'symbols entries must be non-empty strings, '
                       f'got {item!r}')
    symbol = item.strip()
    if symbol not in loaded:
      raise BadRequest(
        f'symbol {symbol!r} is not loaded; loaded: {sorted(loaded)}')
    chosen.add(symbol)
  return tuple(sorted(chosen))


def _text(value: Any, name: str) -> str:
  '''Return ``value`` as a non-empty string.

  Args:
    value: Candidate string.
    name: Field name, for the error message.

  Returns:
    The stripped string.

  Raises:
    BadRequest: If the value is not a non-empty string.
  '''
  if not isinstance(value, str) or not value.strip():
    raise BadRequest(f'{name} must be a non-empty string, got {value!r}')
  return value.strip()


def _positive_number(value: Any, name: str) -> float:
  '''Return ``value`` as a finite positive float.

  Args:
    value: Candidate number.
    name: Field name, for the error message.

  Returns:
    The float.

  Raises:
    BadRequest: If the value is not a finite number greater than zero.
      Booleans are rejected explicitly: ``True`` is an int in Python and
      would otherwise be accepted as capital of 1 rupee.
  '''
  number = _number(value, float('nan'))
  if isinstance(value, bool) or not math.isfinite(number) or number <= 0.0:
    raise BadRequest(f'{name} must be a positive finite number, '
                     f'got {value!r}')
  return number


def _capped_number(value: Any, name: str) -> float:
  '''Return ``value`` as a finite float inside ``(0, 1]``.

  Args:
    value: Candidate number.
    name: Field name, for the error message.

  Returns:
    The float.

  Raises:
    BadRequest: If the value is outside ``(0, 1]``.
  '''
  number = _number(value, float('nan'))
  if isinstance(value, bool) or not math.isfinite(number) \
      or not 0.0 < number <= 1.0:
    raise BadRequest(f'{name} must be in (0, 1], got {value!r}')
  return number


def _count(value: Any, name: str) -> int:
  '''Return ``value`` as an integer of at least one.

  Args:
    value: Candidate integer.
    name: Field name, for the error message.

  Returns:
    The integer.

  Raises:
    BadRequest: If the value is not a whole number of at least one.
  '''
  if isinstance(value, bool) or not isinstance(value, int):
    raise BadRequest(f'{name} must be a whole number, got {value!r}')
  if value < 1:
    raise BadRequest(f'{name} must be >= 1, got {value}')
  return value


def _content_length(declared: Sequence[str] | None) -> int | None:
  '''Return the one body length a request declares, or None.

  ``int()`` is deliberately not used. It accepts ``'1_0'`` as ten and
  ``'+7'`` as seven, so a length this process and an intermediary can
  read as different numbers is a request-smuggling primitive, and a
  length that parses is not by itself evidence that a body of that size
  exists.

  Args:
    declared: Every ``Content-Length`` header value on the request, or
      None when the header is absent.

  Returns:
    The declared length in bytes, 0 when no header is present, or None
    when the request declares one this server will not guess at.
  '''
  if not declared:
    return 0
  if len(declared) > 1:
    # Two lengths are either a disagreement or a smuggling attempt, and
    # RFC 9110 section 8.6 says to reject rather than pick one.
    return None
  text = declared[0].strip()
  if not text:
    return 0
  if not text.isascii() or not text.isdigit():
    return None
  return int(text)


def _control_of(path: str) -> BacktestRequest | None:
  '''Return the control-arm configuration a query string asks for.

  The same validators as :func:`_parse_request` are used, so a query
  cannot smuggle in a value the POST body would have refused, and the two
  endpoints cannot disagree about what a legal configuration is.

  Args:
    path: Raw request path, possibly carrying a query string.

  Returns:
    The requested configuration, or None when the query carries nothing
    and the declared :data:`control_arm` applies.

  Raises:
    BadRequest: If a parameter is unknown or outside its permitted range.
  '''
  query = path.split('?', 1)[1].split('#', 1)[0] if '?' in path else ''
  fields = parse_qs(query, keep_blank_values=True)
  unknown = sorted(set(fields) - _control_fields)
  if unknown:
    raise BadRequest(
      f'unrecognised query parameter(s) {unknown}; accepted: '
      f'{sorted(_control_fields)}')
  if not fields:
    return None
  defaults = BacktestRequest()
  return BacktestRequest(
    symbols=defaults.symbols,
    capital=_positive_number(
      _decimal(_one(fields, 'capital')) or defaults.capital, 'capital'),
    strategy=defaults.strategy,
    rebalance_days=_count(
      _whole(_one(fields, 'rebalance_days'), 'rebalance_days')
      or defaults.rebalance_days,
      'rebalance_days'),
    max_weight=_capped_number(
      _decimal(_one(fields, 'max_weight')) or defaults.max_weight,
      'max_weight'),
    history=_count(_whole(_one(fields, 'history'), 'history')
                   or defaults.history, 'history'),
  )


def _whole(value: str | None, name: str) -> int | None:
  '''Return a query parameter as an int, or None when absent.

  A query string carries text, so a count arrives as ``'5'``. It is
  converted here rather than by :func:`_count`, which is written for the
  JSON body where a whole number must already *be* a number: a body that
  says ``"5"`` is a type error, while a query that says ``history=5`` is
  the ordinary spelling.

  Args:
    value: The raw parameter value.
    name: Parameter name, for the error message.

  Returns:
    The integer, or None when the parameter was absent.

  Raises:
    BadRequest: If the value is not a plain run of ASCII digits.
  '''
  if value is None:
    return None
  text = value.strip()
  if not text.isascii() or not text.isdigit():
    raise BadRequest(
      f'{name} must be a whole number in the query, got {value!r}')
  return int(text)


def _decimal(value: str | None) -> float | None:
  '''Return a query parameter as a float, or None when absent.

  Args:
    value: The raw parameter value.

  Returns:
    The float, or None when the parameter was absent.

  Raises:
    BadRequest: If the value will not parse as a number at all. The
      range is not checked here; the validator that receives the value
      owns that, so there is one range rule rather than two.
  '''
  if value is None:
    return None
  try:
    return float(value.strip())
  except ValueError as exc:
    raise BadRequest(f'not a number: {value!r}') from exc


def _one(
  fields: Mapping[str, list[str]],
  name: str,
) -> str | None:
  '''Return one query value, refusing a repeated one.

  Args:
    fields: Parsed query fields.
    name: Parameter to read.

  Returns:
    The single value, or None when absent.

  Raises:
    BadRequest: If the parameter was given more than once. Silently
      picking one of two is how a proxy and a server end up disagreeing
      about what was asked for.
  '''
  values = fields.get(name) or []
  if len(values) > 1:
    raise BadRequest(f'{name} was given {len(values)} times; give it once')
  return values[0] if values else None


def _hostname(authority: str) -> str:
  '''Return the host from a ``Host`` header or an origin URL.

  Args:
    authority: An authority string, e.g. ``127.0.0.1:8765``.

  Returns:
    The host without its port. IPv6 authorities are bracketed in a Host
    header and left unbracketed in an origin URL, so both shapes are
    handled.
  '''
  host = authority.strip()
  if host.startswith('['):
    end = host.find(']')
    if end != -1:
      return host[1:end]
  if ':' in host:
    return host.rsplit(':', 1)[0]
  return host


def admitted_origin(origin: str | None) -> str | None:
  '''Return the origin to answer a browser request from, or None.

  Loopback origins and the null origin are admitted; everything else is
  refused. This API binds loopback, has no authentication, and publishes
  positions and risk state, so a wildcard would hand all of it to any page
  on the internet. The null origin is admitted because the dashboard
  documents that it runs from ``file://``, and that is the only origin a
  ``file://`` document has.

  Args:
    origin: Value of the request's ``Origin`` header, or None.

  Returns:
    The origin to echo in ``Access-Control-Allow-Origin``, or None when
    the request carries no admitted Origin.
  '''
  if not origin:
    return None
  candidate = origin.strip()
  if candidate == 'null':
    return candidate
  scheme, _, rest = candidate.partition('://')
  if scheme != 'http' or not rest:
    return None
  return candidate if is_loopback(_hostname(rest)) else None


def _error(status: int, message: str) -> Response:
  '''Return a JSON error response.

  Args:
    status: HTTP status code.
    message: What went wrong, in the caller's terms.

  Returns:
    The response.
  '''
  return Response.of_json(status, {
    'error': message,
    'status': int(status),
    'not_advice': not_advice,
  })


#: Routes served, mapped to the HTTP methods allowed on them.
routes: dict[str, frozenset[str]] = {
  '/': frozenset({'GET', 'HEAD'}),
  '/index.html': frozenset({'GET', 'HEAD'}),
  # The two sibling files index.html pulls in. These were missing, which
  # made the dashboard render as unstyled text with no script at all: the
  # document was served, the stylesheet and the script were not, and
  # nothing in the suite noticed because it checked that the document was
  # served and that the assets were readable as text, but never that the
  # document's own references resolved. Every subresource named by the
  # served HTML must have a route here, or the page is a corpse.
  '/style.css': frozenset({'GET', 'HEAD'}),
  '/app.js': frozenset({'GET', 'HEAD'}),
  '/api/health': frozenset({'GET', 'HEAD'}),
  '/api/signals': frozenset({'GET', 'HEAD'}),
  '/api/equity': frozenset({'GET', 'HEAD'}),
  '/api/positions': frozenset({'GET', 'HEAD'}),
  '/api/baselines': frozenset({'GET', 'HEAD'}),
  '/api/risk': frozenset({'GET', 'HEAD'}),
  '/api/backtest': frozenset({'POST'}),
}


def dispatch(
  service: ApiService,
  method: str,
  path: str,
  body: bytes = b'',
) -> Response:
  '''Route one request and return its response, without a socket.

  Routing is a pure function of the service and the request so that the
  whole API is testable without binding a port, spawning a thread or
  waiting on I/O. :class:`ApiHandler` adds nothing but the HTTP framing.

  Args:
    service: Service holding the state and the seams.
    method: HTTP method, upper-cased by the caller.
    path: Request path, possibly with a query string.
    body: Raw request body for POST.

  Returns:
    The response. Unknown routes are 404, wrong methods on a known route
    are 405, malformed input is 400, a valid request that cannot run is
    422, and a provider or asset failure is 500. Nothing reaches the
    client as a traceback.
  '''
  route = _route_of(path)
  if route not in routes:
    return _error(HTTPStatus.NOT_FOUND, f'no such endpoint: {route}')
  lookup = 'GET' if method == 'HEAD' else method
  if lookup not in routes[route]:
    return _error(
      HTTPStatus.METHOD_NOT_ALLOWED,
      f'{method} is not allowed on {route}; allowed: '
      f'{sorted(routes[route])}')
  try:
    if route in ('/', '/index.html'):
      return Response.of_html(HTTPStatus.OK, index_html())
    if route == '/style.css':
      return Response(
        HTTPStatus.OK, stylesheet().encode('utf-8'),
        'text/css; charset=utf-8')
    if route == '/app.js':
      return Response(
        HTTPStatus.OK, script().encode('utf-8'),
        'text/javascript; charset=utf-8')
    if route == '/api/backtest':
      request = _parse_request(body, service.symbols)
      return Response.of_json(
        HTTPStatus.OK, service.backtest(request))
    builder = _readers[route]
    if route == '/api/baselines':
      # The control arm takes a configuration, because a Sharpe measured
      # at settings the caller will not trigger with answers no question.
      return Response.of_json(
        HTTPStatus.OK, builder(service, _control_of(path)))
    return Response.of_json(HTTPStatus.OK, builder(service))
  except BadRequest as exc:
    return _error(HTTPStatus.BAD_REQUEST, str(exc))
  except Refused as exc:
    return _error(HTTPStatus.UNPROCESSABLE_ENTITY, str(exc))
  except FileNotFoundError as exc:
    return _error(
      HTTPStatus.INTERNAL_SERVER_ERROR,
      f'the dashboard asset is missing: {exc}')
  except ProviderFailed as exc:
    return _error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))
  except Exception as exc:  # pylint: disable=broad-exception-caught
    # Terminal clause, and the reason the docstring above promises that
    # nothing reaches the client as a traceback. Without it a TypeError
    # from any of the branches above propagates into the handler, which
    # closes the socket with no response at all: the caller sees a
    # transport error instead of a diagnosis, and a socket error is not
    # something a caller can act on. The repr is enough to identify the
    # fault and the traceback stays in the server's log, where it
    # belongs; a stack trace in an HTTP body is a leak, not a diagnosis.
    _log.exception('unhandled error serving %s %s', method, route)
    return _error(
      HTTPStatus.INTERNAL_SERVER_ERROR,
      f'{type(exc).__name__}: {exc} (this is a bug; see the server log)')


#: Query parameters ``GET /api/baselines`` accepts. The same four the
#: trigger accepts, and validated by the same functions, so the two
#: endpoints cannot disagree about what a legal configuration is.
_control_fields = frozenset(
  {'capital', 'rebalance_days', 'max_weight', 'history'})

#: GET routes mapped to the service method that answers them.
_readers: dict[str, Callable[..., dict[str, Any]]] = {
  '/api/health': ApiService.health,
  '/api/signals': ApiService.signals,
  '/api/equity': ApiService.equity,
  '/api/positions': ApiService.positions,
  '/api/baselines': ApiService.baselines,
  '/api/risk': ApiService.risk,
}


def _route_of(path: str) -> str:
  '''Return the route a request path addresses.

  Query strings and fragments are dropped, and a single trailing slash
  is tolerated so ``/api/health/`` is not a 404 for a typo.

  Args:
    path: Raw request path.

  Returns:
    The normalised route.
  '''
  clean = path.split('?', 1)[0].split('#', 1)[0]
  if len(clean) > 1 and clean.endswith('/'):
    clean = clean[:-1]
  return clean or '/'


# ``do_GET``/``do_HEAD``/``do_POST`` are named by the standard library,
# which dispatches with ``getattr(self, 'do_' + method)``. Renaming them
# to satisfy the project-wide lowercase rule would produce a handler that
# silently answers 501 to every request, so the check is disabled for the
# three methods the stdlib contract forces and nothing else.
class ApiHandler(BaseHTTPRequestHandler):  # pylint: disable=invalid-name
  '''HTTP framing for one :class:`ApiService`.

  The handler holds no state of its own beyond the service it is bound
  to, so two handlers serving two services cannot contaminate each
  other, and the only thing worth testing is the framing -- which
  :func:`dispatch` already covers without a socket.
  '''

  server_version = f'stock_rl/{__version__}'
  sys_version = ''
  protocol_version = 'HTTP/1.1'

  def __init__(self, *args: Any, service: ApiService, **kwargs: Any) -> None:
    '''Bind a service to this handler.

    Args:
      *args: Positional arguments ``socketserver`` supplies.
      service: The service every request is routed to.
      **kwargs: Keyword arguments ``socketserver`` supplies.
    '''
    self.service = service
    super().__init__(*args, **kwargs)

  def do_GET(self) -> None:  # pylint: disable=invalid-name
    '''Serve a GET or HEAD request.

    Args:
      None.

    Returns:
      None. The response is written to the socket.
    '''
    if self._refuse_foreign_host():
      return
    self._respond(dispatch(self.service, 'GET', self.path))

  def do_HEAD(self) -> None:  # pylint: disable=invalid-name
    '''Serve a HEAD request.

    Routed as the GET it describes and then truncated, so a HEAD can
    never disagree with the body a GET would have returned.

    Args:
      None.

    Returns:
      None. The response headers are written to the socket.
    '''
    if self._refuse_foreign_host():
      return
    self._respond(dispatch(self.service, 'HEAD', self.path), body=False)

  def do_POST(self) -> None:  # pylint: disable=invalid-name
    '''Serve a POST request, draining the body whatever the route.

    The body is read before routing so that a 404 on a POST leaves the
    connection in a state the next request on it can be read from. Not
    draining is how a keep-alive connection desynchronises and the next
    request fails for a reason unrelated to itself -- which is why a
    framing refusal closes the connection explicitly and says so in a
    header rather than leaving the peer to reuse it.

    Args:
      None.

    Returns:
      None. The response is written to the socket.
    '''
    body = self._read_body()
    if isinstance(body, Response):
      self._respond(body)
      return
    self._respond(dispatch(self.service, 'POST', self.path, body))

  def do_OPTIONS(self) -> None:  # pylint: disable=invalid-name
    '''Answer a CORS preflight for the routes this server serves.

    A browser reading the API from a ``file://`` document, or from a
    different loopback port, sends ``OPTIONS`` before the real request.
    The framework answers 501 for any method it has no ``do_`` for, and
    501 is a dead end: no preflight, no cross-origin read, and a
    dashboard that only works when served from the same origin as the
    API.

    Args:
      None.

    Returns:
      None. The response is written to the socket.
    '''
    if self._refuse_foreign_host():
      return
    if self._cors_origin() is None:
      self._respond(_error(
        HTTPStatus.FORBIDDEN,
        'this API answers loopback and file:// origins only'))
      return
    self._respond(Response(HTTPStatus.NO_CONTENT, b''))

  def send_error(self, code: Any, message: Any = None,
                 explain: Any = None) -> None:
    '''Return framework-generated errors as JSON.

    :class:`BaseHTTPRequestHandler` answers an unsupported method or a
    malformed request line with an HTML body. A JSON API that answers
    ``DELETE /api/health`` with HTML leaves the caller parsing by
    trial.

    Args:
      code: HTTP status code.
      message: Short description, from the framework when not given.
      explain: Long description, unused.
    '''
    del explain
    status = int(code)
    short, _ = self.responses.get(status, ('error', ''))
    self._respond(_error(status, str(message or short)))

  def log_message(self, message: str, *args: Any) -> None:
    '''Route the framework's access log through :mod:`logging`.

    Args:
      message: Format string supplied by the framework.
      *args: Format arguments.
    '''
    _log.info('%s - %s', self.address_string(), message % args)

  def _read_body(self) -> bytes | Response:
    '''Return the request body, or the response that must replace it.

    Three faults, three answers, and none of them may borrow another's
    words.

    * ``Transfer-Encoding: chunked`` is refused by name and the
      connection is closed. RFC 9112 section 6.3 makes the chunked
      framing the message length when it is present, so reading
      ``Content-Length`` instead leaves the chunk preamble in the socket
      and the *next* request on the connection is parsed out of this
      one's bytes -- a 400 whose text blames a request that was fine.
      Closing is what makes the refusal safe: the stream is discarded
      whole rather than left half-read.
    * A ``Content-Length`` that is absent, repeated or not a plain run of
      ASCII digits is a 400 naming the header. It is not a 413: nothing
      here established that the body was too large, and a header that
      would not parse is a different fault with a different fix.
    * A body over :data:`max_body_bytes` is drained up to
      :data:`max_drain_bytes` and discarded, never buffered, so the peer
      finishes writing and reads the 413 rather than a reset.

    Returns:
      The raw bytes, or a :class:`Response` the caller must send in
      place of them.
    '''
    if self.headers.get('Transfer-Encoding'):
      self.close_connection = True
      return _error(
        HTTPStatus.BAD_REQUEST,
        'chunked Transfer-Encoding is not accepted; send Content-Length '
        'with the body length instead')
    length = _content_length(self.headers.get_all('Content-Length'))
    if length is None:
      self.close_connection = True
      return _error(
        HTTPStatus.BAD_REQUEST,
        'Content-Length must appear once and be a plain run of digits')
    if length > max_body_bytes:
      self._discard(min(length, max_drain_bytes))
      self.close_connection = True
      return _error(
        HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
        f'body exceeds {max_body_bytes} bytes')
    if length == 0:
      return b''
    return self.rfile.read(length)

  def _refuse_foreign_host(self) -> bool:
    '''Answer 421 when ``Host`` names an origin this server is not.

    This process binds loopback and publishes a trading book with no
    authentication. A browser will happily send a request for
    ``http://127.0.0.1:8765/`` to a page on another origin, and if this
    server answers such a request the reply is delivered as though it
    came from the attacker's origin -- which is exactly what makes a
    loopback service a DNS-rebinding target. RFC 9110 section 7.4
    reserves 421 for precisely this: the server is not able to produce a
    response for the requested authority.

    The connection is closed rather than drained, because nothing here
    has read a body that may be arriving, and half-read framing is worse
    than a closed socket.

    Returns:
      True when the request was refused and the answer is already sent.
    '''
    host = self.headers.get('Host')
    if host is None or is_loopback(_hostname(host)):
      return False
    self.close_connection = True
    self._respond(_error(
      HTTPStatus.MISDIRECTED_REQUEST,
      f'Host {host!r} is not a loopback address this server is bound to; '
      f'answering it would let another origin read this response'))
    return True

  def _cors_origin(self) -> str | None:
    '''Return the ``Access-Control-Allow-Origin`` value for this request.

    The dashboard documents that it can be opened from ``file://``, and a
    ``file://`` document sends ``Origin: null``, so the null origin is
    admitted along with loopback origins. Nothing else is: this API has no
    authentication, and a permissive ``*`` would publish the book to any
    page on the internet.

    Returns:
      The origin to echo, or None when the request carries no admitted
      Origin and no cross-origin header should be sent.
    '''
    return admitted_origin(self.headers.get('Origin'))

  def _discard(self, count: int) -> None:
    '''Read and throw away ``count`` bytes without buffering them.

    Args:
      count: Bytes to consume from the request stream.
    '''
    remaining = count
    while remaining > 0:
      chunk = self.rfile.read(min(remaining, 65536))
      if not chunk:
        break
      remaining -= len(chunk)

  def _respond(self, response: Response, body: bool = True) -> None:
    '''Write a response with an explicit length.

    Args:
      response: The response to write.
      body: False to declare the length without sending the bytes, which
        is what HEAD means and the only reason this is a parameter.
    '''
    self.send_response(response.status)
    self.send_header('Content-Type', response.content_type)
    self.send_header('Content-Length', str(len(response.body)))
    self.send_header('Cache-Control', 'no-store')
    # RFC 9112 section 9.6: `Connection: close` is the framing signal that
    # tells the peer this connection is finished. Setting
    # `close_connection` without emitting the header is the half of the
    # keep-alive protocol only this server knows about -- the client has
    # no way to learn the stream is over, so it reuses the socket and the
    # reuse resets.
    if self.close_connection:
      self.send_header('Connection', 'close')
    allowed = self._cors_origin()
    if allowed is not None:
      self.send_header('Access-Control-Allow-Origin', allowed)
      self.send_header('Vary', 'Origin')
    if self.command == 'OPTIONS':
      self.send_header('Access-Control-Allow-Methods', 'GET, HEAD, OPTIONS')
      self.send_header('Access-Control-Allow-Headers', 'Accept, Content-Type')
    self.end_headers()
    if body:
      self.wfile.write(response.body)


def is_loopback(host: str) -> bool:
  '''Return whether a bind address is a loopback address.

  Args:
    host: Candidate bind address.

  Returns:
    True for a loopback IP or for ``localhost``. Public because the
    question "may this process listen here" is worth asking without
    starting a server to find out.
  '''
  try:
    return ipaddress.ip_address(host).is_loopback
  except ValueError:
    return host.strip().lower() == 'localhost'


class ApiServer(ThreadingHTTPServer):
  '''A threading HTTP server with this module's socket policy.

  ``daemon_threads`` matters for a shutdown: a daemon thread holding a
  keep-alive connection cannot stop the process from exiting, and this
  server is expected to be stopped with Ctrl-C from a terminal.
  '''

  daemon_threads = True
  allow_reuse_address = True


class ApiServer6(ApiServer):
  '''The same server on an IPv6 socket.

  Separate rather than configured at runtime because
  ``address_family`` is read when the socket is created, and
  ``http.server`` hard-codes IPv4. Without this, ``::1`` is a loopback
  address the module would accept and then fail to bind, which is the
  worst of both answers.
  '''

  address_family = socket.AF_INET6


def bound_address(server: ThreadingHTTPServer) -> tuple[str, int]:
  '''Return the host and port a server actually bound.

  An IPv6 socket reports ``server_address`` as a nested four-tuple, so
  indexing it blindly yields a tuple where a host belongs. One place to
  unpack it correctly is worth more than one place to get it wrong.

  Args:
    server: A bound or boundable server.

  Returns:
    Tuple of (host string, port).
  '''
  address = server.server_address
  host = address[0] if isinstance(address[0], str) else address[0][0]
  return str(host), int(address[1])


def build_server(
  service: ApiService | None = None,
  host: str = default_host,
  port: int = default_port,
) -> ThreadingHTTPServer:
  '''Build a server bound to ``host``, refusing anything off loopback.

  The refusal is a hard error rather than a warning. This process
  serves positions, signals and audit state, and it has no
  authentication; bound to a routable interface it would publish a
  trading book to the network. Making that impossible to do by
  accident is the whole point, so there is no flag that overrides it --
  the supported way to expose the service is a TLS-terminating reverse
  proxy with a client certificate, which keeps the identity decision
  where it belongs.

  Args:
    service: Service to serve. A fresh empty one by default.
    host: Bind address. Must be loopback, IPv4 or IPv6.
    port: Bind port. 0 asks the kernel for a free port.

  Returns:
    A server not yet serving.

  Raises:
    ValueError: If the host is not a loopback address or the port is out
      of range.
  '''
  if not is_loopback(host):
    raise ValueError(
      f'refusing to bind {host!r}: this API has no authentication and '
      f'serves trading decisions and audit state, so it binds loopback '
      f'only ({default_host}). Put a TLS-terminating reverse proxy with a '
      f'client certificate in front of it instead of widening this.')
  if not 0 <= port <= 65535:
    raise ValueError(f'port must be in [0, 65535], got {port}')
  running = service if service is not None else ApiService()
  server_class = ApiServer6 if ':' in host else ApiServer
  return server_class(
    (host, port), partial(ApiHandler, service=running))


def serve(
  service: ApiService | None = None,
  host: str = default_host,
  port: int = default_port,
) -> None:
  '''Serve until interrupted.

  Args:
    service: Service to serve. A fresh empty one by default.
    host: Bind address. Must be loopback.
    port: Bind port.

  Raises:
    ValueError: If the host is not a loopback address.
  '''
  server = build_server(service, host, port)
  bound_host, bound_port = bound_address(server)
  _log.info(
    'serving %s on http://%s:%s/ (loopback only, no authentication, '
    'not financial advice)', __version__, bound_host, bound_port)
  try:
    server.serve_forever()
  except KeyboardInterrupt:
    _log.info('interrupted; closing')
  finally:
    server.server_close()


def load_panels(directory: str | Path) -> dict[str, list[Bar]]:
  '''Load one CSV per symbol from a directory.

  Deliberately the simplest thing that works: each ``*.csv`` file named
  after its symbol, parsed by :func:`stock_rl.bars.load_csv`, which
  already refuses a scrambled or unsorted series. Alignment across
  symbols is left to
  :func:`stock_rl.portfolio.run_portfolio`, which raises on misaligned
  panels rather than forward-filling bars that did not trade.

  Args:
    directory: Directory of vendor CSV exports.

  Returns:
    Mapping of symbol to bars, empty when the directory holds no CSV.

  Raises:
    ValueError: If the directory is not a directory.
  '''
  path = Path(directory)
  if not path.is_dir():
    raise ValueError(f'{path} is not a directory')
  panels = {}
  for candidate in sorted(path.glob('*.csv')):
    panels[candidate.stem] = load_csv(candidate)
  return panels


def main(argv: Sequence[str] | None = None) -> int:
  '''Parse arguments and serve.

  Args:
    argv: Argument list, defaulting to ``sys.argv[1:]``.

  Returns:
    Process exit status: 0 on a clean shutdown.

  Raises:
    ValueError: If the requested bind address is not loopback, or the
      data directory is unusable.
  '''
  parser = argparse.ArgumentParser(
    prog='stock_rl.api',
    description=(
      'Read-only dashboard API for the backtest core. Binds loopback '
      'only, has no authentication, and is not financial advice.'))
  parser.add_argument(
    '--host', default=default_host,
    help=f'bind address; loopback only (default: {default_host})')
  parser.add_argument(
    '--port', type=int, default=default_port,
    help=f'bind port (default: {default_port})')
  parser.add_argument(
    '--data-dir', default=None,
    help='directory of per-symbol CSV exports to serve')
  arguments = parser.parse_args(argv)
  logging.basicConfig(level=logging.INFO)
  panels = load_panels(arguments.data_dir) if arguments.data_dir else {}
  service = ApiService(panels=panels)
  serve(service, arguments.host, arguments.port)
  return 0


if __name__ == '__main__':
  raise SystemExit(main())

