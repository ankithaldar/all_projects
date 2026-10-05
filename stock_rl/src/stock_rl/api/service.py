#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''The service every read endpoint answers from.

:class:`ApiService` holds the book, the panels and the seams the derived
answers are read through -- ``clock`` for monotonic seconds, ``now`` for
wall-clock stamps, and ``backtest_runner`` and ``signal_source`` for the
two answers that are computed rather than stored. The seams exist so that
a test needs neither a socket, nor a sleep, nor five years of price
history.

Two decisions in here are about disclosure rather than arithmetic. A
metric with no series behind it is ``None`` and not its degenerate value:
:func:`stock_rl.metrics.sharpe_ratio` answers 0.0 for an empty series and
``total_return`` answers 1.0, which are byte for byte the numbers a real
backtest that returned nothing reports. And every rupee figure in
``positions`` shares one denominator, so a book that rose from ten
million to fifty-one million cannot print three numbers that disagree by
twenty-five million.

``POST /api/backtest`` is the only write here, and it writes observation
state rather than a position. A run over different symbols replaces the
reported weights instead of blending into them, so two backtests can
never be mistaken for one portfolio.

Split out of the former single-module ``stock_rl.api`` without change.
'''

from __future__ import annotations

import time
from collections.abc import Iterable, Mapping, Sequence
from functools import partial
from typing import Any

from stock_rl import __version__
from stock_rl.api.constants import (
  not_advice,
  signal_lookback,
  signal_skip,
  signal_top,
  var_confidence,
)
from stock_rl.api.models import (
  BacktestRequest,
  PortfolioState,
  _floats,
  _number,
  _stamp,
  _utcnow,
  _weights,
  control_arm,
)
from stock_rl.api.protocols import (
  BacktestRunner,
  Clock,
  KillSwitchLike,
  SignalSource,
  WallClock,
)
from stock_rl.api.risk import (
  _breached,
  _switch_status,
  _switch_view,
  _thresholds,
  historical_var,
)
from stock_rl.api.runner import (
  _accepts_cap,
  _applied_cap,
  _capped,
  _guarded,
  run_baseline,
  strategies,
)
from stock_rl.api.signals import _last_bar, momentum_signals
from stock_rl.bars import Bar
from stock_rl.metrics import max_drawdown, sharpe_ratio, total_return
from stock_rl.portfolio import WeightProvider, run_portfolio


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

    Echoing the requested configuration is only half of that, and the
    half that is easy. The requested ``max_weight`` is now bound into
    each provider as well as into the engine, and each row carries the
    cap its book was actually built at under ``applied_max_weight``.
    Declaring a configuration that the books were not built under is
    the defect this endpoint's disclosure exists to prevent: it used to
    answer ``?max_weight=0.3`` with a payload claiming 0.3 over a 0.10
    book, and three different caps returned one byte-identical Sharpe.

    Args:
      request: Configuration to measure at, or None for
        :data:`control_arm`.

    Returns:
      Mapping with one row per provider in :mod:`stock_rl.baselines`,
      each carrying Sharpe, drawdown, turnover and cost, or the reason it
      could not run, plus whether the cap reached the provider and the
      cap the book was built at. The settings the rows were measured
      under are disclosed at the top level as well as under ``request``.
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

    The requested ``max_weight`` is bound to the provider as well as to
    the engine, and the row reports both numbers: ``cap_bound`` says the
    provider was handed the cap, and ``applied_max_weight`` is read back
    off the book's own snapshots. Echoing the requested cap alone is
    what let a 0.30 request be answered with a 0.10 book.

    Args:
      name: Provider name as registered.
      provider: The weight provider to run.
      request: Configuration to measure at, or None for
        :data:`control_arm`.

    Returns:
      Mapping with the strategy name and either its metrics or an error.
    '''
    settings = request or control_arm
    row: dict[str, Any] = {
      'name': name,
      # Stamped before anything runs, so a row that never got as far as a
      # book still reports whether its provider could take a cap.
      'cap_bound': _accepts_cap(provider),
    }
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
        _guarded(name, _capped(provider, settings.max_weight)),
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
      # Measured, not echoed. Equal to settings.max_weight whenever the
      # requested cap is the binding constraint, and lower when it is
      # not, which is the whole defect this field exists to make visible.
      'applied_max_weight': _applied_cap(result),
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
