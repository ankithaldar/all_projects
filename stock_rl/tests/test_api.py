#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Tests for the read-only HTTP API and its dashboard.

Three properties are the point of this file, and each has a test that
would fail if it were removed.

**Nothing sleeps and nothing binds a socket for the endpoint tests.**
Every endpoint is exercised through :func:`stock_rl.api.dispatch`, which
is a pure function of the service and the request. Uptime comes from an
injected clock and backtests from an injected runner, so a test asserts
an exact number rather than a range, and the suite runs in milliseconds.
One test does open a real loopback socket, because "the server actually
serves bytes over HTTP" is a claim about the framing that a pure
function cannot make.

**A malformed request is a 400 with a sentence, never a traceback.** A
traceback in an HTTP response is a leak rather than a diagnosis, and the
only way to keep that guarantee is to test every rejection path.

**The equity endpoint reports what ``metrics.py`` reports.** The Sharpe
and the drawdown are asserted against the same functions the backtester
uses, and one test monkeypatches ``api.sharpe_ratio`` to prove the
endpoint calls through instead of keeping its own arithmetic.
'''

import json
import re
import socket
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from http.client import HTTPConnection
from random import Random

import pytest

from stock_rl import __version__, api
from stock_rl.bars import Bar
from stock_rl.metrics import max_drawdown, sharpe_ratio
from stock_rl.web import asset_text, index_html, script, stylesheet

#: A fixed instant, so recorded timestamps are comparable across runs.
FROZEN = datetime(2026, 3, 4, 9, 30, tzinfo=timezone.utc)

#: Bars needed before the 12-1 momentum window exists.
WINDOW = api.signal_lookback + api.signal_skip + 1


def panel(count, base, drift, seed):
  '''Return a deterministic price panel.

  Args:
    count: Number of bars.
    base: Starting price.
    drift: Mean per-bar return.
    seed: Seed for the private Random instance.

  Returns:
    Bars in ascending time order, with the open one bar behind the close
    so a backtest can actually fill.
  '''
  rng = Random(seed)
  bars = []
  price = base
  previous = base
  for index in range(count):
    price = max(1.0, price * (1.0 + drift + rng.gauss(0.0, 0.012)))
    bars.append(Bar(
      timestamp=datetime(2026, 1, 1) + timedelta(days=index),
      open=previous,
      high=max(previous, price),
      low=min(previous, price),
      close=price,
      volume=1e5,
    ))
    previous = price
  return bars


def three_panels():
  '''Return a small universe with distinct momentum.

  Args:
    None.

  Returns:
    Mapping of symbol to bars, long enough for the momentum window and
    ordered so that AAA leads and CCC lags.
  '''
  return {
    'AAA': panel(WINDOW + 40, 100.0, 0.002, 1),
    'BBB': panel(WINDOW + 40, 200.0, 0.001, 2),
    'CCC': panel(WINDOW + 40, 50.0, -0.001, 3),
  }


@dataclass(frozen=True, slots=True)
class FakeThresholds:
  '''The pre-defined trip conditions, as the API reads them.

  Attributes:
    index_fall: Session index fall limit.
    max_drawdown: Peak-to-trough drawdown limit.
    ack_latency: Acknowledgement latency limit, in seconds.
    fno_ban: Whether an F&O ban trips the switch.
  '''

  index_fall: float = 0.05
  max_drawdown: float = 0.15
  ack_latency: float = 2.0
  fno_ban: bool = True


@dataclass(slots=True)
class FakeSwitch:
  '''A kill switch stand-in with the read-only surface the API uses.

  Declared here rather than importing
  :class:`stock_rl.risk.killswitch.KillSwitch` so these tests keep
  passing whether or not that module is present, and so tripping a fake
  cannot write a state file under the user's home directory.

  Attributes:
    algo_id: Algo ID the switch belongs to.
    latched: Whether it reports itself tripped.
    drawdown: Drawdown fed to ``breached``.
    broken: Raise on every read, standing in for the corrupt-state-file
      refusal the real switch raises on purpose.
    refuse_breach: Raise from ``breached`` instead of answering.
    trips: Recorded trips.
    calls: How many times ``breached`` was asked.
    thresholds: The pre-defined conditions.
  '''

  algo_id: str = 'TESTALGO01'
  latched: bool = False
  drawdown: float = 0.0
  broken: bool = False
  refuse_breach: bool = False
  trips: tuple = ()
  calls: int = 0
  thresholds: FakeThresholds = field(default_factory=FakeThresholds)

  @property
  def tripped(self) -> bool:
    '''Return whether the switch is latched, or refuse to say.

    Returns:
      The latch state, or raise when ``broken`` is set.
    '''
    if self.broken:
      raise RuntimeError('state file is truncated; refusing to re-arm')
    return self.latched

  @property
  def trading_enabled(self) -> bool:
    '''Return whether an order may be released.'''
    return not self.tripped

  @property
  def history(self) -> tuple:
    '''Return the recorded trips.'''
    return self.trips

  def breached(self, index_change, drawdown, fno_banned, ack_latency):
    '''Record the call and report the drawdown breach.

    Args:
      index_change: Session index change.
      drawdown: Current drawdown.
      fno_banned: Whether the symbol is banned.
      ack_latency: Outstanding acknowledgement latency, or None.

    Returns:
      Detail text when the drawdown is at or beyond the limit.
    '''
    del index_change, fno_banned, ack_latency
    self.calls += 1
    self.drawdown = drawdown
    if self.refuse_breach:
      raise RuntimeError('state file is truncated; refusing to answer')
    if drawdown >= self.thresholds.max_drawdown:
      return ('drawdown 20.00% against a 15.00% limit',)
    return ()


class BareSwitch:
  '''A switch handle that offers no pre-defined limits.

  The protocol the API reads has an optional part: ``breached`` answers
  the operator's question and the thresholds are supplementary detail.
  This stand-in exists to pin that the supplementary part degrades to an
  empty mapping rather than failing the request.

  Attributes:
    algo_id: Algo ID the handle belongs to.
  '''

  algo_id: str = 'BARE0001'

  @property
  def tripped(self) -> bool:
    '''Return whether the switch is latched.'''
    return False

  @property
  def trading_enabled(self) -> bool:
    '''Return whether an order may be released.'''
    return True

  def breached(self, index_change, drawdown, fno_banned, ack_latency):
    '''Report no breached conditions.

    Args:
      index_change: Session index change.
      drawdown: Current drawdown.
      fno_banned: Whether the symbol is banned.
      ack_latency: Outstanding acknowledgement latency, or None.

    Returns:
      An empty tuple.
    '''
    del index_change, drawdown, fno_banned, ack_latency
    return ()


class FakeServer:
  '''A server stand-in that serves once and then takes an interrupt.

  Attributes:
    served: Whether ``serve_forever`` was called.
    closed: Whether ``server_close`` was called.
  '''

  served: bool = False
  closed: bool = False

  @property
  def server_address(self):
    '''Return a bound address without binding anything.

    Args:
      None.

    Returns:
      Tuple of (host, port).
    '''
    return (api.default_host, 8765)

  def serve_forever(self):
    '''Record the call and stop the way Ctrl-C would.

    Args:
      None.

    Raises:
      KeyboardInterrupt: Always.
    '''
    self.served = True
    raise KeyboardInterrupt

  def server_close(self):
    '''Record that the socket was released.

    Args:
      None.
    '''
    self.closed = True


def frozen_clock(seconds=0.0):
  '''Return a monotonic clock that reports a fixed value.

  Args:
    seconds: Value every call returns.

  Returns:
    A zero-argument callable.
  '''
  return lambda: seconds


def frozen_now():
  '''Return a wall clock pinned to :data:`FROZEN`.

  Args:
    None.

  Returns:
    A zero-argument callable returning :data:`FROZEN`.
  '''
  return lambda: FROZEN


def service(**kwargs):
  '''Return an :class:`~stock_rl.api.ApiService` with test seams.

  Args:
    **kwargs: Passed through to the constructor.

  Returns:
    The service, with an injected clock and wall clock unless the caller
    supplied either.
  '''
  kwargs.setdefault('clock', frozen_clock(10.0))
  kwargs.setdefault('now', frozen_now())
  return api.ApiService(**kwargs)


def get(svc, path):
  '''Return the decoded JSON body of a GET.

  Args:
    svc: Service to query.
    path: Route to request.

  Returns:
    The parsed payload.
  '''
  response = api.dispatch(svc, 'GET', path)
  assert response.status == 200, response.body
  return json.loads(response.body)


def post(svc, body):
  '''Return the response to a POST with a raw body.

  Args:
    svc: Service to query.
    body: Raw request body.

  Returns:
    The response, undecoded.
  '''
  return api.dispatch(svc, 'POST', '/api/backtest', body)


class TestHealth:
  '''Liveness, version and uptime.'''

  def test_health_returns_200_and_a_status(self):
    response = api.dispatch(service(), 'GET', '/api/health')
    assert response.status == 200
    assert response.content_type.startswith('application/json')
    assert json.loads(response.body)['status'] == 'ok'

  def test_health_reports_version_and_uptime(self):
    payload = get(service(), '/api/health')
    assert payload['version'] == __version__
    assert payload['uptime_seconds'] == 0.0
    assert 'Not financial advice' in payload['advisory']
    assert 'no authentication' in payload['advisory']

  def test_uptime_uses_the_injected_clock_not_a_sleep(self):
    now = [100.0]
    svc = service(clock=lambda: now[0])
    assert svc.uptime == 0.0
    now[0] = 130.5
    assert get(svc, '/api/health')['uptime_seconds'] == 30.5
    now[0] = 131.0
    assert svc.uptime == 31.0

  def test_uptime_never_reports_negative(self):
    svc = service(clock=lambda: -5.0)
    assert svc.uptime == 0.0

  def test_health_counts_the_loaded_universe(self):
    payload = get(service(panels=three_panels()), '/api/health')
    assert payload['symbols'] == 3
    assert payload['bars'] == 3 * (WINDOW + 40)

  def test_health_names_an_unwired_kill_switch(self):
    assert get(service(), '/api/health')['kill_switch'] == 'not_wired'

  def test_health_reports_a_latched_switch(self):
    payload = get(service(kill_switch=FakeSwitch(latched=True)),
                  '/api/health')
    assert payload['kill_switch'] == 'TRIPPED'
    assert payload['trading_enabled'] is False

  def test_top_must_be_positive(self):
    with pytest.raises(ValueError, match='top must be >= 1'):
      service(top=0)


class TestEmptyPortfolio:
  '''Every endpoint must be valid JSON with its keys on an empty book.

  An empty portfolio is the state a fresh server is in, so these are the
  requests a user makes before anything has run. If any of them raised,
  the dashboard would show a blank panel on first load.
  '''

  @pytest.mark.parametrize(('route', 'keys'), [
    ('/api/health', {'status', 'version', 'uptime_seconds', 'symbols'}),
    ('/api/signals', {'count', 'signals', 'generated_at',
                      'trading_enabled', 'lookback', 'skip'}),
    ('/api/equity', {'equity', 'points', 'sharpe', 'max_drawdown',
                     'turnover', 'total_cost'}),
    ('/api/positions', {'positions', 'cash', 'drawdown', 'value',
                        'count', 'as_of'}),
    ('/api/baselines', {'strategies', 'count', 'best_sharpe'}),
    ('/api/risk', {'kill_switch', 'var', 'breaches', 'thresholds'}),
  ])
  def test_endpoint_shape_on_an_empty_portfolio(self, route, keys):
    payload = get(service(), route)
    assert keys <= set(payload)

  def test_signals_are_an_empty_list_not_an_error(self):
    payload = get(service(), '/api/signals')
    assert payload['signals'] == []
    assert payload['count'] == 0

  def test_equity_curve_is_empty_and_says_so(self):
    payload = get(service(), '/api/equity')
    assert payload['equity'] == []
    assert payload['points'] == 0
    assert payload['source'] == 'none'

  def test_positions_are_empty_with_zero_cash(self):
    payload = get(service(), '/api/positions')
    assert payload['positions'] == []
    assert payload['cash'] == 0.0
    assert payload['drawdown'] == 0.0

  def test_every_baseline_is_listed_even_when_it_cannot_run(self):
    payload = get(service(), '/api/baselines')
    listed = {row['name'] for row in payload['strategies']}
    assert listed == set(api.strategies())
    assert payload['count'] == 5
    for row in payload['strategies']:
      assert row['status'] == 'skipped'
      assert row['error'] == 'no price panels are loaded'
    assert payload['best_sharpe'] is None

  def test_var_reports_no_observations_rather_than_zero_risk(self):
    payload = get(service(), '/api/risk')
    assert payload['var']['observations'] == 0
    assert payload['var']['one_period_fraction'] == 0.0

  def test_breaches_are_empty_when_no_switch_is_wired(self):
    assert get(service(), '/api/risk')['breaches'] == []
    assert get(service(), '/api/risk')['thresholds'] == {}


class TestBaselines:
  '''The control arm every learned policy has to clear.'''

  def test_every_provider_is_listed_with_a_status(self):
    payload = get(service(panels=three_panels()), '/api/baselines')
    rows = payload['strategies']
    assert payload['count'] == 5
    assert {row['name'] for row in rows} == set(api.strategies())
    for row in rows:
      assert row['status'] in ('ok', 'skipped', 'error')
      if row['status'] == 'ok':
        assert row['points'] == WINDOW + 40
        for key in ('sharpe', 'max_drawdown', 'turnover', 'total_cost',
                    'total_return_multiple', 'rebalances'):
          assert isinstance(row[key], (int, float))
      else:
        assert row['error']
    # Equal weight is the control and only reads the panel keys, so it
    # runs whatever the price-based providers happen to be doing.
    control = [row for row in rows if row['name'] == 'equal_weight']
    assert control[0]['status'] == 'ok'

  def test_best_sharpe_names_one_of_the_measured_rows(self):
    payload = get(service(panels=three_panels()), '/api/baselines')
    measured = [row['name'] for row in payload['strategies']
                if row['status'] == 'ok']
    assert payload['best_sharpe'] in measured

  def test_equal_weight_is_present_as_the_control(self):
    payload = get(service(panels=three_panels()), '/api/baselines')
    assert 'equal_weight' in {row['name'] for row in payload['strategies']}

  def test_misaligned_panels_skip_every_row_instead_of_failing(self):
    panels = three_panels()
    panels['AAA'] = panels['AAA'][:80]
    payload = get(service(panels=panels), '/api/baselines')
    rows = {row['name']: row for row in payload['strategies']}
    assert len(rows) == 5
    for row in rows.values():
      assert row['status'] == 'skipped'
      assert 'aligned' in row['error']
    assert payload['best_sharpe'] is None

  def test_panels_too_short_to_rebalance_skip_every_row(self):
    payload = get(service(panels={'AAA': panel(5, 10.0, 0.0, 9)}),
                  '/api/baselines')
    for row in payload['strategies']:
      assert row['status'] == 'skipped'
      assert 'leaves no room' in row['error']

  def test_a_provider_that_raises_is_contained_in_its_row(self, monkeypatch):
    def exploding(panels):
      '''Raise the way a mis-indexed provider does.

      Args:
        panels: Visible price panels.

      Raises:
        IndexError: Always, standing in for an off-by-one in another
          module's slicing.
      '''
      del panels
      raise IndexError('list index out of range')

    def working(panels):
      '''Return an equal-weight book.

      Args:
        panels: Visible price panels.

      Returns:
        One weight per symbol.
      '''
      return {symbol: 0.1 for symbol in panels}

    monkeypatch.setattr(api, 'strategies',
                        lambda: {'exploding': exploding,
                                 'working': working})
    payload = get(service(panels=three_panels()), '/api/baselines')
    rows = {row['name']: row for row in payload['strategies']}
    assert rows['exploding']['status'] == 'error'
    assert 'IndexError' in rows['exploding']['error']
    assert rows['working']['status'] == 'ok'
    assert payload['best_sharpe'] == 'working'

  def test_a_provider_that_raises_is_a_500_not_a_422(self, monkeypatch):
    def exploding(panels):
      '''Raise instead of returning weights.

      Args:
        panels: Visible price panels.

      Raises:
        IndexError: Always.
      '''
      del panels
      raise IndexError('list index out of range')

    monkeypatch.setattr(api, 'strategies', lambda: {'exploding': exploding})
    svc = service(panels=three_panels())
    response = post(svc, b'{"strategy": "exploding"}')
    assert response.status == 500
    payload = json.loads(response.body)
    assert 'IndexError' in payload['error']
    assert 'Traceback' not in response.body.decode()
    # The book is untouched by a run that never happened.
    assert get(svc, '/api/equity')['source'] == 'none'


class TestEquity:
  '''Sharpe and drawdown come from metrics.py, not from a second copy.'''

  def state_with_curve(self, equity, returns):
    '''Return a service whose book carries the given history.

    Args:
      equity: Equity curve.
      returns: Per-bar returns.

    Returns:
      The service.
    '''
    return service(state=api.PortfolioState(
      capital=1_000_000.0, value=1_000_000.0, equity=equity,
      returns=returns, turnover=0.75, total_cost=1234.5))

  def test_sharpe_equals_metrics_sharpe(self):
    equity = [1.0, 1.02, 1.01, 1.05, 1.04, 1.09]
    returns = [0.0, 0.02, -0.0099, 0.0396, -0.0095, 0.0481]
    payload = get(self.state_with_curve(equity, returns), '/api/equity')
    assert payload['sharpe'] == sharpe_ratio(returns)
    assert payload['sharpe'] != 0.0

  def test_max_drawdown_equals_metrics_drawdown(self):
    equity = [1.0, 1.2, 0.9, 1.1, 0.8, 1.5]
    payload = get(self.state_with_curve(equity, []), '/api/equity')
    assert payload['max_drawdown'] == max_drawdown(equity).depth
    assert payload['max_drawdown'] == pytest.approx(1 - 0.8 / 1.2)

  def test_turnover_and_cost_are_reported_not_recomputed(self):
    payload = get(self.state_with_curve([1.0], []), '/api/equity')
    assert payload['turnover'] == 0.75
    assert payload['total_cost'] == 1234.5

  def test_growth_multiple_is_metrics_total_return(self):
    returns = [0.01, -0.02, 0.03]
    payload = get(self.state_with_curve([1.0], returns), '/api/equity')
    assert payload['total_return_multiple'] == pytest.approx(
      1.01 * 0.98 * 1.03)

  def test_endpoint_delegates_to_metrics_module(self, monkeypatch):
    calls = []

    def fake_sharpe(returns, risk_free=0.0, periods=252):
      '''Stand in for the real Sharpe and record the call.

      Args:
        returns: Periodic returns.
        risk_free: Periodic risk-free rate.
        periods: Periods per year.

      Returns:
        A sentinel that cannot be confused with a computed ratio.
      '''
      del risk_free, periods
      calls.append(list(returns))
      return 42.0

    monkeypatch.setattr(api, 'sharpe_ratio', fake_sharpe)
    payload = get(self.state_with_curve([1.0, 1.1], [0.0, 0.1]),
                  '/api/equity')
    assert payload['sharpe'] == 42.0
    assert calls == [[0.0, 0.1]]

  def test_endpoint_delegates_drawdown_to_metrics_module(self, monkeypatch):
    class Stub:
      '''A Drawdown-shaped stand-in.'''

      depth = 0.5

    monkeypatch.setattr(api, 'max_drawdown', lambda curve: Stub())
    payload = get(self.state_with_curve([1.0, 0.9], []), '/api/equity')
    assert payload['max_drawdown'] == 0.5


class TestPositions:
  '''Weights, cash and drawdown.'''

  def test_reports_held_and_target_weights(self):
    state = api.PortfolioState(
      capital=2_000_000.0, value=2_500_000.0, cash=500_000.0,
      weights={'AAA': 0.4}, targets={'AAA': 0.2})
    payload = get(service(state=state), '/api/positions')
    assert payload['positions'] == [{
      'symbol': 'AAA', 'weight': 0.4, 'target_weight': 0.2,
      'value': 1_000_000.0,
    }]
    assert payload['invested'] == 0.4
    assert payload['drawdown'] == 0.0

  def test_symbols_in_targets_only_still_appear(self):
    state = api.PortfolioState(targets={'BBB': 0.1})
    payload = get(service(state=state), '/api/positions')
    assert payload['count'] == 1
    assert payload['positions'][0]['symbol'] == 'BBB'
    assert payload['positions'][0]['weight'] == 0.0

  def test_drawdown_tracks_the_equity_curve(self):
    state = api.PortfolioState(equity=[1.0, 1.4, 1.1])
    payload = get(service(state=state), '/api/positions')
    assert payload['drawdown'] == max_drawdown([1.0, 1.4, 1.1]).depth


class TestSignals:
  '''Action, confidence and reasons.'''

  def test_one_row_per_symbol_with_the_documented_keys(self):
    payload = get(service(panels=three_panels()), '/api/signals')
    assert payload['count'] == 3
    assert payload['lookback'] == api.signal_lookback
    assert payload['skip'] == api.signal_skip
    for row in payload['signals']:
      assert set(row) == {'symbol', 'action', 'confidence', 'reasons',
                          'target_weight', 'held_weight', 'score', 'rank'}
      assert row['action'] in ('BUY', 'HOLD', 'SELL')
      assert 0.0 <= row['confidence'] <= 1.0
      assert row['reasons']

  def test_ranks_agree_with_actions(self):
    panels = three_panels()
    svc = service(panels=panels)
    rows = {row['symbol']: row
            for row in get(svc, '/api/signals')['signals']}
    assert rows['AAA']['rank'] == 1
    assert rows['CCC']['rank'] == 3
    assert rows['AAA']['action'] == 'BUY'
    assert rows['AAA']['confidence'] == 1.0
    assert rows['CCC']['confidence'] == 0.0
    assert rows['AAA']['score'] > rows['CCC']['score']

  def test_holding_a_target_weight_reports_hold(self):
    svc = service(panels=three_panels(),
                  state=api.PortfolioState(weights={'AAA': 0.1}))
    rows = {row['symbol']: row
            for row in get(svc, '/api/signals')['signals']}
    assert rows['AAA']['action'] == 'HOLD'
    assert 'no trade is implied' in rows['AAA']['reasons'][-1]

  def test_overweighting_a_dropped_symbol_reports_sell(self):
    panels = three_panels()
    svc = service(panels=panels,
                  state=api.PortfolioState(weights={'CCC': 0.1}), top=1)
    rows = {row['symbol']: row
            for row in get(svc, '/api/signals')['signals']}
    assert rows['AAA']['action'] == 'BUY'
    assert rows['CCC']['action'] == 'SELL'
    assert 'reduce' in rows['CCC']['reasons'][-1]

  def test_short_history_is_reported_rather_than_scored(self):
    svc = service(panels={'AAA': panel(10, 10.0, 0.0, 4)})
    row = get(svc, '/api/signals')['signals'][0]
    assert row['action'] == 'HOLD'
    assert row['score'] is None
    assert row['rank'] is None
    assert row['confidence'] == 0.0
    assert 'momentum window' in row['reasons'][0]

  def test_a_flat_panel_produces_no_conviction(self):
    flat = panel(WINDOW + 5, 100.0, 0.0, 21)
    panels = {'AAA': list(flat), 'BBB': list(flat)}
    svc = service(panels=panels)
    rows = get(svc, '/api/signals')['signals']
    assert len(rows) == 2
    assert rows[0]['score'] == rows[1]['score']
    for row in rows:
      assert row['confidence'] == 0.0
      assert 'identical' in ' '.join(row['reasons'])

  def test_a_lone_scorable_symbol_has_no_confidence_to_report(self):
    panels = {'AAA': panel(WINDOW + 10, 100.0, 0.001, 31),
              'NEW': panel(10, 50.0, 0.0, 32)}
    svc = service(panels=panels)
    rows = {row['symbol']: row
            for row in get(svc, '/api/signals')['signals']}
    assert rows['AAA']['rank'] == 1
    assert rows['AAA']['confidence'] == 0.0
    assert rows['AAA']['action'] == 'BUY'
    assert rows['NEW']['score'] is None

  def test_injected_signal_source_replaces_the_default(self):
    wanted = [api.Signal(symbol='ZZZ', action='HOLD', confidence=0.5,
                         reasons=('injected',), target_weight=0.0,
                         held_weight=0.0)]
    svc = service(signal_source=lambda panels, held: wanted)
    assert get(svc, '/api/signals')['signals'] == [wanted[0].to_json()]

  def test_signals_report_a_latched_switch(self):
    svc = service(panels=three_panels(),
                  kill_switch=FakeSwitch(latched=True))
    payload = get(svc, '/api/signals')
    assert payload['trading_enabled'] is False
    assert svc.trading_enabled() is False

  def test_to_json_rounds_without_losing_the_action(self):
    signal = api.Signal(symbol='AAA', action='BUY', confidence=0.123456,
                        reasons=('because',), target_weight=0.1,
                        held_weight=0.0, score=0.5, rank=1)
    payload = signal.to_json()
    assert payload['confidence'] == 0.1235
    assert payload['score'] == 0.5
    assert payload['action'] == 'BUY'


class TestRisk:
  '''Kill-switch state, VaR and breached conditions.'''

  def test_reports_an_armed_switch_with_its_limits(self):
    switch = FakeSwitch()
    payload = get(service(kill_switch=switch), '/api/risk')
    assert payload['kill_switch']['status'] == 'armed'
    assert payload['kill_switch']['tripped'] is False
    assert payload['kill_switch']['algo_id'] == 'TESTALGO01'
    assert payload['thresholds'] == {
      'index_fall': 0.05, 'max_drawdown': 0.15, 'ack_latency': 2.0,
      'fno_ban': True,
    }
    assert payload['breaches'] == []

  def test_reports_a_latched_switch_and_its_trips(self):
    switch = FakeSwitch(latched=True, trips=(FakeSwitch(),))
    payload = get(service(kill_switch=switch), '/api/risk')
    assert payload['kill_switch']['status'] == 'TRIPPED'
    assert payload['kill_switch']['trading_enabled'] is False
    assert payload['kill_switch']['trips'] == [{
      'code': '', 'detail': '', 'at': '', 'observed': 0.0}]

  def test_asks_the_switch_which_conditions_are_breached(self):
    switch = FakeSwitch()
    state = api.PortfolioState(equity=[1.0, 1.0, 0.5])
    payload = get(service(state=state, kill_switch=switch), '/api/risk')
    assert switch.calls == 1
    assert switch.drawdown == max_drawdown([1.0, 1.0, 0.5]).depth
    assert payload['breaches'] == ['drawdown 20.00% against a 15.00% limit']
    assert payload['kill_switch']['status'] == 'armed'

  def test_an_unreadable_switch_is_reported_not_smoothed(self):
    payload = get(service(kill_switch=FakeSwitch(broken=True)), '/api/risk')
    view = payload['kill_switch']
    assert view['status'] == 'unreadable'
    assert view['tripped'] is None
    assert view['trading_enabled'] is None
    assert 'refusing to re-arm' in view['error']

  def test_limits_are_reported_even_when_the_state_is_not(self):
    payload = get(service(kill_switch=FakeSwitch(broken=True)), '/api/risk')
    # The thresholds are ordinary attributes and still readable; only the
    # latch state refuses. Reporting the limits alongside the refusal is
    # more useful than dropping the whole block.
    assert payload['thresholds'] == {
      'index_fall': 0.05, 'max_drawdown': 0.15, 'ack_latency': 2.0,
      'fno_ban': True,
    }
    assert payload['breaches'] == []

  def test_a_switch_that_refuses_to_answer_reports_no_breaches(self):
    payload = get(service(kill_switch=FakeSwitch(refuse_breach=True)),
                  '/api/risk')
    assert payload['breaches'] == []
    assert payload['kill_switch']['status'] == 'armed'

  def test_limits_absent_from_the_handle_are_omitted_not_guessed(self):
    payload = get(service(kill_switch=BareSwitch()), '/api/risk')
    assert payload['thresholds'] == {}
    assert payload['breaches'] == []
    assert payload['kill_switch']['algo_id'] == 'BARE0001'

  def test_var_is_the_historical_percentile_of_the_returns(self):
    returns = [-0.05, -0.04, -0.03, -0.02, -0.01, 0.0, 0.01, 0.02]
    fraction, observed = api.historical_var(returns)
    assert observed == 8
    # floor(0.05 * 8) = 0, so the worst of eight returns is the reading.
    assert fraction == pytest.approx(0.05)

  def test_var_moves_to_the_second_worst_over_a_longer_sample(self):
    returns = [-0.05, -0.04, -0.03, -0.02, -0.01,
               0.0, 0.01, 0.02, 0.03, 0.04,
               0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14]
    fraction, observed = api.historical_var(returns)
    assert observed == 20
    # floor(0.05 * 20) = 1, the second-worst return.
    assert fraction == pytest.approx(0.04)

  def test_var_never_goes_negative_on_an_all_up_series(self):
    fraction, observed = api.historical_var([0.01, 0.02, 0.03])
    assert observed == 3
    assert fraction == 0.0

  def test_var_in_rupees_uses_the_book_value(self):
    state = api.PortfolioState(capital=10_000_000.0, value=12_000_000.0,
                               equity=[1.0, 1.2],
                               returns=[0.0, -0.2])
    payload = get(service(state=state), '/api/risk')
    assert payload['var']['rupees'] == pytest.approx(
      payload['var']['one_period_fraction'] * 12_000_000.0)
    assert payload['var']['confidence'] == 0.95

  def test_var_note_says_where_var_does_not_belong(self):
    note = get(service(), '/api/risk')['var']['note']
    assert 'NSE 11.1' in note
    assert 'not among the 16 pre-trade RMS checks' in note

  def test_var_rejects_a_confidence_outside_the_unit_interval(self):
    for level in (0.0, 1.0, -0.5, 2.0):
      with pytest.raises(ValueError, match='confidence'):
        api.historical_var([0.01, -0.01], level)

  def test_var_ignores_non_finite_returns(self):
    fraction, observed = api.historical_var(
      [float('nan'), float('inf'), -0.02, 0.01])
    assert observed == 2
    assert fraction == pytest.approx(0.02)


class TestBacktest:
  '''Validation, the injected runner and the state it leaves behind.'''

  def test_runs_the_requested_strategy(self):
    svc = service(panels=three_panels())
    body = b'{"strategy": "equal_weight", "capital": 5000000}'
    response = post(svc, body)
    assert response.status == 200
    payload = json.loads(response.body)
    assert payload['strategy'] == 'equal_weight'
    assert payload['request']['capital'] == 5_000_000.0
    assert payload['points'] == WINDOW + 40
    assert payload['sharpe'] == sharpe_ratio(payload['returns'])
    assert payload['max_drawdown'] == max_drawdown(payload['equity']).depth
    assert payload['symbols'] == ['AAA', 'BBB', 'CCC']
    assert payload['weights'] == {'AAA': 0.1, 'BBB': 0.1, 'CCC': 0.1}
    assert payload['rebalances'] > 0
    assert payload['total_cost'] > 0.0

  def test_a_result_replaces_the_reported_book(self):
    svc = service(panels=three_panels())
    body = (b'{"strategy": "equal_weight", "symbols": ["AAA"], '
            b'"capital": 2000000}')
    assert post(svc, body).status == 200
    positions = get(svc, '/api/positions')
    assert [row['symbol'] for row in positions['positions']] == ['AAA']
    assert positions['capital'] == 2_000_000.0
    assert positions['cash'] == pytest.approx(1_800_000.0)
    assert get(svc, '/api/equity')['source'] == 'backtest'

  def test_injected_runner_receives_the_validated_request(self):
    seen = []

    def runner(request):
      '''Record the request and return a fixed result.

      Args:
        request: The validated trigger.

      Returns:
        A minimal JSON-safe payload.
      '''
      seen.append(request)
      return {'equity': [1.0, 1.5], 'returns': [0.0, 0.5],
              'weights': {'AAA': 0.25}, 'turnover': 1.5,
              'total_cost': 99.0}

    svc = service(panels=three_panels(), backtest_runner=runner,
                  state=api.PortfolioState(equity=[1.0, 1.4],
                                           weights={'STALE': 0.9}))
    response = post(svc, b'{"capital": 1234, "max_weight": 0.5}')
    assert response.status == 200
    assert seen[0].capital == 1234.0
    assert seen[0].max_weight == 0.5
    assert json.loads(response.body)['request']['max_weight'] == 0.5
    positions = get(svc, '/api/positions')
    assert positions['positions'] == [{
      'symbol': 'AAA', 'weight': 0.25, 'target_weight': 0.25,
      'value': 0.25 * 1234.0 * 1.5,
    }]
    assert 'STALE' not in {row['symbol'] for row in positions['positions']}

  def test_a_symbol_subset_is_honoured(self):
    svc = service(panels=three_panels())
    body = b'{"strategy": "equal_weight", "symbols": ["CCC", "AAA"]}'
    payload = json.loads(post(svc, body).body)
    assert payload['symbols'] == ['AAA', 'CCC']
    assert payload['request']['symbols'] == ['AAA', 'CCC']

  def test_an_empty_body_is_a_400_with_a_sentence(self):
    response = post(service(panels=three_panels()), b'')
    assert response.status == 400
    payload = json.loads(response.body)
    assert payload['error'] == (
      'request body is empty; POST a JSON object')
    assert payload['status'] == 400
    assert 'Traceback' not in response.body.decode()

  @pytest.mark.parametrize(('body', 'fragment'), [
    (b'not json at all', 'not valid JSON'),
    (b'[]', 'must be a JSON object'),
    (b'"a string"', 'must be a JSON object'),
    (b'{"max_weight_typo": 0.1}', 'unrecognised field'),
    (b'{"capital": 0}', 'capital must be a positive finite number'),
    (b'{"capital": -1}', 'capital must be a positive finite number'),
    (b'{"capital": true}', 'capital must be a positive finite number'),
    (b'{"capital": "1e6"}', 'capital must be a positive finite number'),
    (b'{"max_weight": 0}', 'max_weight must be in (0, 1]'),
    (b'{"max_weight": 1.5}', 'max_weight must be in (0, 1]'),
    (b'{"rebalance_days": 0}', 'rebalance_days must be >= 1'),
    (b'{"rebalance_days": 2.5}', 'rebalance_days must be a whole number'),
    (b'{"history": -3}', 'history must be >= 1'),
    (b'{"strategy": "vibes"}', 'unknown strategy'),
    (b'{"strategy": ""}', 'strategy must be a non-empty string'),
    (b'{"symbols": "AAA"}', 'symbols must be a list'),
    (b'{"symbols": ["ZZZ"]}', 'is not loaded'),
    (b'{"symbols": [""]}', 'must be non-empty strings'),
    (b'{"symbols": [7]}', 'must be non-empty strings'),
  ])
  def test_malformed_bodies_are_400_not_tracebacks(self, body, fragment):
    response = post(service(panels=three_panels()), body)
    assert response.status == 400
    text = response.body.decode()
    assert fragment in text
    assert 'Traceback' not in text

  def test_a_valid_request_with_no_data_is_422(self):
    response = post(service(), b'{"strategy": "equal_weight"}')
    assert response.status == 422
    assert 'no price panels matched' in json.loads(response.body)['error']

  def test_an_impossible_history_is_422_not_a_traceback(self):
    svc = service(panels={'AAA': panel(10, 10.0, 0.0, 5)})
    response = post(svc, b'{"history": 60}')
    assert response.status == 422
    error = json.loads(response.body)['error']
    assert 'leaves no room' in error
    assert 'Traceback' not in response.body.decode()

  def test_an_unknown_symbol_on_loaded_panels_is_400(self):
    svc = service(panels=three_panels())
    response = post(svc, b'{"symbols": ["AAA", "NOPE"]}')
    assert response.status == 400
    assert 'is not loaded' in json.loads(response.body)['error']

  def test_a_runner_that_refuses_is_422(self):
    def runner(request):
      '''Refuse the request.

      Args:
        request: The validated trigger.

      Raises:
        api.Refused: Always.
      '''
      del request
      raise api.Refused('nothing to run')

    response = post(service(backtest_runner=runner), b'{}')
    assert response.status == 422
    assert json.loads(response.body)['error'] == 'nothing to run'

  def test_run_baseline_rejects_an_unknown_strategy_directly(self):
    svc = service(panels=three_panels())
    with pytest.raises(api.Refused, match='unknown strategy'):
      api.run_baseline(svc, api.BacktestRequest(strategy='nope'))


class TestRouting:
  '''Unknown routes, wrong methods and the document.'''

  @pytest.mark.parametrize('route', ['/', '/index.html', '/api/health',
                                      '/api/signals', '/api/equity',
                                      '/api/positions', '/api/baselines',
                                      '/api/risk'])
  def test_every_documented_route_answers(self, route):
    response = api.dispatch(service(), 'GET', route)
    assert response.status == 200

  def test_routes_table_matches_the_documented_set(self):
    assert set(api.routes) == {
      '/', '/index.html', '/api/health', '/api/signals', '/api/equity',
      '/api/positions', '/api/baselines', '/api/risk', '/api/backtest',
    }
    assert api.routes['/api/backtest'] == frozenset({'POST'})
    for route, allowed in api.routes.items():
      if route != '/api/backtest':
        assert 'GET' in allowed

  @pytest.mark.parametrize('path', ['/api/nope', '/nonsense', '/api',
                                     '/favicon.ico'])
  def test_unknown_route_is_404(self, path):
    response = api.dispatch(service(), 'GET', path)
    assert response.status == 404
    assert json.loads(response.body)['error'].startswith('no such endpoint')

  def test_wrong_method_is_405_not_404(self):
    response = api.dispatch(service(), 'GET', '/api/backtest')
    assert response.status == 405
    assert 'allowed' in json.loads(response.body)['error']

  def test_post_to_a_read_only_route_is_405(self):
    response = api.dispatch(service(), 'POST', '/api/equity', b'{}')
    assert response.status == 405

  def test_head_is_served_by_the_get_it_describes(self):
    head = api.dispatch(service(), 'HEAD', '/api/health')
    plain = api.dispatch(service(), 'HEAD', '/api/health')
    assert head.status == plain.status == 200
    assert head.body == plain.body

  def test_head_on_a_post_route_is_405(self):
    response = api.dispatch(service(), 'HEAD', '/api/backtest')
    assert response.status == 405

  def test_query_strings_and_trailing_slashes_are_tolerated(self):
    assert api.dispatch(service(), 'GET', '/api/health?x=1').status == 200
    assert api.dispatch(service(), 'GET', '/api/health/').status == 200
    assert api.dispatch(service(), 'GET', '/api/health#top').status == 200

  def test_a_repeated_slash_collapses_to_the_root(self):
    response = api.dispatch(service(), 'GET', '//')
    assert response.status == 200
    assert response.body.decode() == index_html()

  def test_an_empty_path_collapses_to_the_root(self):
    assert api.dispatch(service(), 'GET', '').status == 200

  def test_the_root_serves_the_dashboard_document(self):
    response = api.dispatch(service(), 'GET', '/')
    assert response.status == 200
    assert response.content_type.startswith('text/html')
    assert response.body.decode() == index_html()

  def test_the_dashboard_states_it_is_not_advice(self):
    document = ' '.join(index_html().split())
    assert 'Not financial advice' in document
    assert 'not a recommendation' in document
    # The page carries the exact sentence the API attaches to every
    # payload, compared with whitespace collapsed because HTML wraps.
    assert api.not_advice in document
    for route in ('/api/health', '/api/signals', '/api/equity',
                  '/api/positions', '/api/baselines', '/api/risk'):
      assert get(service(), route)['not_advice'] == api.not_advice

  @pytest.mark.parametrize('args', [
    ('GET', '/nope', b''), ('GET', '/api/backtest', b''),
    ('POST', '/api/backtest', b'{bad'),
  ])
  def test_even_an_error_carries_the_disclaimer(self, args):
    method, path, body = args
    response = api.dispatch(service(), method, path, body)
    assert response.status >= 400
    assert json.loads(response.body)['not_advice'] == api.not_advice

  def test_the_dashboard_reaches_no_origin_but_loopback(self):
    # The SVG namespace URI is an identifier, never a fetch, so it is
    # excluded explicitly rather than allowed by a blanket exception.
    ignorable = ('http://www.w3.org/2000/svg',)
    pattern = re.compile(r'https?://[^\s"\'<>)]+')
    seen = 0
    for asset in (index_html(), stylesheet(), script()):
      for url in pattern.findall(asset):
        if url in ignorable:
          continue
        seen += 1
        assert url.startswith(('http://127.0.0.1', 'http://localhost')), url
    assert seen > 0

  def test_the_dashboard_loads_no_package_and_no_stylesheet_import(self):
    # Match the at-rule at the start of a line, not the word inside the
    # comment explaining why there is none.
    assert not re.search(r'(?m)^\s*@import', stylesheet())
    assert not re.search(r'(?m)^\s*@font-face', stylesheet())
    assert 'package.json' not in index_html()
    assert 'node_modules' not in index_html()
    assert '<script src="http' not in index_html()
    assert 'href="http' not in index_html()

  def test_a_missing_asset_is_a_500_not_a_traceback(self, monkeypatch):
    def missing():
      '''Stand in for a package whose asset file is absent.

      Args:
        None.

      Raises:
        FileNotFoundError: Always.
      '''
      raise FileNotFoundError('index.html')

    monkeypatch.setattr(api, 'index_html', missing)
    response = api.dispatch(service(), 'GET', '/')
    assert response.status == 500
    assert 'dashboard asset is missing' in json.loads(response.body)['error']


class TestBinding:
  '''Loopback is enforced, not merely defaulted.'''

  def test_default_host_is_loopback(self):
    assert api.default_host == '127.0.0.1'
    assert api.is_loopback(api.default_host)

  @pytest.mark.parametrize('host', ['127.0.0.1', '127.1.2.3', '::1',
                                     'localhost', 'LOCALHOST'])
  def test_loopback_detection_accepts_only_loopback(self, host):
    assert api.is_loopback(host) is True

  @pytest.mark.parametrize('host', ['0.0.0.0', '10.0.0.1', 'example.com',
                                     '', '  ', '169.254.169.254'])
  def test_loopback_detection_rejects_everything_else(self, host):
    assert api.is_loopback(host) is False

  def test_build_server_defaults_to_loopback(self):
    server = api.build_server(service())
    try:
      host, port = api.bound_address(server)
      assert host == '127.0.0.1'
      assert port == 0 or port > 0
    finally:
      server.server_close()

  @pytest.mark.parametrize('host', ['0.0.0.0', '192.168.1.10', '::',
                                     'example.com', ''])
  def test_a_routable_bind_is_refused(self, host):
    with pytest.raises(ValueError, match='loopback'):
      api.build_server(service(), host, 0)

  @pytest.mark.parametrize('host', ['127.0.0.1', '127.0.0.2', 'localhost',
                                   '::1'])
  def test_loopback_names_are_accepted(self, host):
    server = api.build_server(service(), host, 0)
    try:
      bound, port = api.bound_address(server)
      # 'localhost' resolves to 127.0.0.1, so the bound address is
      # compared as a loopback address rather than as the literal given.
      assert api.is_loopback(bound)
      if host != 'localhost':
        assert bound == host
      assert port > 0
    finally:
      server.server_close()

  def test_a_port_outside_the_valid_range_is_refused(self):
    with pytest.raises(ValueError, match='port'):
      api.build_server(service(), '127.0.0.1', 70_000)


class TestLiveSocket:
  '''One real round trip, because the framing is a claim of its own.

  Every other test in this file avoids a socket. This class opens one, on
  an ephemeral loopback port, to check the thing a pure function cannot:
  that bytes go out, a length is declared, and the connection survives
  being reused.
  '''

  @pytest.fixture(name='live')
  def live(self):
    '''Return a running server and its address.

    Args:
      None.

    Yields:
      Tuple of (service, host, port).
    '''
    svc = service(panels=three_panels())
    server = api.build_server(svc, api.default_host, 0)
    host, port = api.bound_address(server)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
      yield svc, host, port
    finally:
      server.shutdown()
      server.server_close()
      thread.join(timeout=5)

  def request(self, host, port, path, method='GET', body=None):
    '''Send one request over a real socket.

    Args:
      host: Loopback host.
      port: Bound port.
      path: Request path.
      method: HTTP method.
      body: Optional request body.

    Returns:
      Tuple of (status, body bytes).
    '''
    connection = HTTPConnection(host, port, timeout=5)
    try:
      connection.request(method, path, body=body)
      response = connection.getresponse()
      return response.status, response.read()
    finally:
      connection.close()

  def test_health_over_a_socket(self, live):
    _, host, port = live
    status, body = self.request(host, port, '/api/health')
    assert status == 200
    assert json.loads(body)['status'] == 'ok'

  def test_the_connection_survives_being_reused(self, live):
    _, host, port = live
    connection = HTTPConnection(host, port, timeout=5)
    try:
      for _ in range(3):
        connection.request('GET', '/api/health')
        response = connection.getresponse()
        assert response.status == 200
        assert json.loads(response.read())['status'] == 'ok'
    finally:
      connection.close()

  def test_a_post_body_is_read_before_the_404(self, live):
    _, host, port = live
    status, body = self.request(host, port, '/api/nope', 'POST',
                                b'{"capital": 1}')
    assert status == 404
    assert json.loads(body)['status'] == 404
    # The next request on a fresh connection must be unaffected.
    status, _ = self.request(host, port, '/api/health')
    assert status == 200

  def test_a_malformed_post_over_a_socket_is_400(self, live):
    _, host, port = live
    status, body = self.request(host, port, '/api/backtest', 'POST',
                                b'{oops')
    assert status == 400
    assert 'not valid JSON' in json.loads(body)['error']

  def test_head_returns_headers_without_a_body(self, live):
    _, host, port = live
    connection = HTTPConnection(host, port, timeout=5)
    try:
      connection.request('HEAD', '/api/health')
      response = connection.getresponse()
      body = response.read()
      assert response.status == 200
      assert body == b''
      assert int(response.getheader('Content-Length')) > 0
    finally:
      connection.close()

  def test_an_unsupported_method_is_json_not_html(self, live):
    _, host, port = live
    status, body = self.request(host, port, '/api/health', 'DELETE')
    assert status == 501
    assert json.loads(body)['status'] == 501

  def test_a_post_with_no_body_is_400(self, live):
    _, host, port = live
    status, body = self.request(host, port, '/api/backtest', 'POST')
    assert status == 400
    assert 'empty' in json.loads(body)['error']

  def test_a_malformed_length_header_does_not_hang(self, live):
    _, host, port = live
    connection = HTTPConnection(host, port, timeout=5)
    try:
      connection.putrequest('POST', '/api/backtest')
      connection.putheader('Content-Length', 'not-a-number')
      connection.endheaders()
      response = connection.getresponse()
      assert response.status in (400, 411, 413)
      response.read()
    finally:
      connection.close()

  def test_the_root_document_is_served_over_a_socket(self, live):
    _, host, port = live
    status, body = self.request(host, port, '/')
    assert status == 200
    assert body.decode() == index_html()

  def test_an_oversized_body_is_refused_without_being_buffered(self, live):
    _, host, port = live
    assert api.max_body_bytes > 0
    oversize = b'{"capital":1' + b' ' * (api.max_body_bytes + 8) + b'}'
    status, body = self.request(host, port, '/api/backtest', 'POST',
                                oversize)
    assert status == 413
    assert 'exceeds' in json.loads(body)['error']

  def test_the_server_still_answers_after_refusing_a_large_body(self, live):
    _, host, port = live
    oversize = b'{"capital":1' + b' ' * (api.max_body_bytes + 8) + b'}'
    assert self.request(host, port, '/api/backtest', 'POST',
                        oversize)[0] == 413
    assert self.request(host, port, '/api/health')[0] == 200

  def test_a_client_that_vanishes_mid_body_is_survived(self, live):
    _, host, port = live
    connection = HTTPConnection(host, port, timeout=5)
    try:
      connection.putrequest('POST', '/api/backtest')
      connection.putheader('Content-Length', str(api.max_body_bytes + 64))
      connection.endheaders()
      connection.send(b'{"capital":1' + b' ' * 4096)
      # Half-close the write side, so the declared length is never
      # delivered. That is what a dropped upload looks like from here.
      connection.sock.shutdown(socket.SHUT_WR)
    finally:
      connection.close()
    # No answer is owed to a peer that stopped talking. The point is
    # that the drain loop exits and the server keeps serving.
    assert self.request(host, port, '/api/health')[0] == 200


class TestWebAssets:
  '''The asset loader the root route depends on.'''

  def test_every_asset_is_readable(self):
    assert index_html().startswith('<!DOCTYPE html>')
    assert 'svg' in script()
    assert '.card' in stylesheet()
    assert len(stylesheet()) > 200
    assert len(script()) > 200

  def test_a_missing_asset_raises_rather_than_returning_empty(self):
    with pytest.raises(FileNotFoundError):
      asset_text('not-a-real-file.css')

  def test_the_script_has_no_import_or_require(self):
    source = script()
    assert 'import ' not in source
    assert 'require(' not in source
    assert 'fetch(' in source

  def test_the_script_degrades_instead_of_going_blank(self):
    source = script()
    assert 'cannot reach' in source
    assert 'file://' in source
    assert 'role="alert"' in index_html()


class TestCoercion:
  '''The defensive coercion behind the reported book, seen from outside.

  These paths are reached through the public surface on purpose. A
  module-level helper can be renamed without breaking anything, and the
  behaviour that matters is what a caller observes: a junk runner payload
  leaves the book empty rather than half-populated.
  '''

  def junk(self, request):
    '''Return a payload whose every field is the wrong type.

    Args:
      request: The validated trigger.

    Returns:
      A mapping with unusable values throughout.
    '''
    del request
    return {
      'equity': 'not a list',
      'returns': None,
      'weights': ['nope'],
      'turnover': 'lots',
      'total_cost': [1, 2],
    }

  def test_junk_series_leave_the_book_empty(self):
    svc = service(backtest_runner=self.junk)
    assert post(svc, b'{}').status == 200
    equity = get(svc, '/api/equity')
    assert equity['equity'] == []
    assert equity['turnover'] == 0.0
    assert equity['total_cost'] == 0.0
    assert equity['source'] == 'none'
    assert get(svc, '/api/positions')['positions'] == []

  def test_partly_usable_series_keep_only_the_numbers(self):
    def runner(request):
      '''Return a mixed series.

      Args:
        request: The validated trigger.

      Returns:
        A payload mixing usable numbers with junk.
      '''
      del request
      return {'equity': [1, 2.5, True, 'x', None, float('inf'), 1.5],
              'returns': [0, 0.5, 0.25],
              'weights': {'AAA': 0.5, 'BBB': 'x', 'CCC': float('nan')},
              'turnover': 3, 'total_cost': 12.5}

    svc = service(backtest_runner=runner)
    assert post(svc, b'{}').status == 200
    assert get(svc, '/api/equity')['equity'] == [1.0, 2.5, 1.5]
    assert get(svc, '/api/equity')['turnover'] == 3.0
    rows = get(svc, '/api/positions')['positions']
    assert [row['symbol'] for row in rows] == ['AAA']

  def test_non_finite_numbers_are_nulled_before_serialisation(self):
    def runner(request):
      '''Return infinities and NaN in nested positions.

      Args:
        request: The validated trigger.

      Returns:
        A payload full of values JSON cannot represent.
      '''
      del request
      return {'sharpe': float('nan'),
              'nested': {'a': float('inf'), 'b': [1, float('-inf')]},
              'tuple': (1.0, float('nan'))}

    response = post(service(backtest_runner=runner), b'{}')
    text = response.body.decode()
    assert 'NaN' not in text
    assert 'Infinity' not in text
    payload = json.loads(text)
    assert payload['sharpe'] is None
    assert payload['nested'] == {'a': None, 'b': [1, None]}
    assert payload['tuple'] == [1.0, None]

  def test_a_naive_wall_clock_is_reported_as_utc(self):
    svc = service(now=lambda: datetime(2026, 3, 4, 9, 30))
    assert get(svc, '/api/positions')['as_of'] == '2026-03-04T09:30:00+00:00'

  def test_an_aware_wall_clock_is_reported_verbatim(self):
    assert get(service(), '/api/positions')['as_of'] == FROZEN.isoformat()

  def test_panels_for_filters_and_sorts(self):
    svc = service(panels=three_panels())
    assert sorted(svc.panels_for(['CCC', 'AAA'])) == ['AAA', 'CCC']
    assert svc.panels_for([]) == svc.panels
    assert not svc.panels_for(['NOPE'])
    assert svc.symbols == ['AAA', 'BBB', 'CCC']

  def test_panels_are_copied_so_a_later_mutation_is_invisible(self):
    panels = three_panels()
    svc = service(panels=panels)
    panels['AAA'].clear()
    assert svc.symbols == ['AAA', 'BBB', 'CCC']
    assert get(svc, '/api/health')['bars'] == 3 * (WINDOW + 40)

  def test_a_trip_without_fields_still_renders(self):
    switch = FakeSwitch(latched=True, trips=(object(),))
    payload = get(service(kill_switch=switch), '/api/risk')
    assert payload['kill_switch']['trips'] == [{
      'code': '', 'detail': '', 'at': '', 'observed': 0.0}]


class TestMain:
  '''The command-line entry point.'''

  def test_serve_runs_until_interrupted_and_closes(self, monkeypatch):
    stand_in = FakeServer()
    monkeypatch.setattr(api, 'build_server',
                        lambda *args, **kwargs: stand_in)
    api.serve(service())
    assert stand_in.served is True
    assert stand_in.closed is True

  def test_serve_refuses_a_non_loopback_host_before_binding(self):
    with pytest.raises(ValueError, match='loopback'):
      api.serve(service(), '0.0.0.0', 0)

  def test_the_default_service_uses_the_real_wall_clock(self):
    svc = api.ApiService()
    assert get(svc, '/api/positions')['as_of'].endswith('+00:00')

  def test_main_parses_a_data_directory(self, tmp_path):
    csv_text = (
      'timestamp,open,high,low,close,volume\n'
      '2026-01-01,100,101,99,100,10\n'
      '2026-01-02,100,102,100,101,11\n'
    )
    (tmp_path / 'AAA.csv').write_text(csv_text, encoding='utf-8')
    panels = api.load_panels(tmp_path)
    assert list(panels) == ['AAA']
    assert [bar.close for bar in panels['AAA']] == [100.0, 101.0]

  def test_load_panels_rejects_a_file(self, tmp_path):
    target = tmp_path / 'nope'
    target.write_text('x', encoding='utf-8')
    with pytest.raises(ValueError, match='not a directory'):
      api.load_panels(target)

  def test_load_panels_of_an_empty_directory_is_empty(self, tmp_path):
    assert not api.load_panels(tmp_path)

  def test_main_refuses_a_non_loopback_host(self):
    with pytest.raises(ValueError, match='loopback'):
      api.main(['--host', '0.0.0.0'])

  def test_main_serves_loopback_when_asked(self, monkeypatch):
    seen = {}

    def fake_serve(svc, host, port):
      '''Record the call instead of serving.

      Args:
        svc: Service to serve.
        host: Bind address.
        port: Bind port.
      '''
      seen['service'] = svc
      seen['host'] = host
      seen['port'] = port

    monkeypatch.setattr(api, 'serve', fake_serve)
    assert api.main(['--port', '9999']) == 0
    assert seen['host'] == api.default_host
    assert seen['port'] == 9999
    assert seen['service'].symbols == []

  def test_main_loads_a_data_directory(self, tmp_path, monkeypatch):
    (tmp_path / 'AAA.csv').write_text(
      'timestamp,open,high,low,close\n2026-01-01,1,1,1,1\n'
      '2026-01-02,1,1,1,2\n', encoding='utf-8')
    seen = {}
    monkeypatch.setattr(api, 'serve',
                        lambda svc, host, port: seen.update(service=svc))
    api.main(['--data-dir', str(tmp_path)])
    assert seen['service'].symbols == ['AAA']
