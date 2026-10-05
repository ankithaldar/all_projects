#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''The cap a payload reports is the cap the book was built at.

``GET /api/baselines?max_weight=X`` reported ``X`` and handed ``X`` to the
engine, and then called each provider with no arguments at all. Every
provider in :mod:`stock_rl.baselines` carries its own ``max_weight``
default of 0.10, so the provider was the binding constraint and the
requested cap was decoration. ``?max_weight=0.1``, ``0.3`` and ``0.9``
returned one byte-identical Sharpe, and a caller asking for 0.30 got a
0.10 book under a payload claiming 0.30.

Two properties are asserted here, and both are about the book rather than
about how the payload words itself.

**The requested cap reaches the weights.** Two different caps must
produce two different books, for each of the five baselines. A test that
only compared *reported* numbers would have passed on the broken code,
which reported all three caps faithfully.

**The reported cap equals the applied cap.** Every row carries
``applied_max_weight``, read back off the snapshots the backtester
recorded, so the two numbers can be compared against each other. A cap is
a ceiling and not a target: equality is the expected outcome only while
the cap is the binding constraint, i.e. ``symbols * cap < 1``. Above that
the construction itself -- 1/N, inverse volatility -- is what binds, and
the applied number sits below the ceiling. It is never above it, and that
is asserted too.
'''

import inspect
import json
from datetime import datetime, timedelta
from random import Random

import pytest

from stock_rl import api, baselines
from stock_rl.bars import Bar

#: Bars in one panel: the 12-1 momentum window plus room to rebalance.
BARS = api.signal_lookback + api.signal_skip + 1 + 40

#: A cap the engine must clamp to, and the widest one the default 0.10
#: providers were silently applying underneath. Three symbols at 0.30 is
#: 0.90 of capital, so the cap binds rather than the 1/N share.
WIDE_CAP = 0.30

#: The cap every provider in :mod:`stock_rl.baselines` defaults to. A
#: book built at this one is indistinguishable from a broken run, which is
#: why it is only ever used as the *narrow* half of a comparison.
NARROW_CAP = 0.10

#: Symbols in the universe. Three is the smallest count at which 1/N
#: exceeds WIDE_CAP, so the cap rather than the construction binds.
SYMBOLS = ('AAA', 'BBB', 'CCC')


def panel(count, base, drift, seed):
  '''Return one deterministic price panel.

  Args:
    count: Number of bars.
    base: Starting price.
    drift: Mean per-bar return, which sets the momentum ranking.
    seed: Seed for the private Random instance.

  Returns:
    Bars in ascending time order, with each open one bar behind its close
    so a backtest has something to fill at.
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


def rising_universe():
  '''Return three steadily rising panels with a total momentum ranking.

  Every symbol trends up, so all three sit above their 40-bar moving
  average and the trend gate in
  :func:`stock_rl.baselines.trend_filtered_momentum` is open rather than
  filtering the book to nothing. Each drifts at a different rate, so the
  cross-sectional ranking is strict.

  Returns:
    Mapping of symbol to bars, aligned and long enough for the momentum
    window.
  '''
  return {
    'AAA': panel(BARS, 100.0, 0.003, 1),
    'BBB': panel(BARS, 200.0, 0.002, 2),
    'CCC': panel(BARS, 50.0, 0.001, 3),
  }


def service(**kwargs):
  '''Return a service over :func:`rising_universe` with frozen clocks.

  Args:
    **kwargs: Passed through to the constructor.

  Returns:
    The service. Uptime is irrelevant here and would otherwise vary.
  '''
  frozen = datetime(2026, 3, 4, 9, 30, tzinfo=None)
  kwargs.setdefault('panels', rising_universe())
  kwargs.setdefault('clock', lambda: 0.0)
  kwargs.setdefault('now', lambda: frozen)
  return api.ApiService(**kwargs)


def get(svc, path):
  '''Return the decoded body of a GET, asserting it succeeded.

  Args:
    svc: Service to query.
    path: Route, query string included.

  Returns:
    The parsed payload.
  '''
  response = api.dispatch(svc, 'GET', path)
  assert response.status == 200, response.body
  return json.loads(response.body)


def post(svc, body):
  '''Return the parsed body of a POST, asserting it succeeded.

  Args:
    svc: Service to query.
    body: Request fields, encoded here.

  Returns:
    The parsed payload.
  '''
  response = api.dispatch(svc, 'POST', '/api/backtest',
                          json.dumps(body).encode())
  assert response.status == 200, response.body
  return json.loads(response.body)


def rows_at(svc, cap):
  '''Return every baseline row measured at ``cap``, by name.

  Args:
    svc: Service to query.
    cap: Requested per-symbol cap, sent the way a caller sends it.

  Returns:
    Mapping of provider name to row.
  '''
  payload = get(svc, f'/api/baselines?max_weight={cap}')
  assert payload['max_weight'] == cap
  return {row['name']: row for row in payload['strategies']}


def book_at(svc, name, cap):
  '''Return the weights one baseline actually held at ``cap``.

  Args:
    svc: Service to query.
    name: Provider name.
    cap: Requested per-symbol cap.

  Returns:
    Mapping of symbol to weight, from the engine's own snapshot.
  '''
  payload = post(svc, {'strategy': name, 'max_weight': cap})
  assert payload['request']['max_weight'] == cap
  assert payload['weights']
  return payload['weights']


def heaviest(weights):
  '''Return the largest single-symbol weight in a book.

  Args:
    weights: Mapping of symbol to weight.

  Returns:
    The heaviest weight held.
  '''
  return max(weights.values())


class TestTheRequestedCapReachesTheWeights:
  '''Two caps, two books. One cap, one book.'''

  @pytest.mark.parametrize('name', sorted(api.strategies()))
  def test_two_caps_produce_two_different_books(self, name):
    '''The defect, stated as a difference in the book rather than in
    the payload.

    Args:
      name: Provider name under test.
    '''
    svc = service()
    narrow = book_at(svc, name, NARROW_CAP)
    wide = book_at(svc, name, WIDE_CAP)
    assert narrow != wide, (
      f'{name} returned the identical book '
      f'{sorted(narrow.items())} at max_weight={NARROW_CAP} and '
      f'{WIDE_CAP}, so the requested cap never reached the weights and '
      f'the payload was reporting a number nothing in the run used')

  @pytest.mark.parametrize('name', sorted(api.strategies()))
  @pytest.mark.parametrize('cap', [NARROW_CAP, WIDE_CAP])
  def test_the_book_is_built_at_the_requested_cap(self, name, cap):
    '''Every single-symbol weight equals the cap that was asked for.

    Three symbols at 0.30 is 0.90 of capital, so the cap is the binding
    constraint and equality is the only correct answer. Under the defect
    the weights were 0.10 whatever the payload said.

    Args:
      name: Provider name under test.
      cap: Requested per-symbol cap.
    '''
    weights = book_at(service(), name, cap)
    assert heaviest(weights) == pytest.approx(cap)
    assert all(weight <= cap + 1e-12 for weight in weights.values())

  @pytest.mark.parametrize('name', sorted(api.strategies()))
  def test_one_cap_twice_is_one_book(self, name):
    '''The positive control, without which the test above proves nothing.

    Args:
      name: Provider name under test.
    '''
    first = book_at(service(), name, WIDE_CAP)
    second = book_at(service(), name, WIDE_CAP)
    assert first == second
    # And within one service, so a cache or a mutated provider would show.
    again = book_at(service(), name, WIDE_CAP)
    assert again == first

  @pytest.mark.parametrize('cap', [0.5, 0.75, 1.0])
  def test_a_cap_above_the_ceiling_is_not_exceeded(self, cap):
    '''A cap is a ceiling, and the engine is what enforces it.

    Above ``1/N`` the construction binds instead of the cap, so the
    weights sit at a third and not at the requested number. That is
    correct; being *above* the requested cap would not be.

    Args:
      cap: Requested per-symbol cap, above the 1/N share of three symbols.
    '''
    weights = book_at(service(), 'equal_weight', cap)
    assert heaviest(weights) == pytest.approx(1.0 / len(SYMBOLS))
    assert heaviest(weights) < cap


class TestTheReportedCapEqualsTheAppliedCap:
  '''The payload's two numbers agree, per row, for every baseline.'''

  @pytest.mark.parametrize('name', sorted(api.strategies()))
  def test_every_row_is_built_at_the_cap_it_reports(self, name):
    '''Reported equals applied for each of the five baselines.

    Args:
      name: Provider name under test.
    '''
    rows = rows_at(service(), WIDE_CAP)
    row = rows[name]
    measured = row['applied_max_weight']
    assert row['status'] == 'ok', row.get('error')
    assert row['cap_bound'] is True
    assert measured == pytest.approx(WIDE_CAP), (
      f'{name} reported max_weight={WIDE_CAP} but built the book at '
      f'{measured}, which is the defect this file exists for')

  def test_the_applied_cap_moves_with_the_requested_one(self):
    '''The applied cap is a function of the request, not a constant.

    Under the defect this was 0.10 for every requested value, which is
    what made three different requests return one Sharpe.
    '''
    svc = service()
    applied = {}
    for cap in (0.10, 0.15, 0.20, 0.25, 0.30):
      for row in rows_at(svc, cap).values():
        applied.setdefault(row['name'], []).append(
          (cap, row['applied_max_weight']))
    assert applied, 'no baseline rows were produced'
    for name, pairs in applied.items():
      assert len(pairs) == 5, name
      for cap, measured in pairs:
        assert measured == pytest.approx(cap), (
          f'{name} asked for {cap} and built {measured}')
      assert len({measured for _, measured in pairs}) == 5, (
        f'{name} built one book for five different requested caps')

  @pytest.mark.parametrize('cap', [0.05, 0.1, 0.2, 0.3, 0.45, 0.6, 0.9, 1.0])
  def test_the_applied_cap_is_never_above_the_reported_one(self, cap):
    '''The engine is a ceiling no provider can talk its way past.

    Args:
      cap: Requested per-symbol cap.
    '''
    for row in rows_at(service(), cap).values():
      if row['status'] != 'ok':
        continue
      assert row['applied_max_weight'] <= cap + 1e-12, row

  def test_a_row_that_never_ran_reports_no_applied_cap(self):
    '''An absence is None, not zero.

    A book that never rebalanced applied no cap at all, and reporting
    0.0 would be a claim about a portfolio that does not exist.
    '''
    rows = rows_at(service(panels={'AAA': panel(5, 10.0, 0.0, 9)}), 0.30)
    assert len(rows) == len(api.strategies())
    for row in rows.values():
      assert row['status'] == 'skipped'
      assert row.get('applied_max_weight') is None


class TestTheCapIsBoundByKeyword:
  '''By keyword, because the five signatures do not agree on position.'''

  def test_the_cap_arrives_as_a_keyword_not_a_positional(self, monkeypatch):
    '''A positional call over this table is a silent wrong answer.

    The spy records ``*args`` and ``**kwargs`` separately, so the two
    spellings cannot be confused by a later refactor: passing 0.30 in
    fifth position to a provider whose fifth parameter is ``ma_window``
    produced ``sma(closes, 0.1)`` and raised nothing.
    '''
    seen = []

    def spy(panels, *args, **kwargs):
      '''Record how the cap was passed and return a flat book.

      Args:
        panels: Visible price panels.
        *args: Recorded positionally.
        **kwargs: Recorded by keyword.

      Returns:
        One weight per symbol, so the run has a book to measure.
      '''
      seen.append((args, dict(kwargs)))
      return {symbol: 0.5 for symbol in panels}

    monkeypatch.setattr(api, 'strategies', lambda: {'spy': spy})
    rows = rows_at(service(), WIDE_CAP)
    assert rows['spy']['status'] == 'ok', rows['spy'].get('error')
    assert seen, 'the provider was never called'
    for args, kwargs in seen:
      assert args == (), (
        f'the cap was passed positionally as {args!r}; a reordering of '
        f'one baseline signature would feed it to the wrong parameter')
      assert kwargs == {'max_weight': WIDE_CAP}, kwargs

  def test_a_reorder_cannot_feed_the_cap_to_a_moving_average(self,
                                                            monkeypatch):
    '''The exact shape of ``trend_filtered_momentum``'s signature.

    Args:
      monkeypatch: Fixture replacing the provider table.
    '''
    seen = {}

    def shaped(panels, window=252, skip=21, top=10, ma_window=40,
               max_weight=0.10):
      '''Record every parameter the call actually bound.

      Args:
        panels: Visible price panels.
        window: Momentum lookback in bars.
        skip: Most recent bars excluded from momentum.
        top: Names held.
        ma_window: Moving-average window for the regime gate.
        max_weight: Cap on any single weight.

      Returns:
        One weight per symbol.
      '''
      seen.update(window=window, skip=skip, top=top, ma_window=ma_window,
                  max_weight=max_weight)
      return {symbol: max_weight for symbol in panels}

    monkeypatch.setattr(api, 'strategies', lambda: {'shaped': shaped})
    rows = rows_at(service(), WIDE_CAP)
    assert rows['shaped']['status'] == 'ok', rows['shaped'].get('error')
    assert seen['max_weight'] == pytest.approx(WIDE_CAP)
    # The window parameters are the ones a positional cap would have
    # landed on, and none of them may have moved.
    assert seen['window'] == 252
    assert seen['ma_window'] == 40
    assert seen['skip'] == 21
    assert seen['top'] == 10

  @pytest.mark.parametrize('name', sorted(api.strategies()))
  def test_every_baseline_admits_the_cap_by_keyword(self, name):
    '''Checked against the real signatures, not assumed from the count.

    Args:
      name: Provider name under test.
    '''
    provider = api.strategies()[name]
    # pylint: disable=protected-access
    assert api._accepts_cap(provider) is True
    # bind() is the same machinery the call uses; a signature that cannot
    # be bound with this keyword would raise here rather than silently
    # swallowing the cap.
    inspect.signature(provider).bind({}, **{api.cap_keyword: WIDE_CAP})

  def test_the_cap_never_reaches_the_real_trend_window(self):
    '''End to end against ``trend_filtered_momentum`` itself.

    Its fifth parameter is ``ma_window``, and a cap landing there would
    be a moving average over a fraction of a bar rather than an error.
    '''
    parameters = inspect.signature(
      baselines.trend_filtered_momentum).parameters
    assert parameters['ma_window'].default == 40
    assert list(parameters).index('max_weight') == 5
    svc = service()
    panels = svc.panels
    rows = rows_at(svc, WIDE_CAP)
    assert rows['trend_filtered_momentum']['status'] == 'ok'
    # The book the endpoint built is the book the provider produces when
    # *it* is handed the cap, and is not the one it produces by default.
    wide = baselines.trend_filtered_momentum(panels, max_weight=WIDE_CAP)
    default = baselines.trend_filtered_momentum(panels)
    assert heaviest(wide) == pytest.approx(WIDE_CAP)
    assert heaviest(default) == pytest.approx(NARROW_CAP)
    assert rows['trend_filtered_momentum']['applied_max_weight'] == (
      pytest.approx(heaviest(wide)))


class TestAProviderThatCannotTakeACap:
  '''What happens to one that has no ``max_weight`` parameter.

  It is measured, and it is marked. It is not refused: the registry is
  discovered from ``baselines.__all__``, so a provider this module cannot
  introspect may simply derive weights from the panels and hold no
  opinion about the cap at all. What it may not do is leave a number in
  the payload that no book backs, which is why the row reports
  ``cap_bound`` False and the cap the book was actually built at.
  '''

  def test_it_is_measured_and_marked_unbound(self, monkeypatch):
    '''An unbound provider still gets a row, and the row says why.

    Args:
      monkeypatch: Fixture replacing the provider table.
    '''
    def plain(panels):
      '''Return one weight per symbol, accepting no cap at all.

      Args:
        panels: Visible price panels.

      Returns:
        A flat book.
      '''
      return {symbol: 0.25 for symbol in panels}

    monkeypatch.setattr(api, 'strategies', lambda: {'plain': plain})
    rows = rows_at(service(), WIDE_CAP)
    row = rows['plain']
    assert row['status'] == 'ok', row.get('error')
    assert row['cap_bound'] is False
    assert row['applied_max_weight'] == pytest.approx(0.25)
    # pylint: disable=protected-access
    assert api._accepts_cap(plain) is False

  def test_its_applied_cap_is_disclosed_rather_than_the_requested(
      self, monkeypatch):
    '''The disagreement is reported, not papered over.

    This provider imposes a 0.05 cap of its own and the caller asked for
    0.30. The payload carries both numbers, so the mismatch is visible
    in the row that has it rather than only in a module nobody reads.

    Args:
      monkeypatch: Fixture replacing the provider table.
    '''
    def tight(panels):
      '''Return a book capped at 0.05 however it is asked for.

      Args:
        panels: Visible price panels.

      Returns:
        One weight per symbol at a fifth of the requested cap.
      '''
      return {symbol: 0.05 for symbol in panels}

    monkeypatch.setattr(api, 'strategies', lambda: {'tight': tight})
    payload = get(service(), f'/api/baselines?max_weight={WIDE_CAP}')
    row = payload['strategies'][0]
    assert payload['max_weight'] == WIDE_CAP
    assert row['cap_bound'] is False
    assert row['applied_max_weight'] == pytest.approx(0.05)
    assert row['applied_max_weight'] < payload['max_weight']

  def test_it_cannot_breach_the_engine_ceiling(self, monkeypatch):
    '''Unbound does not mean unchecked.

    Args:
      monkeypatch: Fixture replacing the provider table.
    '''
    def greedy(panels):
      '''Return weights well above the requested cap.

      Args:
        panels: Visible price panels.

      Returns:
        A book four times the requested cap, which is still under 1.0 so
        the engine's clamp is what has to hold.
      '''
      return {symbol: 0.4 for symbol in panels}

    monkeypatch.setattr(api, 'strategies', lambda: {'greedy': greedy})
    row = rows_at(service(), WIDE_CAP)['greedy']
    assert row['status'] == 'ok', row.get('error')
    assert row['cap_bound'] is False
    assert row['applied_max_weight'] == pytest.approx(WIDE_CAP)

  def test_a_kwargs_provider_does_take_the_cap(self, monkeypatch):
    '''``**kwargs`` counts as accepting it, so the cap is bound.

    Args:
      monkeypatch: Fixture replacing the provider table.
    '''
    seen = []

    def loose(panels, **kwargs):
      '''Accept any keyword and record it.

      Args:
        panels: Visible price panels.
        **kwargs: Recorded by keyword.

      Returns:
        A book at whichever cap was supplied.
      '''
      seen.append(dict(kwargs))
      cap = kwargs.get('max_weight', 0.05)
      return {symbol: cap for symbol in panels}

    monkeypatch.setattr(api, 'strategies', lambda: {'loose': loose})
    row = rows_at(service(), WIDE_CAP)['loose']
    assert row['status'] == 'ok', row.get('error')
    assert row['cap_bound'] is True
    assert row['applied_max_weight'] == pytest.approx(WIDE_CAP)
    assert seen and all(kwargs == {'max_weight': WIDE_CAP}
                        for kwargs in seen)

  def test_a_provider_that_will_not_describe_itself_is_marked_unbound(
      self, monkeypatch):
    '''Refusing to introspect is not the same as refusing to run.

    :func:`stock_rl.api._accepts_cap` asks rather than assumes, and a
    callable that will not answer the question -- a C builtin behind a
    decorator, anything whose ``__signature__`` raises -- is reported as
    not accepting the cap. Guessing "surely it takes one" would be the
    original defect wearing a different hat.

    Args:
      monkeypatch: Fixture replacing the provider table.
    '''
    class Opaque:
      '''A provider that refuses to describe its own signature.'''

      @property
      def __signature__(self):
        '''Raise rather than describe, standing in for a C callable.

        Returns:
          Never returns.

        Raises:
          ValueError: Always.
        '''
        raise ValueError('this callable will not describe itself')

      def __call__(self, panels):
        '''Return one weight per symbol.

        Args:
          panels: Visible price panels.

        Returns:
          A flat book.
        '''
        return {symbol: 0.25 for symbol in panels}

    opaque = Opaque()
    monkeypatch.setattr(api, 'strategies', lambda: {'opaque': opaque})
    row = rows_at(service(), WIDE_CAP)['opaque']
    assert row['status'] == 'ok', row.get('error')
    assert row['cap_bound'] is False
    assert row['applied_max_weight'] == pytest.approx(0.25)
    # pylint: disable=protected-access
    assert api._accepts_cap(opaque) is False


class TestTheTwoEndpointsAgree:
  '''``GET /api/baselines`` and ``POST /api/backtest``, one set of numbers.

  The row claims the POST reproduces it under the same settings. That
  claim is only true if both bind the cap the same way, so it is
  asserted rather than described.
  '''

  @pytest.mark.parametrize('name', sorted(api.strategies()))
  @pytest.mark.parametrize('cap', [NARROW_CAP, WIDE_CAP])
  def test_a_row_reproduces_the_post_under_the_same_settings(self, name,
                                                              cap):
    '''Args:
      name: Provider name under test.
      cap: Requested per-symbol cap.
    '''
    svc = service()
    row = rows_at(svc, cap)[name]
    triggered = post(svc, {'strategy': name, 'max_weight': cap})
    assert row['status'] == 'ok', row.get('error')
    assert row['sharpe'] == pytest.approx(triggered['sharpe'])
    assert row['applied_max_weight'] == pytest.approx(
      triggered['applied_max_weight'])
    assert triggered['applied_max_weight'] == pytest.approx(cap)
    assert heaviest(triggered['weights']) == pytest.approx(cap)
