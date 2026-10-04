#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Adversarial review of ``stock_rl.api`` and its dashboard.

Every test in this file was written to FAIL against the code as it
stands, so each one is a claim about behaviour rather than a description
of it. Where a bug is real the test is left failing on purpose: the
reviewer does not fix what it finds.

The tests reach the module rather than restating its arithmetic. Nothing
here recomputes a Sharpe, a momentum score or a drawdown by hand, because
a test that reproduces the suspect formula inline proves only that the
formula is what the reader thought it was.

The findings, in the order the file probes them:

* The dashboard document references two sibling assets and the API serves
  neither, so the page a user is told to open renders nothing.
* An empty book reports a Sharpe of 0.0 and a growth multiple of 1.0,
  which are the exact numbers ``PortfolioState`` says it refuses to
  report, and they are indistinguishable from a real zero-return run.
* A weight provider that raises ``ValueError`` is reported to the caller
  as a 422 client error, contradicting ``ProviderFailed``.
* ``dispatch`` has no catch-all, so an unexpected exception closes the
  connection with no HTTP response at all.
* A malformed ``Content-Length`` is answered with a 413 that claims the
  body was too large.
* ``web.asset_text`` reads any file the process can read.
'''

from __future__ import annotations

import json
import os
import re
import threading
from datetime import datetime, timedelta
from http.client import HTTPConnection, RemoteDisconnected

import pytest

from stock_rl import api, web
from stock_rl.bars import Bar

#: Matches an href/src attribute value in the dashboard document.
REFERENCE = re.compile(r'(?:href|src)="([^"]*)"')

#: Matches the confidence the dashboard hardcodes into its VaR row label.
VA_LABEL = re.compile(r"'VaR (\d+)%[^']*'")


def service(**kwargs):
  '''Return a service with frozen clocks, as the suite convention is.

  Args:
    **kwargs: Passed through to :class:`~stock_rl.api.ApiService`.

  Returns:
    The service under test.
  '''
  kwargs.setdefault('clock', lambda: 0.0)
  return api.ApiService(**kwargs)


def local_references(document):
  '''Return the same-origin subresources a document pulls in.

  Args:
    document: An HTML document.

  Returns:
    Every href/src value that names a sibling file rather than an
    absolute origin, a fragment or a data URI.
  '''
  found = []
  for raw in REFERENCE.findall(document):
    if not raw or raw.startswith(('#', 'data:', '//')):
      continue
    if '://' in raw:
      continue
    found.append(raw)
  return found


def get(svc, path):
  '''Return the decoded body of a GET, asserting it succeeded.

  Args:
    svc: Service to query.
    path: Route to request.

  Returns:
    The parsed payload.
  '''
  response = api.dispatch(svc, 'GET', path)
  assert response.status == 200, response.body
  return json.loads(response.body)


#: Bars long enough that ``history`` plus ``rebalance_days`` leaves room,
#: so a provider is actually reached rather than refused up front.
LONG_PANEL = 140


def bar(offset, close):
  '''Return one flat bar.

  Args:
    offset: Days after the epoch of this file's bars.
    close: Closing price.

  Returns:
    A bar whose open equals its close.
  '''
  return Bar(timestamp=datetime(2026, 1, 1) + timedelta(days=offset),
             open=close, high=close, low=close, close=close, volume=1.0)


def rising_panels():
  '''Return two aligned panels long enough to trade.

  Returns:
    Mapping of symbol to bars, both rising so a provider is reached.
  '''
  return {
    'AAA': [bar(index, 100.0 + index) for index in range(LONG_PANEL)],
    'BBB': [bar(index, 200.0 + index) for index in range(LONG_PANEL)],
  }


def exploding(exception):
  '''Return a weight provider that raises ``exception``.

  Args:
    exception: The exception instance every call raises.

  Returns:
    A provider callable.
  '''
  def provider(panels):
    '''Raise the configured exception.

    Args:
      panels: Visible price panels.

    Raises:
      Exception: Always, the one this helper was built with.
    '''
    del panels
    raise exception
  return provider


def live_server(svc):
  '''Start a server on an ephemeral loopback port.

  Args:
    svc: Service to serve.

  Returns:
    Tuple of (server, host, port). The caller closes the server.
  '''
  server = api.build_server(svc, '127.0.0.1', 0)
  host, port = api.bound_address(server)
  thread = threading.Thread(target=server.serve_forever, daemon=True)
  thread.start()
  return server, host, port


class TestTheDashboardIsActuallyServed:
  '''The document names two assets. The API serves one document.'''

  def test_the_document_references_at_least_one_subresource(self):
    # If this ever fails the next tests pass vacuously, so the floor is
    # asserted first rather than assumed.
    assert local_references(web.index_html())

  def test_every_subresource_the_document_references_is_served(self):
    svc = service()
    for reference in local_references(web.index_html()):
      path = reference if reference.startswith('/') else '/' + reference
      response = api.dispatch(svc, 'GET', path)
      assert response.status == 200, (
        f'the document pulls in {reference!r} and the API answers '
        f'{response.status} for it: {response.body!r}')

  def test_a_served_subresource_carries_the_asset_bytes(self):
    svc = service()
    for reference in local_references(web.index_html()):
      path = reference if reference.startswith('/') else '/' + reference
      response = api.dispatch(svc, 'GET', path)
      expected = web.asset_text(os.path.basename(reference))
      assert response.body.decode() == expected, path

  def test_every_public_dashboard_asset_is_reachable_over_a_route(self):
    svc = service()
    served = []
    for route in api.routes:
      if api.dispatch(svc, 'GET', route).status != 200:
        continue
      served.append(api.dispatch(svc, 'GET', route).body.decode())
    for name in ('index_html', 'stylesheet', 'script'):
      asset = getattr(web, name)()
      assert asset in served, (
        f'web.{name}() exists to be served and no route serves it; the '
        f'API advertises a dashboard it cannot deliver')

  def test_the_subresources_are_reachable_over_a_real_socket(self):
    server, host, port = live_server(service())
    try:
      for reference in local_references(web.index_html()):
        connection = HTTPConnection(host, port, timeout=5)
        try:
          connection.request('GET', '/' + reference)
          response = connection.getresponse()
          body = response.read()
          assert response.status == 200, (reference, response.status, body)
          assert body.decode() == web.asset_text(
            os.path.basename(reference)), reference
        finally:
          connection.close()
    finally:
      server.shutdown()
      server.server_close()


class TestAnEmptyBookReportsAbsenceNotZeroes:
  '''``PortfolioState`` promises absence, ``metrics`` supplies zeroes.'''

  def test_no_performance_figure_is_reported_for_a_book_that_never_ran(self):
    payload = get(service(), '/api/equity')
    assert payload['points'] == 0
    assert payload['source'] == 'none'
    assert payload['equity'] == []
    for key in ('sharpe', 'max_drawdown', 'total_return_multiple'):
      assert payload[key] is None, (
        f'{key}={payload[key]!r} is an assertion that performance was '
        f'exactly zero, on a book that has never run')

  def test_an_empty_book_is_distinguishable_from_a_flat_backtest(self):
    # "No performance yet" and "performance of exactly zero" are
    # different facts. The Metrics table must be able to say which.
    empty = get(service(), '/api/equity')
    flat = get(
      service(state=api.PortfolioState(
        equity=[1.0] * 40, returns=[0.0] * 39, value=10_000_000.0)),
      '/api/equity')
    assert empty['source'] != flat['source']
    for key in ('sharpe', 'max_drawdown', 'total_return_multiple'):
      assert empty[key] is None or empty[key] != flat[key], (
        f'{key}={empty[key]!r} is printed both for "nothing has run" and '
        f'for a backtest that returned exactly nothing, so the reader '
        f'cannot tell them apart')

  def test_the_var_block_does_not_claim_zero_risk_without_history(self):
    payload = get(service(), '/api/risk')
    assert payload['var']['observations'] == 0
    assert payload['var']['one_period_fraction'] is None
    assert payload['var']['rupees'] is None


class TestProviderFailureIsNotAClientError:
  '''``ProviderFailed`` promises a 500 for anything a provider raises.'''

  def test_a_provider_raising_value_error_is_a_500(self, monkeypatch):
    monkeypatch.setattr(
      api, 'strategies',
      lambda: {'exploding': exploding(ValueError('off-by-one'))})
    response = api.dispatch(
      api.ApiService(panels=rising_panels()), 'POST', '/api/backtest',
      b'{"strategy": "exploding"}')
    assert response.status == 500, (
      f'a crash inside another module was reported as '
      f'{response.status}, which tells the caller to change a request that '
      f'was never the problem')
    assert 'ValueError' in json.loads(response.body)['error']
    assert 'Traceback' not in response.body.decode()

  def test_a_failed_backtest_leaves_the_reported_book_untouched(self):
    '''A crash mid-run must not half-write the book.'''

    def runner(request):
      '''Fail after the request is fully validated.

      Args:
        request: The validated trigger.

      Raises:
        TypeError: Always.
      '''
      del request
      raise TypeError('a bug in the runner')

    svc = service(backtest_runner=runner)
    with pytest.raises(TypeError):
      api.dispatch(svc, 'POST', '/api/backtest', b'{}')
    assert get(svc, '/api/equity')['source'] == 'none'
    assert get(svc, '/api/positions')['positions'] == []

  def test_a_provider_raising_value_error_is_marked_as_an_error(self,
                                                               monkeypatch):
    '''A crashing control arm must not be filed as untried.'''
    monkeypatch.setattr(
      api, 'strategies',
      lambda: {'exploding': exploding(ValueError('off-by-one'))})
    payload = get(api.ApiService(panels=rising_panels()), '/api/baselines')
    row = payload['strategies'][0]
    seen = row['status']
    assert seen == 'error', (
      f'status={seen!r} tells the operator the control arm was skipped '
      f'when in fact it crashed')
    assert 'ValueError' in row['error']


class TestDispatchAlwaysAnswers:
  '''``dispatch`` promises nothing reaches the client as a traceback.'''

  def test_an_unexpected_exception_raises_rather_than_returning_a_body(self):
    def runner(request):
      '''Fail in a way the four handled types do not cover.

      Args:
        request: The validated trigger.

      Raises:
        TypeError: Always.
      '''
      del request
      raise TypeError('a bug in the runner')

    svc = service(backtest_runner=runner)
    response = api.dispatch(svc, 'POST', '/api/backtest', b'{}')
    assert response.status == 500, response.body
    assert 'Traceback' not in response.body.decode()

  def test_an_unexpected_exception_does_not_reset_the_connection(self):
    def runner(request):
      '''Fail in a way the four handled types do not cover.

      Args:
        request: The validated trigger.

      Raises:
        TypeError: Always.
      '''
      del request
      raise TypeError('a bug in the runner')

    server, host, port = live_server(service(backtest_runner=runner))
    try:
      connection = HTTPConnection(host, port, timeout=5)
      try:
        connection.request('POST', '/api/backtest', body=b'{}')
        response = connection.getresponse()
        assert response.status == 500
        response.read()
      except RemoteDisconnected:
        pytest.fail(
          'the server dropped the connection without a response, so the '
          'caller sees a transport error rather than a diagnosis')
      finally:
        connection.close()
      # And the server is still alive for the next caller.
      connection = HTTPConnection(host, port, timeout=5)
      try:
        connection.request('GET', '/api/health')
        assert connection.getresponse().status == 200
      finally:
        connection.close()
    finally:
      server.shutdown()
      server.server_close()

  def test_health_cannot_report_a_halt_that_risk_refuses_to_repeat(self):
    # app.js hides the "KILL SWITCH TRIPPED" banner whenever /api/risk
    # fails, and /api/health is the only other place the latch state is
    # printed. So if health says TRIPPED while risk cannot answer, the
    # dashboard shows a tripped switch and no halt banner at all.
    class Hostile:
      '''A latched switch whose supplementary limits raise.'''

      algo_id = 'HOSTILE1'

      @property
      def tripped(self):
        '''Report the latch.

        Returns:
          True, always.
        '''
        return True

      @property
      def trading_enabled(self):
        '''Report trading as halted.

        Returns:
          False, always.
        '''
        return False

      @property
      def thresholds(self):
        '''Refuse to answer, with a type nothing catches.

        Returns:
          Never returns.

        Raises:
          TypeError: Always.
        '''
        raise TypeError('limits unreadable')

      def breached(self, index_change, drawdown, fno, ack):
        '''Report the drawdown breach.

        Args:
          index_change: Session change.
          drawdown: Current drawdown.
          fno: F&O ban flag.
          ack: Acknowledgement latency.

        Returns:
          The breached condition.
        '''
        del index_change, drawdown, fno, ack
        return ('drawdown 20.00% against a 15.00% limit',)

    svc = service(kill_switch=Hostile())
    health = get(svc, '/api/health')
    latch = health['kill_switch']
    risk = api.dispatch(svc, 'GET', '/api/risk')
    assert latch == 'TRIPPED'
    assert risk.status == 200, (
      f'/api/health printed {latch!r} and /api/risk then failed, which is '
      f'the combination that leaves the dashboard with a latched switch '
      f'and no halt banner')

  def test_an_unreadable_switch_of_an_unexpected_type_is_reported(self):
    class Hostile:
      '''A switch whose every read raises something unhandled.'''

      algo_id = 'HOSTILE1'

      @property
      def tripped(self):
        '''Refuse to answer.

        Returns:
          Never returns.

        Raises:
          OSError: Always.
        '''
        raise OSError('state file is on a dead mount')

      trading_enabled = False
      thresholds = None

      def breached(self, index_change, drawdown, fno, ack):
        '''Refuse to answer.

        Args:
          index_change: Session change.
          drawdown: Current drawdown.
          fno: F&O ban flag.
          ack: Acknowledgement latency.

        Returns:
          Never returns.

        Raises:
          OSError: Always.
        '''
        del index_change, drawdown, fno, ack
        raise OSError('state file is on a dead mount')

    payload = get(service(kill_switch=Hostile()), '/api/risk')
    assert payload['kill_switch']['status'] == 'unreadable'
    assert payload['kill_switch']['tripped'] is None
    assert 'dead mount' in payload['kill_switch']['error']


class TestFramingTellsTheTruth:
  '''A refusal has to name the right fault.'''

  def test_a_malformed_length_header_is_not_reported_as_too_large(self):
    server, host, port = live_server(service())
    try:
      connection = HTTPConnection(host, port, timeout=5)
      try:
        connection.putrequest('POST', '/api/backtest')
        connection.putheader('Content-Length', 'not-a-number')
        connection.endheaders()
        response = connection.getresponse()
        body = response.read().decode()
        assert response.status == 400, (response.status, body)
        assert 'exceeds' not in body, (
          'a header that would not parse was answered with a claim that '
          'the body was too large')
      finally:
        connection.close()
    finally:
      server.shutdown()
      server.server_close()

  def test_an_oversized_body_is_answered_before_the_socket_is_closed(self):
    # max_drain_bytes exists so the peer "reads the refusal rather than a
    # reset". close_connection is set without a Connection: close header,
    # so a client that reuses the connection gets a reset anyway.
    server, host, port = live_server(service())
    try:
      connection = HTTPConnection(host, port, timeout=5)
      try:
        body = b'{"a":1}' + b' ' * (api.max_body_bytes + 8) + b'}'
        connection.request('POST', '/api/backtest', body=body)
        response = connection.getresponse()
        assert response.status == 413, response.status
        assert 'exceeds' in json.loads(response.read())['error']
        # The drain worked, so the connection was left in a usable state
        # by the server's own accounting. Reusing it must not reset.
        connection.request('GET', '/api/health')
        second = connection.getresponse()
        assert second.status == 200, (
          f'reuse after a drained 413 answered {second.status}')
        second.read()
      finally:
        connection.close()
    finally:
      server.shutdown()
      server.server_close()

  def test_a_chunked_body_is_refused_for_the_right_reason(self):
    # RFC 9112 6.3.1: a chunked POST body must be decoded or the framing
    # bytes stay in the socket. _read_body reads only Content-Length, so
    # the chunk preamble is still queued when the next request is read.
    server, host, port = live_server(service())
    try:
      connection = HTTPConnection(host, port, timeout=5)
      try:
        connection.putrequest('POST', '/api/backtest')
        connection.putheader('Transfer-Encoding', 'chunked')
        connection.endheaders()
        connection.send(b'1f\r\n{"capital":100000}\r\n0\r\n\r\n')
        response = connection.getresponse()
        status = response.status
        body = response.read().decode()
        assert status == 400, (status, body)
        assert 'chunked' in body.lower(), (
          'an unread chunked body was reported as an empty body, which '
          'names the wrong fault')
      finally:
        connection.close()
    finally:
      server.shutdown()
      server.server_close()

  def test_a_chunked_body_does_not_desynchronise_the_connection(self):
    server, host, port = live_server(service())
    try:
      connection = HTTPConnection(host, port, timeout=5)
      try:
        connection.putrequest('POST', '/api/backtest')
        connection.putheader('Transfer-Encoding', 'chunked')
        connection.endheaders()
        connection.send(b'1f\r\n{"capital":100000}\r\n0\r\n\r\n')
        connection.getresponse().read()
        # The framing bytes of the undecoded chunk must have been
        # discarded, not left to be parsed as the next request line.
        connection.request('GET', '/api/health')
        second = connection.getresponse()
        payload = second.read()
        assert second.status == 200, (
          'the next request on the connection was answered '
          f'{second.status}: {payload[:120]!r}, for a reason that had '
          'nothing to do with it')
      finally:
        connection.close()
    finally:
      server.shutdown()
      server.server_close()


class TestAssetLoaderCannotEscapeThePackage:
  '''``asset_text`` is public API and takes a caller-supplied name.'''

  @pytest.mark.parametrize('name', [
    '../api.py',
    './../web/__init__.py',
    '/etc/passwd',
    os.path.join(os.sep, 'etc', 'passwd'),
  ])
  def test_a_traversing_name_reads_nothing(self, name):
    with pytest.raises((FileNotFoundError, ValueError)):
      web.asset_text(name)

  def test_the_real_assets_still_load(self):
    assert web.stylesheet().strip()
    assert web.script().strip()
    assert web.index_html().startswith('<!DOCTYPE html>')


class TestLabelsTrackTheNumbers:
  '''The dashboard hardcodes values the API computes.'''

  def test_the_var_label_matches_the_reported_confidence(self):
    reported = get(service(), '/api/risk')['var']['confidence']
    assert reported == api.var_confidence
    labels = VA_LABEL.findall(web.script())
    assert labels, 'the VaR row labels were not found in app.js'
    for hardcoded in labels:
      assert int(hardcoded) == round(api.var_confidence * 100), (
        f'the dashboard prints "VaR {hardcoded}%" while the API reports '
        f'confidence {reported}')

  def test_every_growth_multiple_is_a_multiple_not_a_fraction(self):
    # "Growth multiple" is only that word if it starts at 1.0, so a flat
    # book prints 1.0000 and a book that doubled prints 2.0000.
    state = api.PortfolioState(equity=[1.0, 2.0], returns=[1.0],
                               value=20_000_000.0, capital=10_000_000.0)
    payload = get(service(state=state), '/api/equity')
    assert payload['total_return_multiple'] == pytest.approx(2.0)
    assert payload['final_value'] == pytest.approx(20_000_000.0)

  def test_the_document_carries_the_exact_disclaimer_the_api_sends(self):
    document = ' '.join(web.index_html().split())
    assert api.not_advice in document
    for route in ('/api/health', '/api/signals', '/api/equity',
                  '/api/positions', '/api/baselines', '/api/risk'):
      assert get(service(), route)['not_advice'] == api.not_advice
