#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''The response, the routing table, and the one function that connects
them.

:class:`Response` is a value, not a socket write, so the whole API is
testable without binding a port, spawning a thread or waiting on I/O;
:class:`ApiHandler` adds framing and nothing else. :func:`dispatch` is
therefore a pure function of the service and the request, and its status
codes are the API's contract:

  404 unknown route, 405 wrong method on a known route, 400 malformed or
  out-of-range input, 422 a valid request that cannot run, 500 a provider
  or a missing dashboard asset.

Nothing else reaches the client. The terminal ``except`` clause exists so
that a ``TypeError`` in any branch above cannot propagate into the
handler and close the socket with no response at all: the caller would see
a transport error instead of a diagnosis, and a socket error is not
something a caller can act on. The repr identifies the fault, the
traceback stays in the server's log, and a stack trace in an HTTP body is
a leak rather than a diagnosis.

Split out of the former single-module ``stock_rl.api`` without change,
routing table and clause order included.
'''

from __future__ import annotations

import json
import logging
import math
from collections.abc import Callable
from dataclasses import dataclass
from http import HTTPStatus
from typing import Any

from stock_rl.api.constants import not_advice
from stock_rl.api.errors import BadRequest, ProviderFailed, Refused
from stock_rl.api.service import ApiService
from stock_rl.api.validate import _control_of, _parse_request
from stock_rl.web import index_html, script, stylesheet

_log = logging.getLogger('stock_rl.api')


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
