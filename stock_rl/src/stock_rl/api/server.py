#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''The socket, and the entry points that build it.

Everything that touches a port is here: the handler that frames HTTP,
the two server classes, and the functions that bind and serve. The
socket policy is enforced rather than documented. :func:`is_loopback`
gates :func:`build_server`, which raises rather than warns, because this
process serves positions, signals and audit state with no authentication,
and making that impossible to do by accident is the whole point. The
supported way to expose the service is a TLS-terminating reverse proxy
with a client certificate, which keeps the identity decision where it
belongs; the full argument is in the package docstring at
:mod:`stock_rl.api`.

The same policy appears twice more, in the two places a browser could
otherwise walk around it: :func:`admitted_origin` admits loopback and the
null origin and nothing else, and ``_refuse_foreign_host`` answers 421
for a ``Host`` header naming another origin, which is what keeps a
loopback service from being a DNS-rebinding target.

Split out of the former single-module ``stock_rl.api`` without change.
The two refusal messages and the ``--host``/``--port`` help text are
part of that argument and were left exactly as they were.
'''

from __future__ import annotations

import argparse
import ipaddress
import logging
import socket
from collections.abc import Sequence
from functools import partial
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from stock_rl import __version__
from stock_rl.api.constants import (
  default_host,
  default_port,
  max_body_bytes,
  max_drain_bytes,
)
from stock_rl.api.response import Response, _error, dispatch
from stock_rl.api.service import ApiService
from stock_rl.bars import Bar, load_csv

_log = logging.getLogger('stock_rl.api')


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
  # The graph reads its sentiment readings from beside the price CSVs, so
  # it needs the directory the service was pointed at. Recorded on the
  # service rather than in module state, because a module global would make
  # two services in one process share one data directory.
  service.directory = arguments.data_dir
  serve(service, arguments.host, arguments.port)
  return 0
