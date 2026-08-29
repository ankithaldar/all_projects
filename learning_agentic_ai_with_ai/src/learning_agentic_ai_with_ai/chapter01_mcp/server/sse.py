#!/usr/bin/env python
# -- coding: utf-8 --

'''SSE transport for the hand-rolled MCP server (stdlib only).

SSE (Server-Sent Events) carries responses over HTTP; requests ride HTTP POST
bodies. This transport matters when the MCP server runs in another container
or another machine - no subprocess possible.

Wire contract implemented here (kept deliberately simple):
  POST /mcp          body={"jsonrpc":"2.0","id":1,"method":...}  (sync request)
  GET  /mcp/events?session_id=...   -> SSE stream: one JSON line per event:
        {"event":"message","data":{...json-rpc response or server notification...}}
  The client generates a session_id; all responses/notifications for that
  session are streamed there. In this teaching implementation responses are
  correlated by JSON-RPC id (there is one HTTP POST per request; the events
  stream also mirrors them so async server->client notifications have a door).

Note on the official spec: modern MCP defines "Streamable HTTP" and older
revisions used HTTP+SSE with an /sse GET + POST /messages endpoint and an
endpoint event. For learning we implement the classic shape (GET /sse,
POST /messages) which the official TypeScript SDK also supports - so our
client and server stay interop-testable with it.
'''


from __future__ import annotations

import json
import queue
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional, Type

from chapter01_mcp.logging_util import get_mcp_logger
from chapter01_mcp.server_core import MCPServerCore

logger = get_mcp_logger(__name__)


class SseSessionHub:
  '''Fan-out hub: maps session ids to subscriber queues.'''

  def __init__(self) -> None:
    '''Initialize an empty hub.'''
    self._lock = threading.Lock()
    self._queues: Dict[str, 'queue.Queue[str]'] = {}

  def register(self, session_id: str) -> 'queue.Queue[str]':
    '''Register a listener queue for a session.

    Args:
      session_id: Client session id.

    Returns:
      A queue the HTTP handler will drain.
    '''
    q: 'queue.Queue[str]' = queue.Queue()
    with self._lock:
      self._queues[session_id] = q
    return q

  def unregister(self, session_id: str) -> None:
    '''Remove a listener.

    Args:
      session_id: Client session id.
    '''
    with self._lock:
      self._queues.pop(session_id, None)

  def publish(self, session_id: str, payload: str) -> bool:
    '''Publish one SSE data line to a session queue.

    Args:
      session_id: Target session.
      payload: JSON string (single event).

    Returns:
      True if delivered to an existing session.
    '''
    with self._lock:
      q = self._queues.get(session_id)
    if q is None:
      return False
    q.put(payload)
    return True


SESSIONS = SseSessionHub()


def make_http_handler(server: MCPServerCore, sessions: SseSessionHub) -> Type:
  '''Create an HTTP handler class bound to a server core.

  Args:
    server: The MCP server core.
    sessions: Session hub for SSE fan-out.

  Returns:
    A BaseHTTPRequestHandler subclass.
  '''

  class Handler(BaseHTTPRequestHandler):
    '''HTTP handler implementing GET /sse and POST /messages.'''

    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: D102
      logger.debug('http %s', fmt % args)

    # ------------------------------------------------------------------
    def do_GET(self) -> None:  # noqa: N802
      '''GET /sse -> open SSE stream (long-lived).'''
      if not self.path.startswith('/sse'):
        self._json_response(404, {'error': 'not found'})
        return

      session_id = uuid.uuid4().hex
      self.send_response(200)
      self.send_header('Content-Type', 'text/event-stream')
      self.send_header('Cache-Control', 'no-cache')
      self.send_header('Connection', 'keep-alive')
      self.end_headers()

      # Spec: first event tells the client where to POST messages.
      self._sse_event({'event': 'endpoint', 'url': f'/messages?session_id={session_id}'})

      q = sessions.register(session_id)
      try:
        while True:
          try:
            payload = q.get(timeout=1.0)
          except queue.Empty:
            self.wfile.write(b': keepalive\n\n')
            self.wfile.flush()
            continue
          self._sse_event({'event': 'message', 'data': payload})
      except (BrokenPipeError, ConnectionResetError):
        pass
      finally:
        sessions.unregister(session_id)

    def do_POST(self) -> None:  # noqa: N802
      '''POST /messages?session_id=... -> deliver a JSON-RPC line.'''
      if not self.path.startswith('/messages'):
        self._json_response(404, {'error': 'not found'})
        return

      try:
        length = int(self.headers.get('Content-Length', '0'))
        body = self.rfile.read(length).decode('utf-8')
      except Exception as exc:
        self._json_response(400, {'error': f'bad body: {exc}'})
        return

      session = self._session_from_path()
      response_line = server.handle_line(body)

      if response_line is None:
        # Notification: no direct response; nothing on SSE either.
        self._json_response(202, {'accepted': True})
        return

      delivered = False
      if session:
        delivered = sessions.publish(session, response_line)

      self._json_response(200, {'delivered': delivered, 'response': json.loads(response_line)})

    # ------------------------------------------------------------------
    def _sse_response_line(self, payload: str) -> None:
      '''Push a tools/response event onto the session stream.

      Args:
        payload: JSON-RPC response line.
      '''
      # Not used by HTTP handler directly; kept for symmetry.

    def _sse_event(self, payload: Dict[str, Any]) -> None:
      '''Write one SSE event from a dict payload.

      Args:
        payload: Event payload; serialized as `data: <json>`.
      '''
      self.wfile.write(f'data: {json.dumps(payload, default=str)}\n\n'.encode('utf-8'))
      self.wfile.flush()

    def _json_response(self, status: int, payload: Dict[str, Any]) -> None:
      '''Write a JSON HTTP response.

      Args:
        status: HTTP status code.
        payload: JSON body.
      '''
      body = json.dumps(payload, default=str).encode('utf-8')
      self.send_response(status)
      self.send_header('Content-Type', 'application/json')
      self.send_header('Content-Length', str(len(body)))
      self.end_headers()
      self.wfile.write(body)

    def _session_from_path(self) -> Optional[str]:
      '''Extract session_id query parameter.

      Returns:
        Session id or None.
      '''
      if '?' not in self.path:
        return None
      query = self.path.split('?', 1)[1]
      for part in query.split('&'):
        if part.startswith('session_id='):
          return part.split('=', 1)[1]
      return None

  return Handler


class SseServerRunner:
  '''Runs the MCP server over HTTP/SSE in-process (threaded) or standalone.'''

  def __init__(self, server: MCPServerCore, host: str = '127.0.0.1', port: int = 8765) -> None:
    '''Prepare (but do not start) the HTTP server.

    Args:
      server: MCP server core.
      host: Bind address.
      port: TCP port (0 = OS-assigned free port).
    '''
    self.sessions = SseSessionHub()
    handler = make_http_handler(server, self.sessions)
    self.httpd = ThreadingHTTPServer((host, port), handler)
    self.port = self.httpd.server_address[1]

  def serve_in_background(self) -> None:
    '''Start serving in a daemon thread.'''
    thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
    thread.start()
    logger.info(
      'sse server listening',
      extra={'extra_fields': {'port': self.port}},
    )
