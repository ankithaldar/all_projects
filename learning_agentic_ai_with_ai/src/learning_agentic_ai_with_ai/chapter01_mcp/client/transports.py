#!/usr/bin/env python
# -- coding: utf-8 --

'''Client-side transports for MCP (stdio subprocess + SSE over HTTP).

A transport answers one question: how do I get a JSON-RPC response line for a
JSON-RPC request line? Everything above this interface is transport-blind.
'''


from __future__ import annotations

import json
import subprocess
import threading
import time
import uuid
from abc import ABC
from queue import Empty, Queue
from typing import Any, Dict, List, Optional

import httpx

from agentic_common import paths
from chapter01_mcp.logging_util import get_mcp_logger
from chapter01_mcp.schemas import ServerDescriptor

logger = get_mcp_logger(__name__)


class TransportError(RuntimeError):
  '''Raised when a transport cannot deliver a request.'''


class ClientTransport(ABC):
  '''Abstract request/response line transport.'''

  def send(self, line: str) -> str:
    '''Send one JSON-RPC line and return the response line.

    Args:
      line: Serialized JSON-RPC request.

    Returns:
      Serialized JSON-RPC response.

    Raises:
      TransportError: On communication failure.
    '''
    raise NotImplementedError

  def send_notification(self, line: str) -> None:
    '''Send a notification line (no response expected).

    Args:
      line: Serialized JSON-RPC notification.
    '''
    raise NotImplementedError

  def close(self) -> None:
    '''Release transport resources.'''
    raise NotImplementedError


class StdioClientTransport(ClientTransport):
  '''Talks to an MCP server subprocess over stdin/stdout.'''

  def __init__(
    self,
    descriptor: ServerDescriptor,
    timeout_seconds: float = 15.0,
  ) -> None:
    '''Spawn the server subprocess.

    Args:
      descriptor: Server descriptor with command/args/cwd/env.
      timeout_seconds: Per-request timeout.

    Raises:
      TransportError: If the process cannot be spawned.
    '''
    self._timeout = timeout_seconds
    self._lock = threading.Lock()

    env = dict(descriptor.env or {})
    # Ensure the child can import our packages.
    child_pythonpath = str(paths.SRC_DIR)
    existing = env.get('PYTHONPATH', '')
    suffix = '' if not existing else ':' + existing
    env['PYTHONPATH'] = f'{child_pythonpath}{suffix}'

    try:
      self._process = subprocess.Popen(
        [descriptor.command, *descriptor.args],
        cwd=descriptor.cwd or str(paths.PROJECT_ROOT),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding='utf-8',
        env={**dict(__import__('os').environ), **env},
        bufsize=1,
      )
    except OSError as exc:
      raise TransportError(
        f'failed to spawn stdio server {descriptor.name}: {exc}'
      ) from exc

    self._stderr_buffer: List[str] = []
    self._stderr_lines: List[str] = []
    self._stderr_thread = threading.Thread(
      target=self._drain_stderr,
      daemon=True,
    )
    self._stderr_thread.start()

  def _drain_stderr(self) -> None:
    '''Consume server stderr into a bounded ring buffer for diagnostics.'''
    assert self._process.stderr is not None
    for raw in iter(self._process.stderr.readline, ''):
      text = raw.strip()
      if text:
        logger.debug('server-stderr: %s', text[-2000:])
        self._stderr_lines.append(text[-2000:])
        if len(self._stderr_lines) > 50:
          self._stderr_lines.pop(0)

  def send(self, line: str) -> str:
    '''Send a request line and read lines until the matching response.

    Notifications from the server are skipped (logged). If the process dies,
    TransportError is raised with captured stderr for debugging.

    Args:
      line: JSON-RPC request line.

    Returns:
      Response line.

    Raises:
      TransportError: On IO failure, timeout, or process crash.
    '''
    with self._lock:
      if self._process.poll() is not None:
        raise TransportError(
          f'server exited early (code={self._process.returncode}); '
          f'stderr tail: {self._recent_stderr()}'
        )

      try:
        assert self._process.stdin is not None
        assert self._process.stdout is not None
        self._process.stdin.write(line + '\n')
        self._process.stdin.flush()
      except (BrokenPipeError, ValueError) as exc:
        raise TransportError(f'stdio write failed: {exc}') from exc

      deadline = time.monotonic() + self._timeout
      request_id = self._extract_id(line)

      while time.monotonic() < deadline:
        raw = self._process.stdout.readline()
        if not raw:
          raise TransportError(
            f'server closed stdout; stderr tail: {self._recent_stderr()}'
          )
        raw = raw.strip()
        if not raw:
          continue

        try:
          payload = json.loads(raw)
        except json.JSONDecodeError:
          logger.warning('discarding non-JSON line from server')
          continue

        # Skip notifications and responses of other ids.
        if 'method' in payload and 'id' not in payload:
          continue
        if request_id is not None and payload.get('id') != request_id:
          continue
        return raw

      raise TransportError(
        f'timeout after {self._timeout}s waiting for id={request_id}'
      )

  def send_notification(self, line: str) -> None:
    '''Fire a notification into the server's stdin.

    Args:
      line: Serialized notification.
    '''
    with self._lock:
      if self._process.stdin is None or self._process.poll() is not None:
        return
      try:
        self._process.stdin.write(line + '\n')
        self._process.stdin.flush()
      except (BrokenPipeError, ValueError):
        pass

  def close(self) -> None:
    '''Terminate the subprocess (SIGTERM then SIGKILL on stubborn processes).'''
    if self._process.poll() is None:
      self._process.terminate()
      try:
        self._process.wait(timeout=3)
      except subprocess.TimeoutExpired:  # type: ignore[attr-defined]
        self._process.kill()

  @staticmethod
  def _extract_id(line: str) -> Optional[Any]:
    '''Pull the id out of a serialized request.

    Args:
      line: Serialized request.

    Returns:
      The id, or None when it is a notification.
    '''
    try:
      payload = json.loads(line)
    except json.JSONDecodeError:
      return None
    return payload.get('id')

  def _recent_stderr(self) -> str:
    '''Return the most recent stderr fragments for diagnostics.

    Returns:
      Short stderr text.
    '''
    return ' | '.join(self._stderr_lines[-8:])


class SseClientTransport(ClientTransport):
  '''Talks to an MCP server over HTTP POST + optional SSE stream.

  Supports two server dialects transparently:
  - Our teaching server: POST returns the JSON-RPC response in the body.
  - Spec-style servers (e.g. official SDK): POST returns 202 and responses
    arrive on the GET /sse event stream; we correlate by JSON-RPC id.
  '''

  def __init__(
    self,
    descriptor: ServerDescriptor,
    timeout_seconds: float = 15.0,
  ) -> None:
    '''Connect to the SSE endpoint and learn the message URL.

    Args:
      descriptor: Server descriptor with `url`.
      timeout_seconds: HTTP timeout.

    Raises:
      TransportError: If the endpoint is unreachable.
    '''
    self._timeout = timeout_seconds
    self._client = httpx.Client(timeout=timeout_seconds)
    self._base = descriptor.url.rstrip('/')
    self._messages_url: Optional[str] = None
    self._pending: Dict[str, Queue[Dict[str, Any]]] = {}
    self._pending_lock = threading.Lock()
    self._listener: Optional[threading.Thread] = None
    self._stop = threading.Event()

    self._open_sse()

  def _open_sse(self) -> None:
    '''Open the /sse stream, parse the endpoint event, start the pump.'''
    try:
      request = self._client.build_request('GET', f'{self._base}/sse')
      response = self._client.send(request, stream=True)
    except httpx.HTTPError as exc:
      raise TransportError(
        f'cannot open SSE stream at {self._base}: {exc}'
      ) from exc

    if response.status_code != 200:
      response.close()
      raise TransportError(f'SSE endpoint returned {response.status_code}')

    # Read the first event to discover the messages endpoint.
    endpoint_url: str = ''
    for line in response.iter_lines():
      line = line.strip()
      if not line.startswith('data:'):
        continue
      try:
        payload = json.loads(line[5:].strip())
      except json.JSONDecodeError:
        continue
      if payload.get('event') == 'endpoint':
        url_value = str(payload.get('url', '')).strip()
        if url_value:
          endpoint_url = (
            url_value
            if url_value.startswith('http')
            else f"{self._base}/{url_value.lstrip('/')}"
          )
          if '?' not in url_value:
            endpoint_url += f'?session_id={uuid.uuid4().hex}'
          break

    if not endpoint_url:
      response.close()
      raise TransportError('SSE server did not announce a messages endpoint')

    self._messages_url = endpoint_url
    self._response = response
    self._listener = threading.Thread(target=self._pump_events, daemon=True)
    self._listener.start()
    logger.info(
      'sse transport connected',
      extra={'extra_fields': {'url': endpoint_url}},
    )

  def _pump_events(self) -> None:
    '''Background pump: read SSE lines and dispatch to waiters by id.'''
    try:
      for line in self._response.iter_lines():
        if self._stop.is_set():
          return
        if not line or not line.startswith('data:'):
          continue
        body = line[5:].strip()
        try:
          payload = json.loads(body)
        except json.JSONDecodeError:
          continue

        data = payload.get('data')
        if isinstance(data, dict) and 'id' in data:
          request_id = data['id']
          with self._pending_lock:
            q = self._pending.pop(str(request_id), None)
          if q is not None:
            q.put(data)
    except Exception:  # pylint: disable=broad-exception-caught
      # Pump teardown races close(); only log when unexpected.
      if not self._stop.is_set():
        logger.debug('sse pump ended')

  def send(self, line: str) -> str:
    '''POST one JSON-RPC line; return the response line.

    Prefers the synchronous HTTP response body (our server). Falls back to
    waiting on the SSE stream (official SDK servers).

    Args:
      line: Serialized JSON-RPC request.

    Returns:
      Serialized response line.

    Raises:
      TransportError: On HTTP or protocol failure.
    '''
    if self._messages_url is None:
      raise TransportError('SSE transport not initialized')

    request_id = self._extract_id(line)
    waiter: Optional[Queue[Dict[str, Any]]] = None
    if request_id is not None:
      waiter = Queue()
      with self._pending_lock:
        self._pending[str(request_id)] = waiter

    try:
      response = self._client.post(
        self._messages_url,
        content=line.encode('utf-8'),
        headers={'Content-Type': 'application/json'},
      )
    except httpx.HTTPError as exc:
      raise TransportError(f'POST failed: {exc}') from exc

    if response.status_code == 200:
      try:
        payload = response.json()
      except json.JSONDecodeError as exc:
        raise TransportError(f'bad POST response: {exc}') from exc
      if isinstance(payload, dict) and 'response' in payload:
        return json.dumps(payload['response'], separators=(',', ':'))

    # 202 or no inline response: wait on SSE.
    if waiter is None:
      raise TransportError('no synchronous response and no correlation id')

    deadline = time.monotonic() + self._timeout
    while time.monotonic() < deadline:
      try:
        payload = waiter.get(timeout=0.5)
        return json.dumps(payload, separators=(',', ':'), default=str)
      except Empty:
        continue

    with self._pending_lock:
      self._pending.pop(str(request_id), None)
    raise TransportError(f'timeout waiting for SSE response id={request_id}')

  def send_notification(self, line: str) -> None:
    '''POST a notification (server replies 202; nothing to await).

    Args:
      line: Serialized notification.
    '''
    try:
      self._client.post(
        self._messages_url or f'{self._base}/messages',
        content=line,
        headers={'Content-Type': 'application/json'},
      )
    except httpx.HTTPError as exc:
      logger.warning(
        'notification POST failed',
        extra={'extra_fields': {'error': str(exc)}},
      )

  def close(self) -> None:
    '''Close the SSE stream and HTTP client.'''
    self._stop.set()
    try:
      self._client.close()
    except Exception:  # pylint: disable=broad-exception-caught
      pass

  @staticmethod
  def _extract_id(line: str) -> Optional[Any]:
    '''Extract the id field from a serialized request.

    Args:
      line: Serialized request.

    Returns:
      The id or None.
    '''
    try:
      payload = json.loads(line)
      return payload.get('id')
    except json.JSONDecodeError:
      return None


def transport_from_descriptor(
  descriptor: ServerDescriptor,
  timeout_seconds: float = 15.0,
) -> ClientTransport:
  '''Build the right transport for a descriptor.

  Args:
    descriptor: Server descriptor.
    timeout_seconds: Per-request timeout.

    Returns:
      A connected ClientTransport.

    Raises:
      TransportError: On unsupported transport kinds.
  '''
  if descriptor.transport == 'stdio':
    return StdioClientTransport(descriptor, timeout_seconds=timeout_seconds)

  if descriptor.transport == 'sse':
    return SseClientTransport(descriptor, timeout_seconds=timeout_seconds)

  raise TransportError(f'unsupported transport: {descriptor.transport}')
