#!/usr/bin/env python
# -- coding: utf-8 --

'''stdio transport for the hand-rolled MCP server.

The server is a *plain subprocess*. Its stdout is a pipe carrying newline-
delimited JSON-RPC; its stdin carries requests from the client. That is the
entire "stdio transport": not magic, just disciplined I/O.

      ┌─────────────┐   JSON-RPC lines    ┌──────────────────┐
      │ MCP Client  │ ───────────────────▶│ MCP Server (sub) │
      │ (our agent) │ ◀───────────────────│ this file        │
      └─────────────┘   stdin/stdout      └──────────────────┘
                          stderr = logs
'''


from __future__ import annotations

from chapter01_mcp.logging_util import get_mcp_logger
from chapter01_mcp.server_core import MCPServerCore

logger = get_mcp_logger(__name__)


class StdioServerTransport:
  '''Runs an MCPServerCore over stdin/stdout of the current process.

  Loop: read one line -> dispatch -> write one response line.
  Malformed input lines are answered with PARSE_ERROR (id=null) and never
  crash the loop, so the server survives hostile noise.
  '''

  def __init__(self, server: MCPServerCore) -> None:
    '''Bind the transport to a server core.

    Args:
      server: The transport-agnostic server core.
    '''
    self._server = server

  def serve_forever(self) -> None:
    '''Serve requests until stdin closes (EOF) or a fatal error occurs.'''
    logger.info('stdio server start', extra={'extra_fields': {'server': self._server.name}})
    try:
      while True:
        line = input('')
        if not line.strip():
          continue

        response = self._server.handle_line(line)
        if response is not None:
          print(response, flush=True)
    except EOFError:
      logger.info('stdio closed; server exiting')
    except KeyboardInterrupt:  # pragma: no cover - operator interrupt
      logger.info('interrupted; server exiting')


def run_stdio_server(server: MCPServerCore) -> None:
  '''Convenience wrapper: build a transport and serve forever.

  Args:
    server: The server core to expose.
  '''
  StdioServerTransport(server).serve_forever()
