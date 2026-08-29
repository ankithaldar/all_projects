#!/usr/bin/env python
# -- coding: utf-8 --

'''Entrypoint: run the retail-ops MCP server over stdio.

Usage (normally spawned by the MCP client as a subprocess):
  python -m chapter01_mcp.servers.retail_main
'''


from __future__ import annotations

from chapter01_mcp.logging_util import get_mcp_logger
from chapter01_mcp.server.stdio import run_stdio_server
from chapter01_mcp.servers.ops_db import seed_if_empty
from chapter01_mcp.servers.retail_server import build_retail_server

logger = get_mcp_logger(__name__)


def main() -> None:
  '''Seed data and serve the retail MCP server on stdio.'''
  seed_if_empty()
  server = build_retail_server()
  run_stdio_server(server)


if __name__ == '__main__':
  main()
