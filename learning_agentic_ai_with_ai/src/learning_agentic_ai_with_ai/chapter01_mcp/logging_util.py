#!/usr/bin/env python
# -- coding: utf-8 --

'''Logging helper for Chapter 1 modules.

CRITICAL RULE: MCP servers that speak stdio MUST NOT write anything except
protocol lines to stdout. Human/structured logs go to stderr only.
'''


from __future__ import annotations

from agentic_common.logging import get_logger


def get_mcp_logger(name: str):
  '''Return a JSON-configured logger (stderr only).

  Args:
    name: Logger name.

  Returns:
    Configured logger.
  '''
  return get_logger(name)
