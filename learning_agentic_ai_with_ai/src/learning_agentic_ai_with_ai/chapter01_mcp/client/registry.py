#!/usr/bin/env python
# -- coding: utf-8 --

'''Server catalog + dynamic tool discovery.

Problem: as MCP servers multiply, dumping every tool from every server into
the LLM prompt wastes tokens, confuses the model, and slows calls.

Solution implemented here (industry pattern "summary layer + hint filter"):
1. Each ServerDescriptor carries a one-line `summary` + keyword `hints`.
2. `pick_servers(task)` scores servers against the task text via hints.
3. Only the selected servers are connected; only their tools reach the LLM.
4. `pick_tools(task, tools)` filters further within a server by keyword match
   on tool name/description when the tool count is large.
'''


from __future__ import annotations

import re
import sys
from typing import Dict, List, Optional, Tuple

from chapter01_mcp.logging_util import get_mcp_logger
from chapter01_mcp.schemas import ServerDescriptor, ToolDescriptor

logger = get_mcp_logger(__name__)


def default_catalog() -> List[ServerDescriptor]:
  '''Build the built-in catalog of course MCP servers.

  stdio servers are spawned with the *current* interpreter so they always run
  inside the same venv as the agent process.

  Returns:
    List of ServerDescriptor.
  '''
  return [
    ServerDescriptor(
      name='retail-ops',
      transport='stdio',
      command=sys.executable,
      args=['-m', 'chapter01_mcp.servers.retail_main'],
      summary='Retail inventory, sales trends, and restock ordering',
      hints=[
        'retail', 'store', 'inventory', 'stock', 'restock', 'sku',
        'sales', 'reorder',
      ],
      enabled=True,
    ),
    ServerDescriptor(
      name='telecom-ops',
      transport='stdio',
      command=sys.executable,
      args=['-m', 'chapter01_mcp.servers.telecom_main'],
      summary='Cell-site health, degraded sites, and field-technician dispatch',
      hints=['telecom', 'site', 'cell', 'network', 'latency', 'dispatch',
             'technician', 'degraded', 'outage'],
      enabled=True,
    ),
  ]


class ServerCatalog:
  '''Manages descriptors and serves discovery queries.'''

  def __init__(self, servers: Optional[List[ServerDescriptor]] = None) -> None:
    '''Initialize the catalog.

    Args:
      servers: Descriptors; defaults to `default_catalog()`.
    '''
    self._servers: Dict[str, ServerDescriptor] = {}
    for descriptor in (servers or default_catalog()):
      self.add(descriptor)

  def add(self, descriptor: ServerDescriptor) -> None:
    '''Add (or replace) a server descriptor.

    Args:
      descriptor: The server to expose.
    '''
    self._servers[descriptor.name] = descriptor

  def names(self) -> List[str]:
    '''List enabled server names.

    Returns:
      Enabled server names.
    '''
    return [
      name for name, descriptor in self._servers.items() if descriptor.enabled
    ]

  def get(self, name: str) -> Optional[ServerDescriptor]:
    '''Fetch a descriptor by name.

    Args:
      name: Server name.

    Returns:
      Descriptor or None.
    '''
    return self._servers.get(name)

  def iter_servers(self) -> List[ServerDescriptor]:
    '''List all registered descriptors (enabled and disabled).

    Returns:
      Descriptor list in registration order.
    '''
    return list(self._servers.values())


_TOKEN_RE = re.compile(r'[a-zA-Z0-9]+')


def _task_tokens(text: str) -> set[str]:
  '''Lowercase word tokens of a task string.

  Args:
    text: Task text.

  Returns:
    Token set.
  '''
  return {token.lower() for token in _TOKEN_RE.findall(text or '')}


def pick_servers(
  task: str,
  catalog: ServerCatalog,
  max_servers: int = 2,
) -> List[ServerDescriptor]:
  '''Select the servers most relevant to a task (hint-based filtering).

  Args:
    task: The natural language task given to the agent.
    catalog: Server catalog.
    max_servers: Upper bound on servers to pick.

  Returns:
    Ranked ServerDescriptors (best first).
  '''
  tokens = _task_tokens(task)
  scored: List[Tuple[str, int, ServerDescriptor]] = []

  for descriptor in catalog.iter_servers():
    if not descriptor.enabled:
      continue
    score = hint_hits_score(descriptor, tokens)
    scored.append((descriptor.name, score, descriptor))

  scored.sort(key=lambda item: (-item[1], item[0]))
  selected = [d for _, score, d in scored if score > 0][:max_servers]

  if not selected:
    # Fall back to every enabled server (better to over-provision than fail).
    selected = [d for d in catalog.iter_servers() if d.enabled]

  logger.info(
    'server selection',
    extra={'extra_fields': {
      'picked': [d.name for d in selected],
    }},
  )
  return selected


def hint_hits_score(descriptor: ServerDescriptor, tokens: set[str]) -> int:
  '''Score a server against task tokens.

  Args:
    descriptor: Server descriptor with hints/summary.
    tokens: Task tokens.

  Returns:
    Integer relevance score.
  '''
  hint_hits = sum(1 for hint in descriptor.hints if hint in tokens)
  summary_tokens = _task_tokens(descriptor.summary)
  overlap = len(tokens & summary_tokens)
  return hint_hits * 3 + overlap


def pick_tools(
  task: str,
  tools: List[ToolDescriptor],
  max_tools: int = 6,
) -> List[ToolDescriptor]:
  '''Filter a server's tools by task relevance (keyword match).

  When every tool matches or none match, all are kept (better to give the
  model the full set than to starve it).

  Args:
    task: Task text.
    tools: Server tools.
    max_tools: Maximum tools to keep.

  Returns:
    Filtered tool list.
  '''
  if len(tools) <= max_tools:
    return tools

  tokens = _task_tokens(task)
  scored: List[Tuple[ToolDescriptor, int]] = []

  for tool in tools:
    text = f'{tool.name} {tool.description}'.lower()
    tool_tokens = set(_TOKEN_RE.findall(text))
    score = len(tokens & tool_tokens)
    scored.append((tool, score))

  scored.sort(key=lambda item: -item[1])
  selected = [tool for tool, score in scored[:max_tools] if score > 0]
  if not selected:
    selected = tools[:max_tools]

  return selected
