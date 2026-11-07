#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''MCP server exposing source fetching and web search tools.

Run: python -m job_hunter.mcp_servers.sources_server
'''

from __future__ import annotations

import threading

try:
  from mcp.server.fastmcp import FastMCP
except ModuleNotFoundError:  # mcp 2.x renamed FastMCP to MCPServer
  from mcp.server.mcpserver import MCPServer as FastMCP
from bs4 import BeautifulSoup

from job_hunter.mcp_servers.common import config_path, dumps, ensure_sys_path

ensure_sys_path()

from job_hunter.adapters.base import valid_domain  # noqa: E402
from job_hunter.adapters.career_page import CareerPageDetector  # noqa: E402
from job_hunter.adapters.http_client import HttpClient  # noqa: E402
from job_hunter.adapters.registry import build_adapter  # noqa: E402
from job_hunter.core.bootstrap import bootstrap  # noqa: E402
from job_hunter.core.models import CompanyTarget  # noqa: E402
from job_hunter.services.normalizer import html_to_text  # noqa: E402
from job_hunter.services.web_search import WebSearchIndex  # noqa: E402

mcp = FastMCP('job_hunter-sources')
_http_instance: HttpClient | None = None
_detector_instance: CareerPageDetector | None = None
_http_lock = threading.Lock()
_detector_lock = threading.Lock()


def _http() -> HttpClient:
  '''Return the lazily created shared HttpClient.

  Returns:
    HttpClient, created on first call. Tests may monkeypatch this module's
    HttpClient or replace _http with a fake client directly.
  '''
  global _http_instance
  if _http_instance is None:
    with _http_lock:
      if _http_instance is None:
        _http_instance = HttpClient()
  return _http_instance


def _detector() -> CareerPageDetector:
  '''Return the lazily created careers detector.

  Returns:
    CareerPageDetector bound to the shared HttpClient.
  '''
  global _detector_instance
  if _detector_instance is None:
    with _detector_lock:
      if _detector_instance is None:
        _detector_instance = CareerPageDetector(_resolve_http())
  return _detector_instance


def _resolve_http() -> HttpClient:
  '''Return the current HTTP client, honouring monkeypatched fakes.

  The module attribute ``_http`` is the lazy getter; tests that replace
  it with a client instance (instead of a getter) are used as-is.
  '''
  current = globals().get('_http')
  if callable(current):
    return current()
  if current is not None:
    return current
  return _http()


def _resolve_detector() -> CareerPageDetector:
  '''Return the current detector, honouring monkeypatched fakes.'''
  current = globals().get('_detector')
  if callable(current):
    return current()
  if current is not None:
    return current
  return _detector()



@mcp.tool()
def fetch_ats_board(provider: str, board_ref: str, limit: int = 100) -> str:
  '''Fetch postings from one ATS board.

  Args:
    provider: One of greenhouse|lever|ashby|workable|smartrecruiters|
      recruitee|personio.
    board_ref: Provider-specific token/org/account.
    limit: Max records.

  Returns:
    JSON array of raw job records.
  '''
  try:
    target = CompanyTarget(
      name=board_ref, source_key=provider, board_ref=board_ref
    )
    adapter = build_adapter(provider, _resolve_http())
    import asyncio

    records = asyncio.run(adapter.fetch(target, limit=max(1, min(limit, 300))))
    return dumps([record.model_dump() for record in records])
  except Exception as exc:
    return dumps({'error': str(exc)})


@mcp.tool()
async def fetch_page(url: str) -> str:
  '''Fetch a page body as cleaned text (robots-aware).

  Args:
    url: Page URL.

  Returns:
    JSON with title, text, url.
  '''
  try:
    html = await _resolve_http().get_text(url)
    soup = BeautifulSoup(html or '', 'html.parser')
    return dumps(
      {
        'url': url,
        'title': soup.title.get_text(strip=True) if soup.title else '',
        'text': html_to_text(html)[:20000],
      }
    )
  except Exception as exc:
    return dumps({'error': str(exc), 'url': url})


@mcp.tool()
def search_web(query: str, max_results: int = 8) -> str:
  '''Search the web via the shared web-search index.

  Args:
    query: Query text.
    max_results: Max results.

  Returns:
    JSON array of {title, href} items.
  '''
  import asyncio

  async def run() -> list:
    '''Perform the search request.

    Returns:
      Result item mappings.
    '''
    index = WebSearchIndex()
    try:
      results = await index.search(query, limit=max(1, min(max_results, 15)))
      return [
        {'title': item.get('title', ''), 'href': item.get('href', '')}
        for item in results
      ]
    finally:
      await index.close()

  try:
    return dumps(asyncio.run(run()))
  except Exception as exc:
    return dumps({'error': str(exc)})


@mcp.tool()
def detect_careers(domain: str) -> str:
  '''Fingerprint a company domain for its ATS careers board.

  Args:
    domain: Registrable domain like example.com.

  Returns:
    JSON with provider, board_ref, careers_url, or found=false.
  '''
  import asyncio

  try:
    candidate = (domain or '').strip().lower()
    for prefix in ('https://', 'http://'):
      candidate = candidate.removeprefix(prefix)
    candidate = candidate.strip('/')
    if not candidate:
      raise ValueError(f'invalid domain: {domain!r}')
    domain = valid_domain(candidate)
    result = asyncio.run(_resolve_detector().detect(domain))
    if result is None:
      return dumps({'found': False})
    provider, ref, careers_url = result
    return dumps(
      {
        'found': True,
        'provider': provider,
        'board_ref': ref,
        'careers_url': careers_url,
      }
    )
  except Exception as exc:
    return dumps({'found': False, 'error': str(exc)})


def main() -> None:
  '''Prepare data dirs then serve over stdio.'''
  bootstrap(config_path())
  mcp.run()


if __name__ == '__main__':
  main()
