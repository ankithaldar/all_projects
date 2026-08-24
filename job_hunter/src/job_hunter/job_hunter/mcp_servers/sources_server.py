#!/usr/bin/env python
# -- coding: utf-8 --

'''MCP server exposing source fetching and web search tools.

Run: python -m job_hunter.mcp_servers.sources_server
'''


from __future__ import annotations

import re
from mcp.server.fastmcp import FastMCP
from bs4 import BeautifulSoup
from job_hunter.mcp_servers.common import config_path, dumps, ensure_sys_path

ensure_sys_path()

import httpx  # noqa: E402
from job_hunter.adapters.career_page import CareerPageDetector  # noqa: E402
from job_hunter.adapters.http_client import HttpClient, USER_AGENT  # noqa: E402
from job_hunter.adapters.registry import build_adapter  # noqa: E402
from job_hunter.core.bootstrap import bootstrap  # noqa: E402
from job_hunter.core.models import CompanyTarget  # noqa: E402

mcp = FastMCP('job_hunter-sources')
_http = HttpClient()
_detector = CareerPageDetector(_http)


@mcp.tool()
def fetch_ats_board(provider: str, board_ref: str, limit: int = 100) -> str:
  '''Fetch postings from one ATS board.

  Args:
    provider: One of greenhouse|lever|ashby|workable|smartrecruiters|recruitee|personio.
    board_ref: Provider-specific token/org/account.
    limit: Max records.

  Returns:
    JSON array of raw job records.
  '''
  try:
    target = CompanyTarget(name=board_ref, source_key=provider, board_ref=board_ref)
    adapter = build_adapter(provider, _http)
    import asyncio
    records = asyncio.run(adapter.fetch(target, limit=min(limit, 300)))
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
    html = await _http.get_text(url)
    soup = BeautifulSoup(html or '', 'html.parser')
    for tag in soup(['script', 'style', 'noscript']):
      tag.decompose()
    return dumps({
      'url': url,
      'title': soup.title.get_text(strip=True) if soup.title else '',
      'text': ' '.join(soup.get_text(' ').split())[:20000],
    })
  except Exception as exc:
    return dumps({'error': str(exc), 'url': url})


@mcp.tool()
def search_web(query: str, max_results: int = 8) -> str:
  '''Search the web via the DuckDuckGo HTML endpoint.

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
    await _http.throttle_for('https://duckduckgo.com', 10).wait()
    async with httpx.AsyncClient(headers={'User-Agent': USER_AGENT}, follow_redirects=True, timeout=20) as client:
      response = await client.get('https://duckduckgo.com/html/', params={'q': query})
    soup = BeautifulSoup(response.text, 'html.parser')
    results = []
    for anchor in soup.select('a.result__a')[: max(1, min(max_results, 15))]:
      href = anchor.get('href') or ''
      match = re.search(r'uddg=([^&]+)', str(href))
      if match:
        from urllib.parse import unquote
        href = unquote(match.group(1))
      results.append({'title': anchor.get_text(' ', strip=True), 'href': href})
    return results
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
    result = asyncio.run(_detector.detect(domain))
    if result is None:
      return dumps({'found': False})
    provider, ref, careers_url = result
    return dumps({
      'found': True,
      'provider': provider,
      'board_ref': ref,
      'careers_url': careers_url,
    })
  except Exception as exc:
    return dumps({'found': False, 'error': str(exc)})


def main() -> None:
  '''Prepare data dirs then serve over stdio.'''
  bootstrap(config_path())
  mcp.run()


if __name__ == '__main__':
  main()
