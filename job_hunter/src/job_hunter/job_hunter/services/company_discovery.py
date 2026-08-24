#!/usr/bin/env python
# -- coding: utf-8 --

'''Company Scout: seed ingestion, posting harvest, and ATS verification.'''


from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from job_hunter.adapters.career_page import CareerPageDetector
from job_hunter.adapters.http_client import HttpClient
from job_hunter.core.config import AppSettings
from job_hunter.db.repositories.companies import CompaniesRepository
from job_hunter.db.repositories.jobs import JobsRepository

_DOMAIN_RE = re.compile(r'https?://([a-z0-9.-]+)', re.IGNORECASE)
_BLOCKED_SUFFIXES = (
  'google.com', 'youtube.com', 'linkedin.com', 'facebook.com', 'twitter.com',
  'x.com', 'medium.com', 'wikipedia.org', 'github.com', 'duckduckgo.com',
  'glassdoor.com', 'indeed.com', 'naukri.com', 'instagram.com',
)


def _registrable(url: str) -> str:
  '''Extract a candidate registrable domain from a URL.

  Args:
    url: Any URL.

  Returns:
    Lowercased host or empty string.
  '''
  match = _DOMAIN_RE.search(url)
  if not match:
    return ''
  host = match.group(1).lower()
  parts = host.split('.')
  return '.'.join(parts[-2:]) if len(parts) >= 2 else host


async def ingest_seeds(settings: AppSettings) -> int:
  '''Upsert every company listed in seeds/*.yaml files.

  Args:
    settings: Application settings.

  Returns:
    Number of seed rows processed.
  '''
  repos = CompaniesRepository(settings.db_path)
  count = 0
  for path in sorted(Path(settings.seeds_dir).glob('companies_*.yaml')):
    payload = yaml.safe_load(path.read_text(encoding='utf-8')) or {}
    for entry in payload.get('companies') or []:
      repos.upsert(
        name=str(entry.get('name') or '').strip(),
        domain=str(entry.get('domain') or '').strip(),
        vertical=entry.get('vertical_hint'),
        priority=int(entry.get('priority') or 3),
        discovered_via=f'seed:{path.name}',
      )
      count += 1
  return count


async def harvest_from_jobs(settings: AppSettings, limit: int = 200) -> int:
  '''Attach or create companies from raw posting names.

  Args:
    settings: Application settings.
    limit: Max unattached postings to process.

  Returns:
    Number of newly created companies.
  '''
  conn_jobs = JobsRepository(settings.db_path)
  conn_companies = CompaniesRepository(settings.db_path)
  import json as _json
  rows = conn_jobs.list_jobs(status='active', per_page=limit)
  created = 0
  for row in rows:
    if row.get('company_id'):
      continue
    raw_name = (row.get('company_raw_name') or '').strip()
    if not raw_name:
      continue
    existing_id = conn_companies.resolve_alias(raw_name)
    if existing_id is None:
      domain = ''
      existing_id = conn_companies.upsert(
        name=raw_name,
        status='needs_review',
        discovered_via='harvest:jobs',
        priority=2,
      )
      conn_companies.add_alias(raw_name, existing_id)
      created += 1
    conn_jobs.update_fields(int(row['id']), {'company_id': existing_id})
  _ = _json
  return created


async def verify_pending(settings: AppSettings, chunk: int = 50) -> Dict[str, int]:
  '''Fingerprint careers pages for companies lacking ATS details.

  Args:
    settings: Application settings.
    chunk: Max companies per pass.

  Returns:
    Counters {verified, failed}.
  '''
  repos = CompaniesRepository(settings.db_path)
  candidates = [
    row for row in repos.list_companies(limit=chunk * 3)
    if row.get('domain') and not row.get('ats_provider')
  ][:chunk]
  http = HttpClient()
  detector = CareerPageDetector(http)
  verified = failed = 0
  try:
    for row in candidates:
      result: Optional[Tuple[str, str, str]] = None
      try:
        result = await detector.detect(str(row['domain']))
      except Exception:
        result = None
      if result is None:
        repos.patch(int(row['id']), {'notes': 'ats detection failed'})
        repos.set_status(int(row['id']), 'needs_review')
        failed += 1
        continue
      provider, ref, careers_url = result
      repos.patch(int(row['id']), {
        'ats_provider': provider,
        'board_ref': ref,
        'careers_url': careers_url,
        'notes': '',
      })
      verified += 1
  finally:
    await http.close()
  return {'verified': verified, 'failed': failed}


async def expand_via_search(
  settings: AppSettings,
  query: str,
  max_new: int = 10,
) -> List[Dict[str, Any]]:
  '''Discover new candidate companies from web search result domains.

  Args:
    settings: Application settings.
    query: Search query like 'fintech product companies Bengaluru'.
    max_new: Safety cap on insertions.

  Returns:
    Newly inserted company rows (as mappings).
  '''
  from job_hunter.llm.mcp_tools import MCPClientManager
  repos = CompaniesRepository(settings.db_path)
  manager = MCPClientManager(settings)
  inserted: List[Dict[str, Any]] = []
  try:
    await manager.open()
    if 'sources' not in manager.sessions:
      return inserted
    raw = await manager.call_tool('sources.search_web', {'query': query, 'max_results': 15})
    import json
    results = json.loads(raw)
    if isinstance(results, dict):
      return inserted
    seen_domains: set = set()
    for item in results:
      if len(inserted) >= max_new:
        break
      domain = _registrable(str(item.get('href') or ''))
      if not domain or domain in seen_domains or domain.endswith(_BLOCKED_SUFFIXES):
        continue
      seen_domains.add(domain)
      name_guess = domain.split('.')[0].replace('-', ' ').title()
      if repos.resolve_alias(name_guess) is not None:
        continue
      company_id = repos.upsert(
        name=name_guess,
        domain=domain,
        status='needs_review',
        discovered_via=f'websearch:{query[:60]}',
        priority=2,
      )
      inserted.append(repos.get(company_id) or {})
  finally:
    await manager.close()
  return inserted


async def run_seed_ingestion(config_path: str | Path, seeds_dir: Path) -> int:
  '''CLI helper: bootstrap then ingest seeds from a directory.

  Args:
    config_path: app.yaml path.
    seeds_dir: Directory containing seeds.

  Returns:
    Rows ingested.
  '''
  settings = AppSettings(config_path)
  settings.raw.setdefault('_seeds_override', str(seeds_dir))
  from job_hunter.core.bootstrap import bootstrap
  bootstrap(config_path, seeds_dir=seeds_dir)
  return await ingest_seeds(settings)


_ = asyncio
