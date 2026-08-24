#!/usr/bin/env python
# -- coding: utf-8 --

'''Adapter registry and company-driven fetch dispatch.'''


from __future__ import annotations

from typing import Dict, List, Optional, Type

from job_hunter.adapters.aggregators import AggregatorAdapter
from job_hunter.adapters.ashby import AshbyAdapter
from job_hunter.adapters.base import SourceAdapter
from job_hunter.adapters.greenhouse import GreenhouseAdapter
from job_hunter.adapters.http_client import HttpClient
from job_hunter.adapters.lever import LeverAdapter
from job_hunter.adapters.personio import PersonioAdapter
from job_hunter.adapters.recruitee import RecruiteeAdapter
from job_hunter.adapters.smartrecruiters import SmartRecruitersAdapter
from job_hunter.adapters.workable import WorkableAdapter
from job_hunter.adapters.workday import WorkdayAdapter
from job_hunter.core.errors import AdapterError
from job_hunter.core.models import CompanyTarget, RawJobRecord

REGISTRY: Dict[str, Type[SourceAdapter]] = {
  'greenhouse': GreenhouseAdapter,
  'lever': LeverAdapter,
  'ashby': AshbyAdapter,
  'workable': WorkableAdapter,
  'workday': WorkdayAdapter,
  'smartrecruiters': SmartRecruitersAdapter,
  'recruitee': RecruiteeAdapter,
  'personio': PersonioAdapter,
  'remotive': AggregatorAdapter,
  'remoteok': AggregatorAdapter,
  'weworkremotely': AggregatorAdapter,
}


def build_adapter(source_key: str, http: HttpClient) -> SourceAdapter:
  '''Instantiate the adapter for a source key.

  Args:
    source_key: Registered source identifier.
    http: Shared HTTP client.

    Returns:
      Adapter instance.

    Raises:
      AdapterError: For unknown source keys.
    '''
  cls = REGISTRY.get(source_key)
  if cls is None:
    raise AdapterError(f'unknown source key: {source_key}', source=source_key)
  return cls(http)


async def fetch_for_target(
  target: CompanyTarget,
  http: HttpClient,
  limit: int = 200,
) -> List[RawJobRecord]:
  '''Fetch postings for one (source, board_ref) pair.

  Args:
    target: Company plus source/board info.
    http: Shared HTTP client.
    limit: Safety cap.

  Returns:
    Raw job records (possibly empty).
  '''
  adapter = build_adapter(target.source_key, http)
  return await adapter.fetch(target, limit=limit)


def detect_provider_from_urls(urls: List[str]) -> Optional[tuple]:
  '''Best-effort fingerprint from a list of arbitrary URLs.

  Args:
    urls: Candidate URLs (careers page links).

  Returns:
    (provider, ref) or None.
  '''
  from job_hunter.adapters.career_page import extract_fingerprint
  for url in urls:
    found = extract_fingerprint(f'href="{url}"')
    if found:
      return found
  return None
