#!/usr/bin/env python
# -- coding: utf-8 --

'''SmartRecruiters public API adapter.

Endpoints:
  GET https://api.smartrecruiters.com/v1/companies/{company}/postings
  GET https://api.smartrecruiters.com/v1/companies/{company}/postings/{id}
'''


from __future__ import annotations

import asyncio
from typing import List

from job_hunter.adapters.base import SourceAdapter
from job_hunter.core.models import CompanyTarget, RawJobRecord


def parse_list_payload(payload: dict, company: str) -> List[RawJobRecord]:
  '''Convert a SmartRecruiters postings list to partial raw records.

  Args:
    payload: Decoded JSON.
    company: Company handle used for URL construction.

  Returns:
    Raw records (descriptions pending detail fetch).
  '''
  records: List[RawJobRecord] = []
  for item in payload.get('content') or []:
    location = item.get('location') or {}
    bits = [
      str(location.get('city') or ''),
      str(location.get('region') or ''),
      str(location.get('country') or ''),
    ]
    posting_id = str(item.get('id') or '')
    records.append(RawJobRecord(
      source_key='smartrecruiters',
      external_id=posting_id,
      url=(
        f'https://jobs.smartrecruiters.com/{company}/{posting_id}'
        if posting_id and company else ''
      ),
      title=str(item.get('name') or '').strip(),
      location_text=', '.join(bit for bit in bits if bit),
      employment_type_raw=str(item.get('typeOfEmployment', {}).get('label') or ''),
      extra={'released_date': item.get('releasedDate')},
    ))
  return [record for record in records if record.title and record.url]


def merge_detail(record: RawJobRecord, detail: dict) -> RawJobRecord:
  '''Merge a posting-detail payload into a partial record.

  Args:
    record: Partial record from the list payload.
    detail: Detail JSON.

  Returns:
    Record with description filled in.
  '''
  description = str(
    (detail.get('jobAd') or {}).get('sections', {}).get('description', {}).get('text')
    or '',
  )
  return record.model_copy(update={
    'description_html': description,
    'posted_at': detail.get('releasedDate'),
  })


class SmartRecruitersAdapter(SourceAdapter):
  '''Fetch postings from a SmartRecruiters company handle.'''

  source_key = 'smartrecruiters'

  async def _fetch_detail(self, company: str, posting_id: str) -> dict:
    '''Fetch one posting detail.

    Args:
      company: Company handle.
      posting_id: Posting id.

    Returns:
      Detail JSON (empty on failure).
    '''
    try:
      url = f'https://api.smartrecruiters.com/v1/companies/{company}/postings/{posting_id}'
      return dict(await self._http.get_json(url, rpm=20))
    except Exception:
      return {}

  async def fetch(self, target: CompanyTarget, limit: int = 200) -> List[RawJobRecord]:
    '''Fetch postings, enriching up to 60 with full descriptions.

    Args:
      target: Company with board_ref set.
      limit: Safety cap.

    Returns:
      Raw job records.
    '''
    url = f'https://api.smartrecruiters.com/v1/companies/{target.board_ref}/postings'
    payload = await self._http.get_json(url, params={'limit': min(limit, 100)}, rpm=20)
    records = parse_list_payload(dict(payload), target.board_ref)[:limit]
    semaphore = asyncio.Semaphore(5)

    async def guarded(record: RawJobRecord) -> RawJobRecord:
      '''Fetch detail under concurrency cap.

      Args:
        record: Partial record.

      Returns:
        Merged record.
      '''
      async with semaphore:
        detail = await self._fetch_detail(target.board_ref, record.external_id)
      return merge_detail(record, detail) if detail else record

    enriched = await asyncio.gather(*[guarded(r) for r in records[:60]])
    return list(enriched)

  async def health(self, target: CompanyTarget) -> bool:
    '''Probe the postings endpoint.

    Args:
      target: Company with board_ref set.

    Returns:
      True when the API answers.
    '''
    try:
      url = f'https://api.smartrecruiters.com/v1/companies/{target.board_ref}/postings'
      await self._http.get_json(url)
      return True
    except Exception:
      return False
