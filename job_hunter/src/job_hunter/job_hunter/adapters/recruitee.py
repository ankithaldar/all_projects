#!/usr/bin/env python
# -- coding: utf-8 --

'''Recruitee public offers API adapter.

Endpoint: GET https://{company}.recruitee.com/api/offers/
'''


from __future__ import annotations

from typing import List

from job_hunter.adapters.base import SourceAdapter
from job_hunter.core.models import CompanyTarget, RawJobRecord


def parse_payload(payload: dict) -> List[RawJobRecord]:
  '''Convert a Recruitee offers payload to raw records.

  Args:
    payload: Decoded JSON.

  Returns:
    Raw job records.
  '''
  records: List[RawJobRecord] = []
  for item in payload.get('offers') or []:
    records.append(RawJobRecord(
      source_key='recruitee',
      external_id=str(item.get('id') or ''),
      url=str(item.get('careers_url') or item.get('url') or ''),
      title=str(item.get('title') or '').strip(),
      location_text=str(item.get('location') or ''),
      employment_type_raw=str(item.get('employment_type') or ''),
      work_mode_hint='remote' if item.get('remote') else '',
      description_html=str(item.get('description') or ''),
      posted_at=item.get('published_at'),
    ))
  return [record for record in records if record.title and record.url]


class RecruiteeAdapter(SourceAdapter):
  '''Fetch postings from a Recruitee subdomain.'''

  source_key = 'recruitee'

  async def fetch(self, target: CompanyTarget, limit: int = 200) -> List[RawJobRecord]:
    '''Fetch all offers for the subdomain.

    Args:
      target: Company with board_ref set (subdomain slug).
      limit: Safety cap.

    Returns:
      Raw job records.
    '''
    url = f'https://{target.board_ref}.recruitee.com/api/offers/'
    payload = await self._http.get_json(url, rpm=20)
    return parse_payload(dict(payload))[:limit]

  async def health(self, target: CompanyTarget) -> bool:
    '''Probe the offers endpoint.

    Args:
      target: Company with board_ref set.

    Returns:
      True when the API answers.
    '''
    try:
      await self._http.get_json(f'https://{target.board_ref}.recruitee.com/api/offers/')
      return True
    except Exception:
      return False
