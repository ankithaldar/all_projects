#!/usr/bin/env python
# -- coding: utf-8 --

'''Greenhouse board API adapter.

Endpoint: GET https://boards-api.greenhouse.io/v1/boards/{token}/jobs?content=true
'''


from __future__ import annotations

from typing import Any, List

from job_hunter.adapters.base import SourceAdapter
from job_hunter.core.models import CompanyTarget, RawJobRecord


def parse_payload(payload: dict, source_key: str = 'greenhouse') -> List[RawJobRecord]:
  '''Convert a Greenhouse boards payload to raw records.

  Args:
    payload: Decoded JSON from the boards API.
    source_key: Source identifier for records.

  Returns:
    Raw job records.
  '''
  records: List[RawJobRecord] = []
  for item in payload.get('jobs') or []:
    location = (item.get('location') or {})
    records.append(RawJobRecord(
      source_key=source_key,
      external_id=str(item.get('id') or ''),
      url=str(item.get('absolute_url') or ''),
      title=str(item.get('title') or '').strip(),
      location_text=str(location.get('name') or ''),
      description_html=str(item.get('content') or ''),
      posted_at=item.get('updated_at'),
      extra={'offices': item.get('offices') or []},
    ))
  return [record for record in records if record.title and record.url]


class GreenhouseAdapter(SourceAdapter):
  '''Fetch postings from a Greenhouse board token.'''

  source_key = 'greenhouse'

  async def fetch(self, target: CompanyTarget, limit: int = 200) -> List[RawJobRecord]:
    '''Fetch all postings for the board token.

    Args:
      target: Company with board_ref set.
      limit: Safety cap.

    Returns:
      Raw job records.
    '''
    url = f'https://boards-api.greenhouse.io/v1/boards/{target.board_ref}/jobs'
    payload = await self._http.get_json(url, params={'content': 'true'}, rpm=30)
    return parse_payload(dict(payload), self.source_key)[:limit]

  async def health(self, target: CompanyTarget) -> bool:
    '''Probe the board endpoint without heavy content.

    Args:
      target: Company with board_ref set.

    Returns:
      True when the API answers.
    '''
    try:
      await self._http.get_json(
        f'https://boards-api.greenhouse.io/v1/boards/{target.board_ref}/jobs',
      )
      return True
    except Exception:
      return False
