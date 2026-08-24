#!/usr/bin/env python
# -- coding: utf-8 --

'''Workable widget API adapter.

Endpoint: GET https://apply.workable.com/api/v1/widget/accounts/{account}?details=true
'''


from __future__ import annotations

from typing import List

from job_hunter.adapters.base import SourceAdapter
from job_hunter.core.models import CompanyTarget, RawJobRecord


def parse_payload(payload: dict, source_key: str = 'workable') -> List[RawJobRecord]:
  '''Convert a Workable widget payload to raw records.

  Args:
    payload: Decoded JSON.
    source_key: Source identifier.

  Returns:
    Raw job records.
  '''
  records: List[RawJobRecord] = []
  for item in payload.get('jobs') or []:
    location = item.get('location') or {}
    bits = [
      str(location.get('city') or ''),
      str(location.get('region') or ''),
      str(location.get('country') or ''),
    ]
    records.append(RawJobRecord(
      source_key=source_key,
      external_id=str(item.get('id') or item.get('shortcode') or ''),
      url=str(item.get('url') or item.get('shortlink') or ''),
      title=str(item.get('title') or '').strip(),
      location_text=', '.join(bit for bit in bits if bit),
      employment_type_raw=str(item.get('employment_type') or ''),
      work_mode_hint='remote' if item.get('remote') else '',
      description_html=str(item.get('description') or ''),
      description_text=str(item.get('description_text') or ''),
      posted_at=item.get('created_at') or item.get('published_on'),
    ))
  return [record for record in records if record.title and record.url]


class WorkableAdapter(SourceAdapter):
  '''Fetch postings from a Workable account.'''

  source_key = 'workable'

  async def fetch(self, target: CompanyTarget, limit: int = 200) -> List[RawJobRecord]:
    '''Fetch all postings for the account.

    Args:
      target: Company with board_ref set.
      limit: Safety cap.

    Returns:
      Raw job records.
    '''
    url = f'https://apply.workable.com/api/v1/widget/accounts/{target.board_ref}'
    payload = await self._http.get_json(url, params={'details': 'true'}, rpm=30)
    return parse_payload(dict(payload), self.source_key)[:limit]

  async def health(self, target: CompanyTarget) -> bool:
    '''Probe the account endpoint.

    Args:
      target: Company with board_ref set.

    Returns:
      True when the API answers.
    '''
    try:
      await self._http.get_json(
        f'https://apply.workable.com/api/v1/widget/accounts/{target.board_ref}',
      )
      return True
    except Exception:
      return False
