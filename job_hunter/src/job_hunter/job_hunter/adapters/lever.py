#!/usr/bin/env python
# -- coding: utf-8 --

'''Lever postings API adapter.

Endpoint: GET https://api.lever.co/v0/postings/{company}?mode=json
'''


from __future__ import annotations

from typing import List

from job_hunter.adapters.base import SourceAdapter
from job_hunter.core.models import CompanyTarget, RawJobRecord


def parse_payload(payload: list, source_key: str = 'lever') -> List[RawJobRecord]:
  '''Convert a Lever postings payload to raw records.

  Args:
    payload: Decoded JSON list from Lever.
    source_key: Source identifier.

  Returns:
    Raw job records.
  '''
  records: List[RawJobRecord] = []
  for item in payload or []:
    categories = item.get('categories') or {}
    description_parts = [
      str(item.get('descriptionPlain') or ''),
      str(item.get('description') or ''),
    ]
    records.append(RawJobRecord(
      source_key=source_key,
      external_id=str(item.get('id') or ''),
      url=str(item.get('hostedUrl') or item.get('applyUrl') or ''),
      title=str(item.get('text') or '').strip(),
      location_text=str(categories.get('location') or ''),
      employment_type_raw=str(categories.get('commitment') or ''),
      work_mode_hint=str(categories.get('workplaceType') or ''),
      description_text='\n'.join(part for part in description_parts if part),
      posted_at=str(item.get('createdAt') or '') or None,
      extra={'team': categories.get('team')},
    ))
  return [record for record in records if record.title and record.url]


class LeverAdapter(SourceAdapter):
  '''Fetch postings from a Lever company handle.'''

  source_key = 'lever'

  async def fetch(self, target: CompanyTarget, limit: int = 200) -> List[RawJobRecord]:
    '''Fetch all postings for the handle.

    Args:
      target: Company with board_ref set.
      limit: Safety cap.

    Returns:
      Raw job records.
    '''
    url = f'https://api.lever.co/v0/postings/{target.board_ref}'
    payload = await self._http.get_json(url, params={'mode': 'json'}, rpm=30)
    return parse_payload(list(payload), self.source_key)[:limit]

  async def health(self, target: CompanyTarget) -> bool:
    '''Probe the handle endpoint.

    Args:
      target: Company with board_ref set.

    Returns:
      True when the API answers.
    '''
    try:
      await self._http.get_json(f'https://api.lever.co/v0/postings/{target.board_ref}')
      return True
    except Exception:
      return False
