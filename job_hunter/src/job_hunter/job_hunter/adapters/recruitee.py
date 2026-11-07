#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Recruitee public offers API adapter.

Endpoint: GET https://{company}.recruitee.com/api/offers/
'''

from __future__ import annotations

from typing import List

from job_hunter.adapters.base import SourceAdapter, validate_board_ref
from job_hunter.core.models import CompanyTarget, RawJobRecord


def parse_payload(payload: dict) -> List[RawJobRecord]:
  '''Convert a Recruitee offers payload to raw records.

  Args:
    payload: Decoded JSON.

  Returns:
    Raw job records.
  '''
  if not isinstance(payload, dict):
    return []
  records: List[RawJobRecord] = []
  for item in payload.get('offers') or []:
    if not isinstance(item, dict):
      continue
    records.append(
      RawJobRecord(
        source_key='recruitee',
        external_id=str(item.get('id') or ''),
        url=str(item.get('careers_url') or item.get('url') or ''),
        title=str(item.get('title') or '').strip(),
        location_text=str(item.get('location') or ''),
        employment_type_raw=str(item.get('employment_type') or ''),
        work_mode_hint='remote' if item.get('remote') else '',
        description_html=str(item.get('description') or ''),
        posted_at=item.get('published_at'),
      )
    )
  return [record for record in records if record.title and record.url]


class RecruiteeAdapter(SourceAdapter):
  '''Fetch postings from a Recruitee subdomain.'''

  source_key = 'recruitee'

  async def fetch(
    self, target: CompanyTarget, limit: int = 200
  ) -> List[RawJobRecord]:
    '''Fetch all offers for the subdomain.

    Args:
      target: Company with board_ref set (subdomain slug).
      limit: Safety cap.

    Returns:
      Raw job records.
    '''
    ref = validate_board_ref(target.board_ref, source=self.source_key)
    url = f'https://{ref}.recruitee.com/api/offers/'
    payload = await self._http.get_json(url, rpm=20)
    return parse_payload(payload)[:max(1, limit)]
