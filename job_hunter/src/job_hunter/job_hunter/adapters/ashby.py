#!/usr/bin/env python
# -- coding: utf-8 --

'''Ashby job-board API adapter.

Endpoint: GET https://api.ashbyhq.com/posting-api/job-board/{org}?includeCompensation=true
'''


from __future__ import annotations

from typing import List

from job_hunter.adapters.base import SourceAdapter
from job_hunter.core.models import CompanyTarget, RawJobRecord


def parse_compensation(comp: dict) -> str:
  '''Render an Ashby compensation block as a salary string.

  Args:
    comp: Compensation mapping.

  Returns:
    Human-readable summary or empty string.
  '''
  if not isinstance(comp, dict):
    return ''
  period = comp.get('period') or ''
  chunks = []
  for tranch in comp.get('tranches') or []:
    low = ((tranch.get('compensationTierSummary') or {}).get('minimumAmount'))
    high = ((tranch.get('compensationTierSummary') or {}).get('maximumAmount'))
    currency = (tranch.get('currency') or 'INR')
    if low or high:
      chunks.append(f'{currency} {low}-{high} {period}')
  return '; '.join(chunks)


def parse_payload(payload: dict, source_key: str = 'ashby') -> List[RawJobRecord]:
  '''Convert an Ashby posting-api payload to raw records.

  Args:
    payload: Decoded JSON.
    source_key: Source identifier.

  Returns:
    Raw job records.
  '''
  records: List[RawJobRecord] = []
  for item in payload.get('jobs') or []:
    location_bits = [
      str(item.get('location') or ''),
      str(item.get('secondaryLocation') or ''),
    ]
    records.append(RawJobRecord(
      source_key=source_key,
      external_id=str(item.get('id') or ''),
      url=str(item.get('jobUrl') or item.get('applyUrl') or ''),
      title=str(item.get('title') or '').strip(),
      location_text=' / '.join(bit for bit in location_bits if bit),
      employment_type_raw=str(item.get('employmentType') or ''),
      work_mode_hint='remote' if item.get('isRemote') else '',
      description_text=str(
        item.get('descriptionPlain')
        or item.get('descriptionHtml')
        or item.get('description')
        or '',
      ),
      salary_raw=parse_compensation(item.get('compensation')),
      posted_at=item.get('publishedAt') or item.get('updatedAt'),
    ))
  return [record for record in records if record.title and record.url]


class AshbyAdapter(SourceAdapter):
  '''Fetch postings from an Ashby job board org.'''

  source_key = 'ashby'

  async def fetch(self, target: CompanyTarget, limit: int = 200) -> List[RawJobRecord]:
    '''Fetch all postings for the org.

    Args:
      target: Company with board_ref set.
      limit: Safety cap.

    Returns:
      Raw job records.
    '''
    url = f'https://api.ashbyhq.com/posting-api/job-board/{target.board_ref}'
    payload = await self._http.get_json(
      url,
      params={'includeCompensation': 'true'},
      method='POST',
      rpm=30,
    )
    return parse_payload(dict(payload), self.source_key)[:limit]

  async def health(self, target: CompanyTarget) -> bool:
    '''Probe the org endpoint.

    Args:
      target: Company with board_ref set.

    Returns:
      True when the API answers.
    '''
    try:
      await self._http.get_json(
        f'https://api.ashbyhq.com/posting-api/job-board/{target.board_ref}',
        method='POST',
      )
      return True
    except Exception:
      return False
