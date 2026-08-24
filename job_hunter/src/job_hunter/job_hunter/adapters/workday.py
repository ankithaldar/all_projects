#!/usr/bin/env python
# -- coding: utf-8 --

'''Workday CXS public jobs API adapter.

Endpoint (no auth):
  POST https://{tenant}.wd{n}.myworkdayjobs.com/wday/cxs/{tenant}/{site}/jobs
board_ref format: '{tenant}.{wdN}.myworkdayjobs.com/{site}'
'''


from __future__ import annotations

from typing import List

from job_hunter.adapters.base import SourceAdapter
from job_hunter.core.models import CompanyTarget, RawJobRecord


def parse_payload(payload: dict, host: str, site: str) -> List[RawJobRecord]:
  '''Convert a Workday CXS jobs response to raw records.

  Args:
    payload: Decoded JSON.
    host: Full myworkdayjobs host.
    site: Career site path segment.

  Returns:
    Raw job records.
  '''
  records: List[RawJobRecord] = []
  for item in payload.get('jobPostings') or []:
    external_path = str(item.get('externalPath') or '')
    title = str(item.get('title') or '').strip()
    if not title or not external_path:
      continue
    records.append(RawJobRecord(
      source_key='workday',
      external_id=external_path,
      url=f'https://{host}/en-US/{site}{external_path}',
      title=title,
      location_text=str(item.get('locationsText') or ''),
      extra={'bulletFields': item.get('bulletFields') or []},
    ))
  return records


class WorkdayAdapter(SourceAdapter):
  '''Fetch postings from a Workday tenant career site.'''

  source_key = 'workday'

  async def fetch(self, target: CompanyTarget, limit: int = 200) -> List[RawJobRecord]:
    '''Fetch up to limit postings page by page.

    Args:
      target: board_ref = '{host}/{site}'.
      limit: Safety cap.

    Returns:
      Raw job records.

    Raises:
      AdapterError: On malformed board_ref.
    '''
    if '/' not in target.board_ref:
      from job_hunter.core.errors import AdapterError
      raise AdapterError(f'bad workday board_ref: {target.board_ref}', source='workday')
    host, site = target.board_ref.split('/', 1)
    base = f'https://{host}/wday/cxs/{host.split(".")[0]}/{site}/jobs'
    records: List[RawJobRecord] = []
    offset = 0
    page_size = 20
    while len(records) < limit:
      payload = await self._http.get_json(
        base,
        params={'limit': page_size, 'offset': offset, 'searchText': ''},
        method='POST',
        rpm=20,
      )
      if not isinstance(payload, dict):
        break
      batch = parse_payload(payload, host, site)
      records.extend(batch)
      total = int(payload.get('total') or 0)
      offset += page_size
      if not batch or offset >= total or offset >= limit:
        break
    return records[:limit]

  async def health(self, target: CompanyTarget) -> bool:
    '''Probe the CXS endpoint with a single posting request.

    Args:
      target: Company with board_ref set.

    Returns:
      True when the endpoint answers valid JSON.
    '''
    try:
      await self.fetch(target, limit=1)
      return True
    except Exception:
      return False
