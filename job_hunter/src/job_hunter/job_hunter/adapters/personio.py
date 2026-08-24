#!/usr/bin/env python
# -- coding: utf-8 --

'''Personio XML feed adapter.

Endpoint: GET https://{company}.jobs.personio.de/xml
'''


from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import List

from job_hunter.adapters.base import SourceAdapter
from job_hunter.core.models import CompanyTarget, RawJobRecord


def parse_xml(text: str) -> List[RawJobRecord]:
  '''Parse a Personio XML feed body.

  Args:
    text: Raw XML document.

  Returns:
    Raw job records.
  '''
  records: List[RawJobRecord] = []
  try:
    root = ET.fromstring(text)
  except ET.ParseError:
    return records
  for position in root.iter('position'):
    fields = {child.tag: (child.text or '').strip() for child in position}
    title = fields.get('name', '')
    url = fields.get('detailUrl', '')
    if not title or not url:
      continue
    records.append(RawJobRecord(
      source_key='personio',
      external_id=fields.get('id', ''),
      url=url,
      title=title,
      location_text=fields.get('office', ''),
      employment_type_raw=fields.get('employmentType', ''),
      description_html=fields.get('jobDescriptions', '') or fields.get('description', ''),
    ))
  return records


class PersonioAdapter(SourceAdapter):
  '''Fetch postings from a Personio jobs subdomain.'''

  source_key = 'personio'

  async def fetch(self, target: CompanyTarget, limit: int = 200) -> List[RawJobRecord]:
    '''Fetch and parse the XML feed.

    Args:
      target: Company with board_ref set (subdomain slug).
      limit: Safety cap.

    Returns:
      Raw job records.
    '''
    url = f'https://{target.board_ref}.jobs.personio.de/xml'
    text = await self._http.get_text(url, rpm=20)
    return parse_xml(text)[:limit]

  async def health(self, target: CompanyTarget) -> bool:
    '''Probe the XML feed.

    Args:
      target: Company with board_ref set.

    Returns:
      True when the feed returns parseable XML.
    '''
    try:
      text = await self._http.get_text(
        f'https://{target.board_ref}.jobs.personio.de/xml',
      )
      return bool(parse_xml(text))
    except Exception:
      return False
