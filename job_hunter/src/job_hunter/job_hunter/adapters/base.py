#!/usr/bin/env python
# -- coding: utf-8 --

'''Abstract source adapter contract.'''


from __future__ import annotations

import abc
from typing import List

from job_hunter.core.models import CompanyTarget, RawJobRecord


class SourceAdapter(abc.ABC):
  '''One job-source integration: fetch postings for a target.'''

  source_key: str = ''

  def __init__(self, http: 'HttpClient') -> None:
    '''Initialize the adapter.

    Args:
      http: Shared polite HTTP client.
    '''
    self._http = http

  @abc.abstractmethod
  async def fetch(self, target: CompanyTarget, limit: int = 200) -> List[RawJobRecord]:
    '''Fetch current postings for one company/board.

    Args:
      target: Company plus board reference.
      limit: Safety cap on returned records.

    Returns:
      Raw posting records.
    '''

  @abc.abstractmethod
  async def health(self, target: CompanyTarget) -> bool:
    '''Cheap liveness probe for the configured board reference.

    Args:
      target: Company plus board reference.

    Returns:
      True when the board responds successfully.
    '''
