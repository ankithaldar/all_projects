#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Adapter registry and company-driven fetch dispatch.'''


from __future__ import annotations

from typing import Dict, Type

from job_hunter.adapters.aggregators import AggregatorAdapter
from job_hunter.adapters.ashby import AshbyAdapter
from job_hunter.adapters.base import SourceAdapter
from job_hunter.adapters.cutshort import CutshortAdapter
from job_hunter.adapters.greenhouse import GreenhouseAdapter
from job_hunter.adapters.http_client import HttpClient
from job_hunter.adapters.lever import LeverAdapter
from job_hunter.adapters.personio import PersonioAdapter
from job_hunter.adapters.recruitee import RecruiteeAdapter
from job_hunter.adapters.smartrecruiters import SmartRecruitersAdapter
from job_hunter.adapters.workable import WorkableAdapter
from job_hunter.adapters.workday import WorkdayAdapter
from job_hunter.core.errors import AdapterError

REGISTRY: Dict[str, Type[SourceAdapter]] = {
  'greenhouse': GreenhouseAdapter,
  'lever': LeverAdapter,
  'ashby': AshbyAdapter,
  'workable': WorkableAdapter,
  'workday': WorkdayAdapter,
  'smartrecruiters': SmartRecruitersAdapter,
  'recruitee': RecruiteeAdapter,
  'personio': PersonioAdapter,
  'remotive': AggregatorAdapter,
  'remoteok': AggregatorAdapter,
  'weworkremotely': AggregatorAdapter,
  'himalayas': AggregatorAdapter,
  'arbeitnow': AggregatorAdapter,
  'jobicy': AggregatorAdapter,
  'cutshort': CutshortAdapter,
}


def build_adapter(source_key: str, http: HttpClient) -> SourceAdapter:
  '''Instantiate the adapter for a source key.

  Args:
    source_key: Registered source identifier.
    http: Shared HTTP client.

    Returns:
      Adapter instance.

    Raises:
      AdapterError: For unknown source keys.
    '''
  cls = REGISTRY.get(source_key)
  if cls is None:
    raise AdapterError(f'unknown source key: {source_key}', source=source_key)
  return cls(http)
