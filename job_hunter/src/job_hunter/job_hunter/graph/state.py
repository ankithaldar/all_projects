#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''LangGraph state definitions and reducers.'''

from __future__ import annotations

import logging
import operator
from typing import Annotated, Any, Dict, List

from pydantic import ValidationError
from typing_extensions import TypedDict
from job_hunter.core.models import (
  CandidateProfile,
  NodeError,
  NormalizedJob,
  RawJobRecord,
  ScoredJob,
  TokenBudget,
)

logger = logging.getLogger(__name__)


def merge_stats(left: Dict[str, int], right: Dict[str, int]) -> Dict[str, int]:
  '''Merge counter dicts by summing values.

  Args:
    left: Existing counters.
    right: Incoming counters.

  Returns:
    Combined counters.
  '''
  combined = dict(left or {})
  for key, value in (right or {}).items():
    combined[key] = combined.get(key, 0) + int(value)
  return combined


def _normalize_inserted(
  state: Dict[str, Any],
) -> List[NormalizedJob]:
  '''Convert checkpoint-resumed dict inserted items to jobs.

  Args:
    state: Graph state possibly carrying dict items.

  Returns:
    Normalized inserted list; ``state['inserted']`` is updated
    in place when the key is present.
  '''
  raw_items = state.get('inserted') or []
  normalized: List[NormalizedJob] = []
  raw_fields = (
    'source_key',
    'external_id',
    'url',
    'company_name',
    'title',
    'location_text',
    'description_html',
    'description_text',
    'posted_at',
    'employment_type_raw',
    'work_mode_hint',
    'salary_raw',
    'extra',
  )
  normalized_fields = (
    'job_id',
    'company_id',
    'city',
    'region',
    'country',
    'work_mode',
    'employment_type',
    'salary_min_lpa',
    'salary_max_lpa',
    'is_new',
  )
  for item in raw_items:
    if isinstance(item, NormalizedJob):
      normalized.append(item)
    elif isinstance(item, dict):
      raw_value = item.get('raw')
      raw_title = (
        raw_value.get('title') if isinstance(raw_value, dict) else None
      )
      if not (
        item.get('job_id')
        or item.get('canonical_url')
        or item.get('content_hash')
        or raw_title
      ):
        logger.warning(
          'skipping inserted item without identity %r', item,
        )
        continue
      try:
        normalized.append(NormalizedJob.model_validate(item))
      except (ValueError, TypeError):  # flat shape: manual build
        try:
          raw_data = item.get('raw')
          raw_source: Dict[str, Any] = (
            raw_data if isinstance(raw_data, dict) else {}
          )
          raw_kwargs: Dict[str, Any] = {}
          for key in raw_fields:
            if key in raw_source and raw_source[key] is not None:
              raw_kwargs[key] = raw_source[key]
            elif key in item and item[key] is not None:
              raw_kwargs[key] = item[key]
          url = (
            raw_kwargs.get('url')
            or item.get('canonical_url')
            or item.get('url')
            or ''
          )
          raw_kwargs['url'] = url
          raw_kwargs['title'] = raw_kwargs.get('title') or ''
          raw_kwargs['company_name'] = raw_kwargs.get('company_name') or ''
          if not raw_kwargs.get('source_key'):
            raw_kwargs['source_key'] = 'backlog'
          raw = RawJobRecord(**raw_kwargs)
          norm_kwargs: Dict[str, Any] = {
            'raw': raw,
            'canonical_url': url,
            'content_hash': item.get('content_hash') or '',
          }
          for key in normalized_fields:
            if key in item and item[key] is not None:
              norm_kwargs[key] = item[key]
            elif key in raw_source and raw_source[key] is not None:
              norm_kwargs[key] = raw_source[key]
          if 'salary_min_lpa' not in norm_kwargs:
            if item.get('salary_min') is not None:
              norm_kwargs['salary_min_lpa'] = item.get('salary_min')
            elif isinstance(item.get('salary'), (list, tuple)):
              salary = item.get('salary')
              if salary and len(salary) >= 1:
                norm_kwargs['salary_min_lpa'] = salary[0]
          if 'salary_max_lpa' not in norm_kwargs:
            if item.get('salary_max') is not None:
              norm_kwargs['salary_max_lpa'] = item.get('salary_max')
            elif isinstance(item.get('salary'), (list, tuple)):
              salary = item.get('salary')
              if salary and len(salary) >= 2:
                norm_kwargs['salary_max_lpa'] = salary[1]
          if not norm_kwargs.get('work_mode'):
            norm_kwargs.pop('work_mode', None)
          normalized.append(NormalizedJob(**norm_kwargs))
        except (ValidationError, TypeError, ValueError) as exc:
          logger.warning(
            'skipping malformed inserted item %r: %s', item, exc,
          )
          continue
    else:
      logger.warning('skipping non-dict inserted item %r', item)
      continue
  if 'inserted' in state:
    state['inserted'] = normalized
  return normalized


def _normalize_raw_jobs(
  state: Dict[str, Any],
) -> List[RawJobRecord]:
  '''Convert checkpoint-resumed dict raw_jobs items to records.

  Args:
    state: Graph state possibly carrying dict items.

  Returns:
    Normalized raw job list; ``state['raw_jobs']`` is updated in
    place when the key is present.
  '''
  raw_items = state.get('raw_jobs') or []
  normalized: List[RawJobRecord] = []
  for item in raw_items:
    if isinstance(item, RawJobRecord):
      normalized.append(item)
    elif isinstance(item, dict):
      try:
        normalized.append(RawJobRecord.model_validate(item))
      except (ValidationError, TypeError, ValueError) as exc:
        logger.warning(
          'skipping malformed raw job item %r: %s', item, exc,
        )
        continue
    else:
      logger.warning('skipping non-dict raw job item %r', item)
      continue
  if 'raw_jobs' in state:
    state['raw_jobs'] = normalized
  return normalized


class DiscoveryState(TypedDict, total=False):
  '''State for the nightly discovery graph.'''

  run_id: int
  triggered_by: str
  candidate: CandidateProfile
  # Always dict: build_plan returns plan.model_dump() and callers
  # seed {'quick_poll': ...}.
  plan: Dict[str, Any]
  raw_jobs: Annotated[List[RawJobRecord], operator.add]
  inserted: List[NormalizedJob]
  # Always dict of plain dicts: enrich_jds stores dict values and
  # gate_failures expects a dict payload.
  enriched: Dict[str, Any]
  scored: List[ScoredJob]
  errors: Annotated[List[NodeError], operator.add]
  stats: Annotated[Dict[str, int], merge_stats]
  # Kept as TokenBudget (not always dict): initial_state seeds a
  # TokenBudget instance while enrich_jds returns a plain dict
  # (budget.model_dump()); enrich_jds accepts both shapes.
  budget: TokenBudget


def initial_state(run_id: int, triggered_by: str) -> Dict[str, Any]:
  '''Build the seed state for a run.

  Args:
    run_id: Run row id.
    triggered_by: Origin label.

  Returns:
    Partial DiscoveryState.
  '''
  return {
    'run_id': run_id,
    'triggered_by': triggered_by,
    'raw_jobs': [],
    'inserted': [],
    'enriched': {},
    'scored': [],
    'errors': [],
    'stats': {},
    'budget': TokenBudget(),
  }
