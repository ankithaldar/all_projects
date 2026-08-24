#!/usr/bin/env python
# -- coding: utf-8 --

'''LangGraph state definitions and reducers.'''


from __future__ import annotations

import operator
from typing import Annotated, Any, Dict, List

from typing_extensions import TypedDict
from job_hunter.core.models import (
  CandidateProfile,
  CompanyTarget,
  EnrichedJob,
  NodeError,
  NormalizedJob,
  RawJobRecord,
  RunPlan,
  ScoredJob,
  TokenBudget,
)


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


class DiscoveryState(TypedDict, total=False):
  '''State for the nightly discovery graph.'''

  run_id: int
  triggered_by: str
  candidate: CandidateProfile
  plan: RunPlan
  raw_jobs: Annotated[List[RawJobRecord], operator.add]
  inserted: List[NormalizedJob]
  enriched: Dict[str, EnrichedJob]
  scored: List[ScoredJob]
  errors: Annotated[List[NodeError], operator.add]
  stats: Annotated[Dict[str, int], merge_stats]
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
