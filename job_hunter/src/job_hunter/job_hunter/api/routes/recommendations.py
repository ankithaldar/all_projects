#!/usr/bin/env python
# -- coding: utf-8 --

'''Recommendation review endpoints.'''


from __future__ import annotations

from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException
from job_hunter.api.deps import get_settings
from job_hunter.core.config import AppSettings
from job_hunter.db.repositories.recommendations import RecommendationsRepository

router = APIRouter(prefix='/api/recommendations', tags=['recommendations'])


@router.get('')
def list_recommendations(
  min_score: float = 0.0,
  status: str = '',
  vertical: str = '',
  page: int = 1,
  per_page: int = 30,
  include_gated: bool = False,
  settings: AppSettings = Depends(get_settings),
) -> List[Dict[str, Any]]:
  '''List ranked recommendations for the candidate.

  Args:
    min_score: Minimum total score.
    status: Status filter.
    vertical: Vertical filter.
    page: Page number.
    per_page: Page size.
    include_gated: Include gate-failed rows.
    settings: App settings.

  Returns:
    Recommendation rows.
  '''
  return RecommendationsRepository(settings.db_path).list_for_candidate(
    min_score=min_score,
    status=status,
    vertical=vertical,
    page=page,
    per_page=min(per_page, 100),
    gate_only=not include_gated,
  )


@router.patch('/{recommendation_id}')
def set_status(
  recommendation_id: int,
  status: str,
  settings: AppSettings = Depends(get_settings),
) -> Dict[str, Any]:
  '''Transition a recommendation's status (saved|dismissed|applied|new).

  Args:
    recommendation_id: Row id.
    status: New status.
    settings: App settings.

  Returns:
    Acknowledgement.

  Raises:
    HTTPException: On unknown status or missing row.
  '''
  if status not in ('new', 'saved', 'dismissed', 'applied'):
    raise HTTPException(status_code=422, detail='invalid status')
  changed = RecommendationsRepository(settings.db_path).set_status(
    recommendation_id, status,
  )
  if not changed:
    raise HTTPException(status_code=404, detail='recommendation not found')
  return {'id': recommendation_id, 'status': status}
